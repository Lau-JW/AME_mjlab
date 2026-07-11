# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Modifications for AME-2 student training (PPO + action distillation + rep loss).

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.algorithms import PPO


class StudentPPO(PPO):
    """PPO for AME-2 student training with teacher distillation and rep loss.

    Combines:
    - PPO losses (surrogate + value)
    - Action distillation loss (student actions vs teacher actions)
    - Representation loss (student map embedding vs teacher map embedding)

    The first ``disable_surrogate_iters`` iterations disable the PPO surrogate
    loss to align the student policy to the teacher before enabling RL.
    """

    def __init__(
        self,
        policy,
        teacher=None,
        distill_coef: float = 1.0,
        rep_coef: float = 1.0,
        disable_surrogate_iters: int = 5000,
        **kwargs,
    ):
        self.distill_coef = distill_coef
        self.rep_coef = rep_coef
        self.disable_surrogate_iters = disable_surrogate_iters
        super().__init__(policy, **kwargs)
        self.teacher = teacher
        self._teacher_set = False
        if self.teacher is not None:
            self._set_teacher(teacher)

        self.current_iteration = 0
        self.mse_loss = nn.MSELoss()

    def set_teacher(self, teacher):
        self.teacher = teacher
        self._set_teacher(teacher)

    def _set_teacher(self, teacher):
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        self._teacher_set = True

    def update(self):
        """Override PPO update to add teacher distillation and representation loss."""
        if not self._teacher_set:
            raise RuntimeError("StudentPPO teacher not set. Please load a teacher model before training.")
        self.current_iteration += 1
        use_surrogate = self.current_iteration > self.disable_surrogate_iters
        if not use_surrogate:
            print(f"[StudentPPO] Iteration {self.current_iteration}: surrogate loss disabled")

        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_distill_loss = 0
        mean_rep_loss = 0
        mean_vq_loss = 0
        mean_recon_loss = 0
        mean_symmetry_loss = 0
        mean_rnd_loss = 0

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            rnd_state_batch,
        ) in generator:
            original_batch_size = obs_batch.shape[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"], obs_type="policy"
                )
                critic_obs_batch, _ = data_augmentation_func(
                    obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="critic"
                )
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)
            else:
                num_aug = 1

            # -- actor forward (student) --
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # -- teacher distillation --
            with torch.no_grad():
                teacher_actions, teacher_map_embed = self.teacher.forward_actor_with_map_embed(critic_obs_batch)
            student_actions, student_map_embed = self.policy.actor.forward_actor_with_map_embed(obs_batch)
            distill_loss = self.mse_loss(student_actions, teacher_actions)
            rep_loss = self.mse_loss(student_map_embed, teacher_map_embed)

            # -- critic forward --
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])

            # KL adaptive LR
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss (disabled first N iterations)
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            if not use_surrogate:
                surrogate_loss = surrogate_loss.detach() * 0.0

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            vq_loss = getattr(getattr(self.policy, "actor", None), "vq_loss", None)
            if vq_loss is None:
                vq_loss = torch.tensor(0.0, device=self.device)
            elif not torch.is_tensor(vq_loss):
                vq_loss = torch.tensor(float(vq_loss), device=self.device)

            recon_loss = getattr(getattr(self.policy, "actor", None), "recon_loss", None)
            if recon_loss is None:
                recon_loss = torch.tensor(0.0, device=self.device)
            elif not torch.is_tensor(recon_loss):
                recon_loss = torch.tensor(float(recon_loss), device=self.device)

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + self.vq_loss_coef * vq_loss
                + self.recon_loss_coef * recon_loss
                + self.distill_coef * distill_loss
                + self.rep_coef * rep_loss
            )

            # Symmetry loss
            symmetry_loss = None
            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(
                        obs=obs_batch, actions=None, env=self.symmetry["_env"], obs_type="policy"
                    )
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"], obs_type="policy"
                )
                mse_loss = nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND loss
            rnd_loss = None
            if self.rnd:
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                mseloss = nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_distill_loss += distill_loss.item()
            mean_rep_loss += rep_loss.item()
            mean_vq_loss += vq_loss.item()
            mean_recon_loss += recon_loss.item()
            if symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
            if rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_distill_loss /= num_updates
        mean_rep_loss /= num_updates
        mean_vq_loss /= num_updates
        mean_recon_loss /= num_updates

        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "vq": mean_vq_loss,
            "recon": mean_recon_loss,
            "distill": mean_distill_loss,
            "rep": mean_rep_loss,
        }
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss / num_updates
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss / num_updates

        return loss_dict
