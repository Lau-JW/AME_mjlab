"""AME-2 runner with AME-2 encoder actor + optional MoE critic.

- Actor: ProprioEncoder + AME2Encoder (CNN+MHA) + MLP decoder (paper Fig 3)
- Critic: standard MLP (can be replaced with MoE)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

from rsl_rl.runners.on_policy_runner import _unpack_obs
from rsl_rl.modules.normalizer import EmpiricalNormalization
from src.tasks.ame_loco.rl.ame_encoder import (
    ProprioEncoder, AME2Encoder, SimpleMapEncoder, MoECritic, LSIOProprioEncoder,
)


class AME2Actor(nn.Module):
    """AME-2 actor: proprio → encoder + map → AME2 → concat → MLP decoder.

    Wraps the full pipeline so it can replace ActorCritic.actor.
    Input: observation tensor (B, proprio + map_flat)
    Output: action mean (B, num_actions)
    """

    def __init__(self, num_actions=29, proprio_dim=96,
                 map_channels=3, map_height=18, map_width=7,
                 proprio_hidden=128, encoder_proprio_dim=64,
                 local_feat_dim=64, global_feat_dim=64,
                 decoder_hidden=(512, 256, 128)):
        super().__init__()
        self.proprio_dim = proprio_dim
        self.map_channels = map_channels
        self.map_height = map_height
        self.map_width = map_width
        self.map_embed_dim = global_feat_dim + local_feat_dim
        self.last_map_embed = None
        self.distill_loss = None
        self.rep_loss = None

        self.proprio_encoder = ProprioEncoder(
            input_dim=proprio_dim, hidden_dim=proprio_hidden,
            output_dim=encoder_proprio_dim,
        )
        self.map_encoder = AME2Encoder(
            map_channels=map_channels, map_height=map_height, map_width=map_width,
            local_feat_dim=local_feat_dim, global_feat_dim=global_feat_dim,
            proprio_dim=encoder_proprio_dim,
        )
        decoder_in = encoder_proprio_dim + global_feat_dim + local_feat_dim
        layers = []
        prev = decoder_in
        for h in decoder_hidden:
            layers.extend([nn.Linear(prev, h), nn.ELU()])
            prev = h
        layers.append(nn.Linear(prev, num_actions))
        self.decoder = nn.Sequential(*layers)

    def forward_with_embed(self, obs):
        proprio = obs[:, :self.proprio_dim]
        map_flat = obs[:, self.proprio_dim:]
        elev_map = map_flat.view(-1, self.map_channels, self.map_height, self.map_width)
        prop_embed = self.proprio_encoder(proprio)
        map_embed = self.map_encoder(elev_map, prop_embed)
        combined = torch.cat([prop_embed, map_embed], dim=-1)
        mean = self.decoder(combined)
        self.last_map_embed = map_embed
        return mean, map_embed

    def forward(self, obs):
        mean, _ = self.forward_with_embed(obs)
        return mean


class StudentAME2Actor(nn.Module):
    """Student AME-2 actor: LSIO proprio history + 4ch map.

    Observation layout from mjlab per-term history (flatten):
      ang_vel(T*3) | gravity(T*3) | joint_pos(T*29) | joint_vel(T*29) |
      actions(T*29) | command(3) | map_4ch(4*H*W)
    """

    def __init__(
        self,
        num_actions=29,
        history_length=20,
        command_dim=3,
        map_channels=4,
        map_height=18,
        map_width=13,
        encoder_proprio_dim=64,
        local_feat_dim=64,
        global_feat_dim=64,
        decoder_hidden=(512, 256, 128),
        ang_dim=3,
        grav_dim=3,
        joint_dim=29,
        action_dim=29,
    ):
        super().__init__()
        self.history_length = history_length
        self.command_dim = command_dim
        self.ang_dim = ang_dim
        self.grav_dim = grav_dim
        self.joint_dim = joint_dim
        self.action_dim = action_dim
        self.frame_dim = ang_dim + grav_dim + joint_dim + joint_dim + action_dim
        self.hist_ang = history_length * ang_dim
        self.hist_grav = history_length * grav_dim
        self.hist_jp = history_length * joint_dim
        self.hist_jv = history_length * joint_dim
        self.hist_act = history_length * action_dim
        self.history_dim = (
            self.hist_ang + self.hist_grav + self.hist_jp + self.hist_jv + self.hist_act
        )
        self.proprio_dim = self.history_dim + command_dim
        self.map_channels = map_channels
        self.map_height = map_height
        self.map_width = map_width
        self.map_embed_dim = global_feat_dim + local_feat_dim
        self.last_map_embed = None
        self.vq_loss = None
        self.recon_loss = None

        self.proprio_encoder = LSIOProprioEncoder(
            frame_dim=self.frame_dim,
            history_length=history_length,
            command_dim=command_dim,
            output_dim=encoder_proprio_dim,
        )
        self.map_encoder = AME2Encoder(
            map_channels=map_channels,
            map_height=map_height,
            map_width=map_width,
            local_feat_dim=local_feat_dim,
            global_feat_dim=global_feat_dim,
            proprio_dim=encoder_proprio_dim,
        )
        decoder_in = encoder_proprio_dim + global_feat_dim + local_feat_dim
        layers = []
        prev = decoder_in
        for h in decoder_hidden:
            layers.extend([nn.Linear(prev, h), nn.ELU()])
            prev = h
        layers.append(nn.Linear(prev, num_actions))
        self.decoder = nn.Sequential(*layers)

    def _assemble_history(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = obs.shape[0]
        T = self.history_length
        i = 0
        ang = obs[:, i:i + self.hist_ang].view(B, T, self.ang_dim); i += self.hist_ang
        grav = obs[:, i:i + self.hist_grav].view(B, T, self.grav_dim); i += self.hist_grav
        jp = obs[:, i:i + self.hist_jp].view(B, T, self.joint_dim); i += self.hist_jp
        jv = obs[:, i:i + self.hist_jv].view(B, T, self.joint_dim); i += self.hist_jv
        act = obs[:, i:i + self.hist_act].view(B, T, self.action_dim); i += self.hist_act
        cmd = obs[:, i:i + self.command_dim]; i += self.command_dim
        hist = torch.cat([ang, grav, jp, jv, act], dim=-1).reshape(B, T * self.frame_dim)
        return hist, cmd

    def forward_with_embed(self, obs):
        hist, cmd = self._assemble_history(obs)
        map_flat = obs[:, self.proprio_dim:]
        elev_map = map_flat.view(-1, self.map_channels, self.map_height, self.map_width)
        prop_embed = self.proprio_encoder(hist, cmd)
        map_embed = self.map_encoder(elev_map, prop_embed)
        mean = self.decoder(torch.cat([prop_embed, map_embed], dim=-1))
        self.last_map_embed = map_embed
        return mean, map_embed

    def forward(self, obs):
        mean, _ = self.forward_with_embed(obs)
        return mean



class AMEOnPolicyRunner(MjlabOnPolicyRunner):
    """On-policy runner with AME-2 encoder actor + optional MoE critic."""

    env: RslRlVecEnvWrapper

    def __init__(self, env, cfg, log_dir=None, device="cuda:0", **kwargs):
        super().__init__(env, cfg, log_dir, device, **kwargs)
        # Immediately replace actor after parent init completes
        if hasattr(self, "alg") and hasattr(self.alg, "policy"):
            self._install_ame2_actor()
        else:
            print("[AME] WARNING: alg/policy not available after init, skipping AME-2")

    def _install_ame2_actor(self):
        """Replace actor with AME-2 encoder, add SimpleMapEncoder for critic."""
        policy = self.alg.policy

        # Get actual observation dimensions from env
        obs, extras = _unpack_obs(self.env.get_observations())
        num_actor_obs = obs.shape[1]
        num_critic_obs = extras["observations"].get("critic", obs).shape[1]

        # Infer proprio vs map split from elevation_map observation params.
        student_mode = bool(self.cfg.get("student_mode", False))
        map_channels = 4 if student_mode else 3
        critic_map_channels = 3
        try:
            elev_term = self.env.env.cfg.observations["actor"].terms.get("elevation_map", None)
            if elev_term:
                map_h = elev_term.params.get("map_height", 18)
                map_w = elev_term.params.get("map_width", 13)
            else:
                map_h, map_w = 18, 13
            crit_elev = self.env.env.cfg.observations["critic"].terms.get("elevation_map", None)
            if crit_elev:
                critic_map_h = crit_elev.params.get("map_height", map_h)
                critic_map_w = crit_elev.params.get("map_width", map_w)
            else:
                critic_map_h, critic_map_w = map_h, map_w
        except Exception:
            map_h, map_w = 18, 13
            critic_map_h, critic_map_w = map_h, map_w

        map_flat_dim = map_channels * map_h * map_w
        critic_map_flat_dim = critic_map_channels * critic_map_h * critic_map_w
        actor_proprio_dim = num_actor_obs - map_flat_dim
        critic_proprio_dim = num_critic_obs - critic_map_flat_dim

        print(f"[AME] Obs dims: actor={num_actor_obs}, critic={num_critic_obs}, "
              f"proprio={actor_proprio_dim}/{critic_proprio_dim}, "
              f"map={map_h}x{map_w}x{map_channels}, critic_map={critic_map_h}x{critic_map_w}x{critic_map_channels}")

        # ── Actor ──
        num_actions = self.env.num_actions
        if student_mode or map_channels == 4:
            hist = int(self.cfg.get("student_history_length", 20))
            ame_actor = StudentAME2Actor(
                num_actions=num_actions,
                history_length=hist,
                command_dim=int(self.cfg.get("student_command_dim", 3)),
                map_channels=map_channels,
                map_height=map_h,
                map_width=map_w,
            ).to(self.device)
            print(f"[AME] Student LSIO actor: history={hist}, proprio_dim={ame_actor.proprio_dim}")
        else:
            ame_actor = AME2Actor(num_actions=num_actions,
                                  proprio_dim=actor_proprio_dim,
                                  map_channels=map_channels,
                                  map_height=map_h, map_width=map_w).to(self.device)
        policy.actor = ame_actor
        params = sum(p.numel() for p in ame_actor.parameters())
        print(f"[AME] AME-2 encoder actor installed: {params} params")
        print(f"[AME] AME-2 architecture:\n{ame_actor}")

        # ── Critic: wrap with SimpleMapEncoder (CNN downsample, no MHA) ──
        # We insert a CNN encoder before the MLP to reduce map dimensionality.
        orig_critic = policy.critic
        use_moe = self.cfg.get("use_moe_critic", False)

        class CriticWithMapEncoder(nn.Module):
            def __init__(self, proprio_dim, map_c, map_h, map_w, orig_mlp, use_moe_critic=False):
                super().__init__()
                self.proprio_dim = proprio_dim
                self.map_c = map_c
                self.map_h = map_h
                self.map_w = map_w
                self.map_encoder = SimpleMapEncoder(
                    map_channels=map_c, map_height=map_h, map_width=map_w,
                    output_dim=64,
                )
                new_in = proprio_dim + 64
                if use_moe_critic:
                    self.mlp = MoECritic(input_dim=new_in, num_experts=8, expert_hidden=256)
                else:
                    new_first = nn.Linear(new_in, orig_mlp[0].out_features)
                    remaining = orig_mlp[1:]
                    self.mlp = nn.Sequential(new_first, *remaining)

            def forward(self, obs):
                proprio = obs[:, :self.proprio_dim]
                map_flat = obs[:, self.proprio_dim:]
                elev_map = map_flat.view(-1, self.map_c, self.map_h, self.map_w)
                map_feat = self.map_encoder(elev_map)
                combined = torch.cat([proprio, map_feat], dim=-1)
                return self.mlp(combined)

        policy.critic = CriticWithMapEncoder(
            critic_proprio_dim, critic_map_channels, critic_map_h, critic_map_w, orig_critic, use_moe
        ).to(self.device)
        c_params = sum(p.numel() for p in policy.critic.parameters())
        critic_name = "MoE critic with SimpleMapEncoder" if use_moe else "Critic with SimpleMapEncoder"
        print(f"[AME] {critic_name} installed: {c_params} params")

        # Parent runner created the PPO optimizer before we replaced modules.
        # Rebuild it so AME actor/critic parameters are actually trained.
        self.alg.optimizer = optim.Adam(policy.parameters(), lr=self.alg.learning_rate)

    def save(self, path: str, infos=None) -> None:
        """Save model, optimizer, and environment curriculum state."""
        env = self.env.unwrapped
        env_state = {
            "common_step_counter": env.common_step_counter,
            "sim_step_counter": env._sim_step_counter,
        }
        terrain = env.scene.terrain
        if terrain is not None and terrain.terrain_origins is not None:
            env_state["terrain_levels"] = terrain.terrain_levels.detach().cpu()
            env_state["terrain_types"] = terrain.terrain_types.detach().cpu()

        curriculum_state = {}
        for term_name in env.curriculum_manager.active_terms:
            term = env.curriculum_manager.get_term_cfg(term_name).func
            if hasattr(term, "state_dict"):
                curriculum_state[term_name] = term.state_dict()
        if curriculum_state:
            env_state["curriculum"] = curriculum_state

        infos = {**(infos or {}), "env_state": env_state}
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["privileged_obs_norm_state_dict"] = (
                self.privileged_obs_normalizer.state_dict()
            )
        torch.save(saved_dict, path)
        # MjlabOnPolicyRunner.save() calls self.logger.save_model, but our
        # rsl_rl fork doesn't set up self.logger. Skip if not available.
        if hasattr(self, "logger") and self.cfg.get("upload_model", False):
            self.logger.save_model(path, self.current_learning_iteration)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        infos = super().load(
            path,
            load_cfg=load_cfg,
            strict=strict,
            map_location=map_location,
        )
        loaded_dict = torch.load(path, map_location=map_location, weights_only=False)
        if self.empirical_normalization:
            if "obs_norm_state_dict" in loaded_dict:
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            else:
                print(
                    "[AME] WARNING: checkpoint has no obs normalizer state; "
                    "play/resume may fail for policies trained with obs_normalization=True."
                )
            if "privileged_obs_norm_state_dict" in loaded_dict:
                self.privileged_obs_normalizer.load_state_dict(
                    loaded_dict["privileged_obs_norm_state_dict"]
                )
        env_state = (infos or {}).get("env_state", {})
        if not env_state:
            return infos

        env = self.env.unwrapped
        if "sim_step_counter" in env_state:
            env._sim_step_counter = int(env_state["sim_step_counter"])

        curriculum_state = env_state.get("curriculum", {})
        for term_name, state in curriculum_state.items():
            if term_name not in env.curriculum_manager.active_terms:
                continue
            term = env.curriculum_manager.get_term_cfg(term_name).func
            if hasattr(term, "load_state_dict"):
                term.load_state_dict(state)

        terrain = env.scene.terrain
        levels = env_state.get("terrain_levels")
        types = env_state.get("terrain_types")
        terrain_restored = False
        if (
            terrain is not None
            and terrain.terrain_origins is not None
            and levels is not None
            and types is not None
        ):
            if (
                levels.shape == terrain.terrain_levels.shape
                and types.shape == terrain.terrain_types.shape
            ):
                terrain.terrain_levels.copy_(levels.to(terrain.terrain_levels.device))
                terrain.terrain_types.copy_(types.to(terrain.terrain_types.device))
                terrain.env_origins[:] = terrain.terrain_origins[
                    terrain.terrain_levels, terrain.terrain_types
                ]
                terrain_restored = True
            else:
                print(
                    "[AME] Checkpoint terrain state skipped because num_envs changed: "
                    f"checkpoint={tuple(levels.shape)}, "
                    f"environment={tuple(terrain.terrain_levels.shape)}"
                )

        if terrain_restored:
            for term_name in env.curriculum_manager.active_terms:
                term = env.curriculum_manager.get_term_cfg(term_name).func
                if hasattr(term, "suspend_next_update"):
                    term.suspend_next_update()
            self.env.reset()
            print(
                "[AME] Restored terrain curriculum: "
                f"mean_level={terrain.terrain_levels.float().mean().item():.3f}"
            )
        return infos

    def _inject_terrain_curriculum(self, locs: dict):
        """Inject per-terrain curriculum levels into ep_infos so parent log prints them."""
        try:
            env = self.env.env if hasattr(self.env, 'env') else self.env
            terrain = env.scene.terrain
            col_types = terrain.terrain_types
            levels = terrain.terrain_levels.float()
            num_cols = terrain.terrain_origins.shape[1]
            success_ema = None
            if "terrain_levels" in env.curriculum_manager.active_terms:
                curriculum_term = env.curriculum_manager.get_term_cfg(
                    "terrain_levels"
                ).func
                if hasattr(curriculum_term, "success_ema"):
                    success_ema = curriculum_term.success_ema.mean().item()

            # Build column→name mapping from terrain generator proportions (once)
            if not hasattr(self, '_terrain_col_names'):
                from src.tasks.ame_loco.mdp.terrain import SIMPLE_TERRAINS_CFG
                sub_cfgs = list(SIMPLE_TERRAINS_CFG.sub_terrains.values())
                sub_keys = list(SIMPLE_TERRAINS_CFG.sub_terrains.keys())
                props = np.array([s.proportion for s in sub_cfgs])
                props /= props.sum()
                cumsum = np.cumsum(props)
                self._terrain_col_names = {}
                for c in range(num_cols):
                    idx = np.min(np.where(c / num_cols + 0.001 < cumsum)[0])
                    self._terrain_col_names[c] = sub_keys[idx]

            # Group envs by sub-terrain name
            import collections
            groups = collections.defaultdict(list)
            for i in range(env.num_envs):
                name = self._terrain_col_names[col_types[i].item()]
                groups[name].append(levels[i].item())

            # Inject into ep_infos: add entries right after Curriculum/terrain_levels
            for ep_info in locs.get("ep_infos", []):
                # Build new dict with terrain entries inserted after Curriculum/
                new_ep = {}
                for k, v in ep_info.items():
                    new_ep[k] = v
                    if k == "Curriculum/terrain_levels":
                        if success_ema is not None:
                            new_ep["Curriculum/success_ema"] = success_ema
                        for name in sorted(groups.keys()):
                            vals = groups[name]
                            if vals:
                                display = name.replace("_", " ").title()
                                new_ep[f"Curriculum/{display}"] = sum(vals) / len(vals)
                # Replace ep_info contents
                ep_info.clear()
                ep_info.update(new_ep)
        except Exception as e:
            print(f"[AME] terrain injection error: {type(e).__name__}: {e}")

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        """Override: inject terrain data, then call parent log."""
        self._inject_terrain_curriculum(locs)
        super().log(locs, width, pad)

    def learn(self, num_learning_iterations, init_at_random_ep_len=True):
        return super().learn(num_learning_iterations, init_at_random_ep_len)


def _critic_batch_to_teacher_obs(critic_obs: torch.Tensor) -> torch.Tensor:
    """Slice student critic obs into teacher actor obs (3ch GT map)."""
    # lin3|ang3|grav3|cmd4|jp29|jv29|act29|map702|contact...
    lin = critic_obs[:, 0:3]
    ang = critic_obs[:, 3:6]
    grav = critic_obs[:, 6:9]
    cmd = critic_obs[:, 9:12]  # drop t_left
    jp = critic_obs[:, 13:42]
    jv = critic_obs[:, 42:71]
    act = critic_obs[:, 71:100]
    elev = critic_obs[:, 100:802]
    return torch.cat([lin, ang, grav, cmd, jp, jv, act, elev], dim=-1)


class AMEStudentOnPolicyRunner(AMEOnPolicyRunner):
    """Student runner: LSIO+4ch actor, frozen teacher distill, 5k surrogate off."""

    def __init__(self, env, cfg, log_dir=None, device="cuda:0", **kwargs):
        cfg = dict(cfg)
        cfg["student_mode"] = True
        cfg.setdefault("student_history_length", 20)
        cfg.setdefault("student_command_dim", 3)
        cfg.setdefault("distill_loss_coef", 1.0)
        cfg.setdefault("rep_loss_coef", 0.1)
        cfg.setdefault("surrogate_disable_iters", 5000)
        self._teacher_checkpoint = kwargs.pop("teacher_checkpoint", None)
        self._frozen_teacher = None
        self._teacher_obs_normalizer = None
        super().__init__(env, cfg, log_dir, device, **kwargs)
        self.alg.surrogate_coef = 1.0
        self.alg.vq_loss_coef = float(cfg.get("rep_loss_coef", 0.1))
        self.alg.recon_loss_coef = float(cfg.get("distill_loss_coef", 1.0))
        if self._teacher_checkpoint:
            self._load_frozen_teacher(self._teacher_checkpoint)
        self._wrap_update_for_distill()

    def _load_frozen_teacher(self, path: str) -> None:
        num_actions = self.env.num_actions
        teacher = AME2Actor(
            num_actions=num_actions,
            proprio_dim=99,
            map_channels=3,
            map_height=18,
            map_width=13,
        ).to(self.device)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        raw = ckpt.get("actor_state_dict") or ckpt.get("model_state_dict") or ckpt
        actor_state = {}
        for k, v in raw.items():
            if k.startswith("distribution."):
                continue
            if k.startswith("actor."):
                k = k[len("actor."):]
            if k.startswith("mlp."):
                k = k[len("mlp."):]
            actor_state[k] = v
        missing, unexpected = teacher.load_state_dict(actor_state, strict=False)
        print(f"[AME] Frozen teacher loaded from {path}")
        if missing:
            print(f"[AME] Teacher missing keys: {list(missing)[:8]}")
        if unexpected:
            print(f"[AME] Teacher unexpected keys: {list(unexpected)[:8]}")
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        self._frozen_teacher = teacher

        # Teacher actor obs normalizer (801-d). Required because PPO update batches are
        # student-normalized; we inverse student critic norm then apply teacher norm.
        teacher_obs_dim = 99 + 3 * 18 * 13
        self._teacher_obs_normalizer = EmpiricalNormalization(
            shape=[teacher_obs_dim], until=1.0e8
        ).to(self.device)
        if "obs_norm_state_dict" in ckpt:
            self._teacher_obs_normalizer.load_state_dict(ckpt["obs_norm_state_dict"])
            self._teacher_obs_normalizer.eval()
            print(f"[AME] Teacher obs normalizer loaded (dim={teacher_obs_dim})")
        else:
            print("[AME] WARNING: teacher ckpt has no obs_norm_state_dict; using identity scale")

    def _wrap_update_for_distill(self) -> None:
        orig_update = self.alg.update
        runner = self
        # Remember configured entropy so we can restore after warm-start.
        base_entropy_coef = float(runner.alg.entropy_coef)

        def update_with_distill():
            it = runner.current_learning_iteration
            disable_n = int(runner.cfg.get("surrogate_disable_iters", 5000))
            warm = it < disable_n
            runner.alg.surrogate_coef = 0.0 if warm else 1.0
            # Entropy bonus fights distillation during warm-start (std was climbing).
            runner.alg.entropy_coef = 0.0 if warm else base_entropy_coef

            actor = runner.alg.policy.actor
            teacher = runner._frozen_teacher
            if teacher is None:
                return orig_update()

            orig_forward = actor.forward

            def forward_with_losses(obs):
                mean, s_embed = actor.forward_with_embed(obs)
                actor._pending_mean = mean
                actor._pending_embed = s_embed
                actor.recon_loss = obs.new_zeros(())
                actor.vq_loss = obs.new_zeros(())
                return mean

            orig_evaluate = runner.alg.policy.evaluate

            def evaluate_with_distill(critic_obs, **kwargs):
                values = orig_evaluate(critic_obs, **kwargs)
                if (
                    teacher is not None
                    and getattr(actor, "_pending_mean", None) is not None
                    and actor._pending_mean.shape[0] == critic_obs.shape[0]
                ):
                    with torch.no_grad():
                        # critic_obs in the PPO batch is student-normalized.
                        if hasattr(runner, "privileged_obs_normalizer"):
                            raw_critic = runner.privileged_obs_normalizer.inverse(critic_obs)
                        else:
                            raw_critic = critic_obs
                        t_raw = _critic_batch_to_teacher_obs(raw_critic)
                        if runner._teacher_obs_normalizer is not None:
                            t_obs = runner._teacher_obs_normalizer(t_raw)
                        else:
                            t_obs = t_raw
                        t_mean, t_embed = teacher.forward_with_embed(t_obs)
                    actor.recon_loss = torch.nn.functional.mse_loss(
                        actor._pending_mean, t_mean
                    )
                    actor.vq_loss = torch.nn.functional.mse_loss(
                        actor._pending_embed, t_embed
                    )
                return values

            actor.forward = forward_with_losses
            runner.alg.policy.evaluate = evaluate_with_distill
            try:
                return orig_update()
            finally:
                actor.forward = orig_forward
                runner.alg.policy.evaluate = orig_evaluate
                actor._pending_mean = None
                actor._pending_embed = None

        self.alg.update = update_with_distill
