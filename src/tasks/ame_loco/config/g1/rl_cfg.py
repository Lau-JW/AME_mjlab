"""RL configuration for G1 AME-2 training.

Paper: Appendix C, Table VI
"""

from dataclasses import dataclass, field
from typing import List

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass
class AmeRunnerCfg(RslRlOnPolicyRunnerCfg):
    use_moe_critic: bool = False
    """Use MoE (Mixture-of-Experts) for critic network (Sec IV-B)."""
    student_mode: bool = False
    student_history_length: int = 20
    student_command_dim: int = 3
    distill_loss_coef: float = 1.0
    rep_loss_coef: float = 0.1
    surrogate_disable_iters: int = 5000


def g1_ame_teacher_runner_cfg() -> AmeRunnerCfg:
    """Teacher PPO config (Table VI)."""
    return AmeRunnerCfg(
        use_moe_critic=True,
        student_mode=False,
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),  # MoE replaces this
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.004,  # decays to 0.001
            num_learning_epochs=4,
            num_mini_batches=3,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="g1_ame_teacher",
        logger="tensorboard",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=80000,
    )


def g1_ame_student_runner_cfg() -> AmeRunnerCfg:
    """Student training config (Phase-1).

    PPO surrogate disabled for first 5k iterations.
    Action distillation uses recon_loss_coef; map-embed alignment uses vq_loss_coef.
    """
    return AmeRunnerCfg(
        use_moe_critic=True,
        student_mode=True,
        student_history_length=20,
        student_command_dim=3,
        distill_loss_coef=1.0,
        rep_loss_coef=0.1,
        surrogate_disable_iters=5000,
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.004,
            num_learning_epochs=4,
            num_mini_batches=3,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="g1_ame_student",
        logger="tensorboard",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=40001,
    )
