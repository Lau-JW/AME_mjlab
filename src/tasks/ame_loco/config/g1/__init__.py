from mjlab.tasks.registry import register_mjlab_task
from .env_cfgs import g1_ame_env_cfg, g1_ame_student_env_cfg
from .rl_cfg import g1_ame_teacher_runner_cfg, g1_ame_student_runner_cfg
from src.tasks.ame_loco.rl.runner import AMEOnPolicyRunner

register_mjlab_task(
    task_id="Unitree-G1-AME-Teacher",
    env_cfg=g1_ame_env_cfg(play=False),
    play_env_cfg=g1_ame_env_cfg(play=True),
    rl_cfg=g1_ame_teacher_runner_cfg(),
    runner_cls=AMEOnPolicyRunner,
)

register_mjlab_task(
    task_id="Unitree-G1-AME-Student",
    env_cfg=g1_ame_student_env_cfg(play=False),
    play_env_cfg=g1_ame_student_env_cfg(play=True),
    rl_cfg=g1_ame_student_runner_cfg(),
    runner_cls=AMEOnPolicyRunner,
)
