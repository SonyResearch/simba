import gymnasium as gym
from dm_control import suite
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation
from shimmy import DmControlCompatibilityV0 as DmControltoGymnasium

# 20 tasks
DMC_EASY_MEDIUM = [
    "acrobot-swingup",
    "cartpole-balance",
    "cartpole-balance_sparse",
    "cartpole-swingup",
    "cartpole-swingup_sparse",
    "cheetah-run",
    "finger-spin",
    "finger-turn_easy",
    "finger-turn_hard",
    "fish-swim",
    "hopper-hop",
    "hopper-stand",
    "pendulum-swingup",
    "quadruped-walk",
    "quadruped-run",
    "reacher-easy",
    "reacher-hard",
    "walker-stand",
    "walker-walk",
    "walker-run",
]

# 8 tasks
DMC_SPARSE = [
    "cartpole-balance_sparse",
    "cartpole-swingup_sparse",
    "ball_in_cup-catch",
    "finger-spin",
    "finger-turn_easy",
    "finger-turn_hard",
    "reacher-easy",
    "reacher-hard",
]

# 7 tasks
DMC_HARD = [
    "humanoid-stand",
    "humanoid-walk",
    "humanoid-run",
    "dog-stand",
    "dog-walk",
    "dog-run",
    "dog-trot",
]


def make_dmc_env(
    env_name: str,
    seed: int,
    flatten: bool = True,
) -> gym.Env:
    domain_name, task_name = env_name.split("-")
    env = suite.load(
        domain_name=domain_name,
        task_name=task_name,
        task_kwargs={"random": seed},
    )
    env = DmControltoGymnasium(env, render_mode="rgb_array")
    if flatten and isinstance(env.observation_space, spaces.Dict):
        env = FlattenObservation(env)

    return env
