import gymnasium as gym
from typing import Tuple
from gymnasium.vector import VectorEnv
from gymnasium.wrappers import RescaleAction, TimeLimit

from scale_rl.envs.dmc import make_dmc_env
from scale_rl.envs.mujoco import make_mujoco_env
from scale_rl.envs.humanoid_bench import make_humanoid_env
from scale_rl.envs.myosuite import make_myosuite_env

from scale_rl.envs.wrappers import RepeatAction, ScaleReward, DoNotTerminate
from scale_rl.envs.wrappers.vector import AsyncVectorEnv, SyncVectorEnv, VectorEnv


def create_envs(
    env_type: str,
    seed: int,
    env_name: str,
    num_train_envs: int,
    num_eval_envs: int,
    rescale_action: bool,
    no_termination: bool,
    action_repeat: int,
    reward_scale: float,
    max_episode_steps: int,
    **kwargs,
)-> Tuple[VectorEnv, VectorEnv]:
    
    train_env = create_vec_env(
        env_type=env_type,
        env_name=env_name,
        seed=seed,
        num_envs=num_train_envs,
        no_termination=no_termination,
        action_repeat=action_repeat,
        rescale_action=rescale_action,
        reward_scale=reward_scale,
        max_episode_steps=max_episode_steps,
    )
    eval_env = create_vec_env(
        env_type=env_type,
        env_name=env_name,
        seed=seed,
        num_envs=num_eval_envs,
        no_termination=no_termination,
        action_repeat=action_repeat,
        rescale_action=rescale_action,
        reward_scale=1.0, # reward scale is not applied on eval.
        max_episode_steps=max_episode_steps,
    )
    
    return train_env, eval_env


def create_vec_env(
    env_type: str,
    env_name: str,
    num_envs: int,
    seed: int,
    rescale_action: bool = True,
    no_termination: bool = False,
    action_repeat: int = 1,
    reward_scale: float = 1.0,
    max_episode_steps: int = 1000,
) -> VectorEnv:
    
    def make_one_env(
        env_type: str,
        env_name:str, 
        seed:int, 
        rescale_action:bool, 
        no_termination:bool,
        action_repeat:int, 
        reward_scale: float,
        max_episode_steps: int,
        **kwargs
    ) -> gym.Env:
        
        if env_type == 'dmc':
            env = make_dmc_env(env_name, seed, **kwargs)
        elif env_type == 'mujoco':
            env = make_mujoco_env(env_name, seed, **kwargs)
        elif env_type == 'humanoid_bench':
            env = make_humanoid_env(env_name, seed, **kwargs)
        elif env_type == 'myosuite':
            env = make_myosuite_env(env_name, seed, **kwargs)
        else:
            raise NotImplementedError

        if rescale_action:
            env = RescaleAction(env, -1.0, 1.0)

        if no_termination:
            env = DoNotTerminate(env)

        # limit max_steps before action_repeat.
        env = TimeLimit(env, max_episode_steps)

        if action_repeat > 1:
            env = RepeatAction(env, action_repeat)

        env = ScaleReward(env, reward_scale)

        env.observation_space.seed(seed)
        env.action_space.seed(seed)
        
        return env
    
    env_fns = [
        (
            lambda i=i: make_one_env(
                env_type=env_type,
                env_name=env_name,
                seed=seed + i,
                no_termination=no_termination,
                rescale_action=rescale_action,
                action_repeat=action_repeat,
                reward_scale=reward_scale,
                max_episode_steps=max_episode_steps,
            )
        )
        for i in range(num_envs)
    ]
    if len(env_fns) > 1:
        envs = AsyncVectorEnv(env_fns)
    else:
        envs = SyncVectorEnv(env_fns)

    return envs
