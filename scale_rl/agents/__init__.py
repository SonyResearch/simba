import gymnasium as gym
from typing import TypeVar
from omegaconf import OmegaConf
from scale_rl.agents.base_agent import BaseAgent
from scale_rl.agents.random_agent import RandomAgent
from scale_rl.agents.sac.sac_agent import SACAgent
from scale_rl.agents.ddpg.ddpg_agent import DDPGAgent
from scale_rl.agents.wrappers import ObservationNormalizer

Config = TypeVar('Config')


def create_agent(
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    cfg: Config,
) -> BaseAgent:

    cfg = OmegaConf.to_container(cfg, throw_on_missing=True)
    agent_type = cfg.pop('agent_type')

    if agent_type == 'random':
        agent = RandomAgent(observation_space, action_space, cfg)

    elif agent_type == 'sac':
        agent = SACAgent(observation_space, action_space, cfg)

    elif agent_type == 'ddpg':
        agent = DDPGAgent(observation_space, action_space, cfg)

    else:
        raise NotImplementedError

    # agent-wrappers
    if cfg['normalize_observation']:
        agent = ObservationNormalizer(agent)

    return agent
