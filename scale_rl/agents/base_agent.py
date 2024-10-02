from abc import ABC, abstractmethod
from typing import Dict, TypeVar

import gymnasium as gym
import numpy as np

Config = TypeVar("Config")


class BaseAgent(ABC):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        cfg: Config,
    ):
        """
        A generic agent class.
        """
        self._observation_space = observation_space
        self._action_space = action_space
        self._cfg = cfg

    @abstractmethod
    def sample_actions(
        self,
        interaction_step: int,
        prev_timestep: Dict[str, np.ndarray],
        training: bool,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def update(self, update_step: int, batch: Dict[str, np.ndarray]) -> Dict:
        pass


class AgentWrapper(BaseAgent):
    """Wraps the agent to allow a modular transformation.

    This class is the base class for all wrappers for agent class.
    The subclass could override some methods to change the behavior of the original agent
    without touching the original code.

    Note:
        Don't forget to call ``super().__init__(env)`` if the subclass overrides :meth:`__init__`.
    """

    def __init__(self, agent: BaseAgent):
        self.agent = agent

    # explicitly forward the methods defined in Agent to self.agent
    def sample_actions(
        self,
        interaction_step: int,
        prev_timestep: Dict[str, np.ndarray],
        training: bool,
    ) -> np.ndarray:
        return self.agent.sample_actions(
            interaction_step=interaction_step,
            prev_timestep=prev_timestep,
            training=training,
        )

    def update(self, update_step: int, batch: Dict[str, np.ndarray]) -> Dict:
        return self.agent.update(
            update_step=update_step,
            batch=batch,
        )

    def set_attr(self, name, values):
        return self.agent.set_attr(name, values)

    # implicitly forward all other methods and attributes to self.env
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        """
        logger.warn(
            f"env.{name} to get variables from other wrappers is deprecated and will be removed in v1.0, "
            f"to get this variable you can do `env.unwrapped.{name}` for environment variables."
        )
        """
        return getattr(self.agent, name)

    @property
    def unwrapped(self):
        return self.agent.unwrapped

    def __repr__(self):
        return f"<{self.__class__.__name__}, {self.agent}>"
