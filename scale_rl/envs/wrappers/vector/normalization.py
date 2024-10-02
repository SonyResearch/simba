"""A collection of stateful observation wrappers.

* ``NormalizeObservation`` - Normalize the observations
"""

from __future__ import annotations

from typing import Any, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium.wrappers.utils import RunningMeanStd
from numpy.typing import NDArray

from scale_rl.envs.wrappers.vector.vector_env import (
    VectorEnv,
    VectorEnvWrapper,
    VectorObservationWrapper,
)

__all__ = ["NormalizeObservation", "SharedNormalizeObservation", "NormalizeReward"]


class NormalizeObservation(VectorObservationWrapper, gym.utils.RecordConstructorArgs):
    """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

    The property `_update_running_mean` allows to freeze/continue the running mean calculation of the observation
    statistics. If `True` (default), the `RunningMeanStd` will get updated every step and reset call.
    If `False`, the calculated statistics are used but not updated anymore; this may be used during evaluation.

    Note:
        The normalization depends on past trajectories and observations will not be normalized correctly if the wrapper was
        newly instantiated or the policy was changed recently.

    Example without the normalize reward wrapper:
        >>> import gymnasium as gym
        >>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
        >>> obs, info = envs.reset(seed=123)
        >>> _ = envs.action_space.seed(123)
        >>> for _ in range(100):
        ...     obs, *_ = envs.step(envs.action_space.sample())
        >>> np.mean(obs)
        np.float32(0.024251968)
        >>> np.std(obs)
        np.float32(0.62259156)
        >>> envs.close()

    Example with the normalize reward wrapper:
        >>> import gymnasium as gym
        >>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
        >>> envs = NormalizeObservation(envs)
        >>> obs, info = envs.reset(seed=123)
        >>> _ = envs.action_space.seed(123)
        >>> for _ in range(100):
        ...     obs, *_ = envs.step(envs.action_space.sample())
        >>> np.mean(obs)
        np.float32(-0.2359734)
        >>> np.std(obs)
        np.float32(1.1938739)
        >>> envs.close()
    """

    def __init__(self, env: VectorEnv, epsilon: float = 1e-8):
        """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

        Args:
            env (Env): The environment to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.
        """
        gym.utils.RecordConstructorArgs.__init__(self, epsilon=epsilon)
        VectorObservationWrapper.__init__(self, env)

        self.obs_rms = RunningMeanStd(
            shape=self.single_observation_space.shape,
            dtype=self.single_observation_space.dtype,
        )
        self.epsilon = epsilon
        self._update_running_mean = True

    @property
    def update_running_mean(self) -> bool:
        """Property to freeze/continue the running mean calculation of the observation statistics."""
        return self._update_running_mean

    @update_running_mean.setter
    def update_running_mean(self, setting: bool):
        """Sets the property to freeze/continue the running mean calculation of the observation statistics."""
        self._update_running_mean = setting

    def observations(self, observations: ObsType) -> ObsType:
        """Defines the vector observation normalization function.

        Args:
            observations: A vector observation from the environment

        Returns:
            the normalized observation
        """
        if self._update_running_mean:
            self.obs_rms.update(observations)
        return (observations - self.obs_rms.mean) / np.sqrt(
            self.obs_rms.var + self.epsilon
        )


class SharedNormalizeObservation(
    VectorObservationWrapper, gym.utils.RecordConstructorArgs
):
    """
    Hacky solution to share the statistics between train_env and eval_env.
    """

    _obs_rms = []

    def __init__(self, env: VectorEnv, epsilon: float = 1e-8):
        """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

        Args:
            env (Env): The environment to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.
        """
        gym.utils.RecordConstructorArgs.__init__(self, epsilon=epsilon)
        VectorObservationWrapper.__init__(self, env)

        if len(self._obs_rms) == 0:
            self._obs_rms.append(
                RunningMeanStd(
                    shape=self.single_observation_space.shape,
                    dtype=self.single_observation_space.dtype,
                )
            )
        self.epsilon = epsilon
        self._update_running_mean = True

    @property
    def obs_rms(self):
        return self._obs_rms[0]

    @property
    def update_running_mean(self) -> bool:
        """Property to freeze/continue the running mean calculation of the observation statistics."""
        return self._update_running_mean

    @update_running_mean.setter
    def update_running_mean(self, setting: bool):
        """Sets the property to freeze/continue the running mean calculation of the observation statistics."""
        self._update_running_mean = setting

    def observations(self, observations: ObsType) -> ObsType:
        """Defines the vector observation normalization function.

        Args:
            observations: A vector observation from the environment

        Returns:
            the normalized observation
        """
        if self._update_running_mean:
            self.obs_rms.update(observations)
        return (observations - self.obs_rms.mean) / np.sqrt(
            self.obs_rms.var + self.epsilon
        )


class NormalizeReward(VectorEnvWrapper, gym.utils.RecordConstructorArgs):
    r"""This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

    The exponential moving average will have variance :math:`(1 - \gamma)^2`.

    The property `_update_running_mean` allows to freeze/continue the running mean calculation of the reward
    statistics. If `True` (default), the `RunningMeanStd` will get updated every time `self.normalize()` is called.
    If False, the calculated statistics are used but not updated anymore; this may be used during evaluation.

    Note:
        The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly
        instantiated or the policy was changed recently.

    Example without the normalize reward wrapper:
        >>> import gymnasium as gym
        >>> import numpy as np
        >>> envs = gym.make_vec("MountainCarContinuous-v0", 3)
        >>> _ = envs.reset(seed=123)
        >>> _ = envs.action_space.seed(123)
        >>> episode_rewards = []
        >>> for _ in range(100):
        ...     observation, reward, *_ = envs.step(envs.action_space.sample())
        ...     episode_rewards.append(reward)
        ...
        >>> envs.close()
        >>> np.mean(episode_rewards)
        np.float64(-0.03359492141887935)
        >>> np.std(episode_rewards)
        np.float64(0.029028230434438706)

    Example with the normalize reward wrapper:
        >>> import gymnasium as gym
        >>> import numpy as np
        >>> envs = gym.make_vec("MountainCarContinuous-v0", 3)
        >>> envs = NormalizeReward(envs)
        >>> _ = envs.reset(seed=123)
        >>> _ = envs.action_space.seed(123)
        >>> episode_rewards = []
        >>> for _ in range(100):
        ...     observation, reward, *_ = envs.step(envs.action_space.sample())
        ...     episode_rewards.append(reward)
        ...
        >>> envs.close()
        >>> np.mean(episode_rewards)
        np.float64(-0.1598639586606745)
        >>> np.std(episode_rewards)
        np.float64(0.27800309628058434)
    """

    def __init__(
        self,
        env: VectorEnv,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        """This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

        Args:
            env (env): The environment to apply the wrapper
            epsilon (float): A stability parameter
            gamma (float): The discount factor that is used in the exponential moving average.
        """
        gym.utils.RecordConstructorArgs.__init__(self, gamma=gamma, epsilon=epsilon)
        VectorEnvWrapper.__init__(self, env)

        self.return_rms = RunningMeanStd(shape=())
        self.accumulated_reward: np.array = np.zeros((self.num_envs,), dtype=np.float32)
        self.gamma = gamma
        self.epsilon = epsilon
        self._update_running_mean = True

    @property
    def update_running_mean(self) -> bool:
        """Property to freeze/continue the running mean calculation of the reward statistics."""
        return self._update_running_mean

    @update_running_mean.setter
    def update_running_mean(self, setting: bool):
        """Sets the property to freeze/continue the running mean calculation of the reward statistics."""
        self._update_running_mean = setting

    def step(
        self, actions: NDArray[Any]
    ) -> Tuple[Any, NDArray[Any], NDArray[Any], NDArray[Any], dict]:
        """Steps through the environment, normalizing the reward returned."""
        obs, reward, terminated, truncated, info = super().step(actions)
        self.accumulated_reward = (
            self.accumulated_reward * self.gamma * (1 - terminated) + reward
        )
        return obs, self.normalize(reward), terminated, truncated, info

    def normalize(self, reward: SupportsFloat):
        """Normalizes the rewards with the running mean rewards and their variance."""
        if self._update_running_mean:
            self.return_rms.update(self.accumulated_reward)
        return reward / np.sqrt(self.return_rms.var + self.epsilon)
