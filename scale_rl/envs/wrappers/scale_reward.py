import gymnasium as gym
import numpy as np


class ScaleReward(gym.Wrapper):
    def __init__(self, env: gym.Env, reward_scale=1.0):
        super().__init__(env)
        self._reward_scale = reward_scale

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward *= self._reward_scale

        return obs, reward, terminated, truncated, info
