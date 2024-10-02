import gymnasium as gym
import numpy as np


class DoNotTerminate(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        terminated = False

        return obs, reward, terminated, truncated, info
