import gymnasium as gym

MYOSUITE_TASKS = [
    "myo-reach",
    "myo-reach-hard",
    "myo-pose",
    "myo-pose-hard",
    "myo-obj-hold",
    "myo-obj-hold-hard",
    "myo-key-turn",
    "myo-key-turn-hard",
    "myo-pen-twirl",
    "myo-pen-twirl-hard",
]


MYOSUITE_TASKS_DICT = {
    "myo-reach": "myoHandReachFixed-v0",
    "myo-reach-hard": "myoHandReachRandom-v0",
    "myo-pose": "myoHandPoseFixed-v0",
    "myo-pose-hard": "myoHandPoseRandom-v0",
    "myo-obj-hold": "myoHandObjHoldFixed-v0",
    "myo-obj-hold-hard": "myoHandObjHoldRandom-v0",
    "myo-key-turn": "myoHandKeyTurnFixed-v0",
    "myo-key-turn-hard": "myoHandKeyTurnRandom-v0",
    "myo-pen-twirl": "myoHandPenTwirlFixed-v0",
    "myo-pen-twirl-hard": "myoHandPenTwirlRandom-v0",
}


class MyosuiteGymnasiumVersionWrapper(gym.Wrapper):
    """
    myosuite originally requires gymnasium==0.15
    however, we are currently using  gymnasium==1.0.0a2,
    hence requiring some minor fix to the
      - fix a.
      - fix b.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.unwrapped_env = env.unwrapped

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["success"] = info["solved"]
        return obs, reward, terminated, truncated, info

    def render(
        self, width: int = 192, height: int = 192, camera_id: str = "hand_side_inter"
    ):
        return self.unwrapped_env.sim.renderer.render_offscreen(
            width=width,
            height=height,
            camera_id=camera_id,
        )


def make_myosuite_env(
    env_name: str,
    seed: int,
    **kwargs,
) -> gym.Env:
    from myosuite.utils import gym as myo_gym

    env = myo_gym.make(MYOSUITE_TASKS_DICT[env_name])
    env = MyosuiteGymnasiumVersionWrapper(env)

    return env
