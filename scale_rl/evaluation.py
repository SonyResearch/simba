from typing import Dict

import gymnasium as gym
import numpy as np
import wandb
from gymnasium.vector import VectorEnv


def evaluate(
    agent,
    env: VectorEnv,
    num_episodes: int,
) -> Dict[str, float]:
    n = env.num_envs
    assert num_episodes % n == 0, "num_episodes must be divisible by env.num_envs"
    num_eval_episodes_per_env = num_episodes // n

    total_returns = []
    total_successes = []
    total_lengths = []

    for _ in range(num_eval_episodes_per_env):
        returns = np.zeros(n)
        lengths = np.zeros(n)
        successes = np.zeros(n)

        observations, infos = env.reset()

        prev_timestep = {"next_observation": observations}

        dones = np.zeros(n)
        while np.sum(dones) < n:
            actions = agent.sample_actions(
                interaction_step=0,
                prev_timestep=prev_timestep,
                training=False,
            )
            next_observations, rewards, terminateds, truncateds, infos = env.step(
                actions
            )

            prev_timestep = {"next_observation": next_observations}

            returns += rewards * (1 - dones)
            lengths += 1 - dones

            if "success" in infos:
                successes += infos["success"].astype("float") * (1 - dones)

            elif "final_info" in infos:
                final_successes = np.zeros(n)
                for idx in range(n):
                    final_info = infos["final_info"][idx]
                    if "success" in final_info:
                        final_successes[idx] = final_info["success"].astype("float")
                successes += final_successes * (1 - dones)

            else:
                pass

            # once an episode is done in a sub-environment, we assume it to be done.
            # also, we assume to be done whether it is terminated or truncated during evaluation.
            dones = np.maximum(dones, terminateds)
            dones = np.maximum(dones, truncateds)

            # proceed
            observations = next_observations

        for env_idx in range(n):
            total_returns.append(returns[env_idx])
            total_lengths.append(lengths[env_idx])
            total_successes.append(successes[env_idx].astype("bool").astype("float"))

    eval_info = {
        "avg_return": np.mean(total_returns),
        "avg_length": np.mean(total_lengths),
        "avg_success": np.mean(total_successes),
    }

    return eval_info


def record_video(
    agent,
    env: VectorEnv,
    num_episodes: int,
    video_length: int = 100,
) -> Dict[str, float]:
    n = env.num_envs
    assert num_episodes % n == 0, "num_episodes must be divisible by env.num_envs"
    num_eval_episodes_per_env = num_episodes // n

    total_videos = []

    for _ in range(num_eval_episodes_per_env):
        videos = []

        observations, infos = env.reset()
        prev_timestep = {"next_observation": observations}
        images = env.call("render")
        dones = np.zeros(n)
        while np.sum(dones) < n:
            actions = agent.sample_actions(
                interaction_step=0,
                prev_timestep=prev_timestep,
                training=False,
            )
            next_observations, rewards, terminateds, truncateds, infos = env.step(
                actions
            )

            prev_timestep = {"next_observation": next_observations}

            # once an episode is done in a sub-environment, we assume it to be done.
            dones = np.maximum(dones, terminateds)
            dones = np.maximum(dones, truncateds)

            # proceed
            videos.append(images)
            images = env.call("render")
            observations = next_observations

        total_videos.append(np.stack(videos, axis=1))  # (n, t, c, h, w)

    total_videos = np.concatenate(total_videos, axis=0)  # (b, t, h, w, c)
    total_videos = total_videos[:, :video_length]
    total_videos = total_videos.transpose(0, 1, 4, 2, 3)  # (b, t, c, h, w)

    video_info = {"video": wandb.Video(total_videos, fps=10, format="gif")}

    return video_info
