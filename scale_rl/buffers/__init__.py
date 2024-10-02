import gymnasium as gym
from typing import Tuple, Optional
from scale_rl.buffers.base_buffer import BaseBuffer, Batch
from scale_rl.buffers.numpy_buffer import NpyUniformBuffer, NpyPrioritizedBuffer


def create_buffer(
    buffer_class_type: str,
    buffer_type: str,
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    n_step: int,
    gamma: float,
    max_length: int,
    min_length: int,
    add_batch_size: int,
    sample_batch_size: int,
    **kwargs,
) -> BaseBuffer:

    if buffer_class_type == 'numpy':
        if buffer_type == 'uniform':
            buffer = NpyUniformBuffer(
                observation_space=observation_space,
                action_space=action_space,
                n_step=n_step,
                gamma=gamma,
                max_length=max_length,
                min_length=min_length, 
                add_batch_size=add_batch_size, 
                sample_batch_size=sample_batch_size,
            )

        else:
            raise NotImplementedError

    elif buffer_class_type == 'jax':
        raise NotImplementedError

    return buffer
