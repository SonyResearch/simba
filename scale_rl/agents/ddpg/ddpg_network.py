from typing import Any

import flax.linen as nn
import jax.numpy as jnp
from jax.lax import convert_element_type

from scale_rl.networks.critics import LinearCritic
from scale_rl.networks.layers import MLPBlock, ResidualBlock
from scale_rl.networks.policies import TanhPolicy
from scale_rl.networks.utils import orthogonal_init


# Reference: https://arxiv.org/pdf/2002.04745
class DDPGEncoder(nn.Module):
    block_type: str
    num_blocks: int
    hidden_dim: int
    dtype: Any

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.block_type == "mlp":
            x = MLPBlock(self.hidden_dim, dtype=self.dtype)(x)

        elif self.block_type == "residual":
            x = nn.Dense(
                self.hidden_dim, kernel_init=orthogonal_init(1), dtype=self.dtype
            )(x)
            for _ in range(self.num_blocks):
                x = ResidualBlock(self.hidden_dim, dtype=self.dtype)(x)
            x = nn.LayerNorm(dtype=self.dtype)(x)

        return x


class DDPGActor(nn.Module):
    block_type: str
    num_blocks: int
    hidden_dim: int
    action_dim: int
    dtype: Any

    def setup(self):
        self.encoder = DDPGEncoder(
            block_type=self.block_type,
            num_blocks=self.num_blocks,
            hidden_dim=self.hidden_dim,
            dtype=self.dtype,
        )
        self.predictor = TanhPolicy(self.action_dim)

    def __call__(
        self,
        observations: jnp.ndarray,
    ) -> jnp.ndarray:
        observations = convert_element_type(observations, self.dtype)
        z = self.encoder(observations)
        action = self.predictor(z)
        return action


class DDPGCritic(nn.Module):
    block_type: str
    num_blocks: int
    hidden_dim: int
    dtype: Any

    def setup(self):
        self.encoder = DDPGEncoder(
            block_type=self.block_type,
            num_blocks=self.num_blocks,
            hidden_dim=self.hidden_dim,
            dtype=self.dtype,
        )
        self.predictor = LinearCritic()

    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        inputs = jnp.concatenate((observations, actions), axis=1)
        inputs = convert_element_type(inputs, self.dtype)
        z = self.encoder(inputs)
        q = self.predictor(z)
        return q


class DDPGClippedDoubleCritic(nn.Module):
    """
    Vectorized Double-Q for Clipped Double Q-learning.
    https://arxiv.org/pdf/1802.09477v3
    """

    block_type: str
    num_blocks: int
    hidden_dim: int
    dtype: Any

    num_qs: int = 2

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        VmapCritic = nn.vmap(
            DDPGCritic,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_qs,
        )

        qs = VmapCritic(
            block_type=self.block_type,
            num_blocks=self.num_blocks,
            hidden_dim=self.hidden_dim,
            dtype=self.dtype,
        )(observations, actions)

        return qs
