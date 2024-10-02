"""
Implementation of commonly used policies that can be shared across agents.
"""
from typing import Any

import flax.linen as nn
import jax.numpy as jnp
from jax.lax import convert_element_type
from tensorflow_probability.substrates import jax as tfp

from scale_rl.networks.utils import orthogonal_init

tfd = tfp.distributions
tfb = tfp.bijectors


class NormalTanhPolicy(nn.Module):
    action_dim: int
    state_dependent_std: bool = True
    kernel_init_scale: float = 1.0
    log_std_min: float = -10.0
    log_std_max: float = 2.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self,
        inputs: jnp.ndarray,
        temperature: float = 1.0,
    ) -> tfd.Distribution:
        means = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal_init(self.kernel_init_scale),
            dtype=self.dtype,
        )(inputs)

        if self.state_dependent_std:
            log_stds = nn.Dense(
                self.action_dim,
                kernel_init=orthogonal_init(self.kernel_init_scale),
                dtype=self.dtype,
            )(inputs)
        else:
            log_stds = self.param("log_stds", nn.initializers.zeros, (self.action_dim,))

        log_stds = convert_element_type(log_stds, jnp.float32)

        # suggested by Ilya for stability
        log_stds = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (
            1 + nn.tanh(log_stds)
        )

        # N(mu, exp(log_sigma))
        dist = tfd.MultivariateNormalDiag(
            loc=convert_element_type(means, jnp.float32),
            scale_diag=jnp.exp(log_stds) * temperature,
        )

        # tanh(N(mu, sigma))
        dist = tfd.TransformedDistribution(distribution=dist, bijector=tfb.Tanh())

        return dist


class TanhPolicy(nn.Module):
    action_dim: int
    kernel_init_scale: float = 1.0

    @nn.compact
    def __call__(
        self,
        inputs: jnp.ndarray,
    ) -> tfd.Distribution:
        actions = nn.Dense(
            self.action_dim, kernel_init=orthogonal_init(self.kernel_init_scale)
        )(inputs)

        return nn.tanh(actions)
