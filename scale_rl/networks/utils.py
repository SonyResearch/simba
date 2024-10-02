from typing import List

import flax.linen as nn
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


def tree_norm(tree):
    return jnp.sqrt(sum((x**2).sum() for x in jax.tree_util.tree_leaves(tree)))


###########################
###   Initialization    ###
###########################


def orthogonal_init(scale: float = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


def xavier_normal_init():
    return nn.initializers.glorot_normal()


def xavier_uniform_init():
    return nn.initializers.glorot_uniform()


def he_normal_init():
    return nn.initializers.he_normal()


def he_uniform_init():
    return nn.initializers.he_uniform()


###########################


def noisy_sample(dist: tfd.Distribution, action_noise: List[jnp.ndarray]):
    """
    reference: https://github.com/martius-lab/pink-noise-rl/blob/main/pink/sb3.py
    """
    if isinstance(dist, tfp.distributions.TransformedDistribution):
        dist = dist.distribution
    mean = dist.loc
    scale_diag = dist.stddev()
    actions = mean + scale_diag * jnp.stack(action_noise)
    return nn.tanh(actions)
