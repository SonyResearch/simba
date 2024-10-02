from functools import partial
from typing import Any, Optional, Sequence, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import dynamic_scale as dynamic_scale_lib

from scale_rl.networks.utils import tree_norm

PRNGKey = jnp.ndarray


@flax.struct.dataclass
class Trainer:
    network_def: nn.Module = flax.struct.field(pytree_node=False)
    params: flax.core.FrozenDict[str, Any]
    tx: Optional[optax.GradientTransformation] = flax.struct.field(pytree_node=False)
    opt_state: Optional[optax.OptState] = None
    update_step: int = 0
    dynamic_scale: Optional[dynamic_scale_lib.DynamicScale] = None
    """
    dataclass decorator makes custom class to be passed safely to Jax.
    https://flax.readthedocs.io/en/latest/api_reference/flax.struct.html

    Trainer class wraps network & optimizer to easily optimize the network under the hood.

    args:
        network_def:
        params: network parameters.
        tx: optimizer (e.g., optax.Adam).
        opt_state: current state of the optimizer (e.g., beta_1 in Adam).
        update_step: number of update step so far.
    """

    @classmethod
    def create(
        cls,
        network_def: nn.Module,
        network_inputs: flax.core.FrozenDict[str, jnp.ndarray],
        tx: Optional[optax.GradientTransformation] = None,
        dynamic_scale: Optional[dynamic_scale_lib.DynamicScale] = None,
    ) -> "Trainer":
        variables = network_def.init(**network_inputs)
        params = variables.pop("params")

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        network = cls(
            network_def=network_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            dynamic_scale=dynamic_scale,
        )

        return network

    def __call__(self, *args, **kwargs):
        return self.network_def.apply({"params": self.params}, *args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.network_def.apply(*args, **kwargs)

    def apply_gradient(self, loss_fn) -> Tuple[Any, "Trainer"]:
        if self.dynamic_scale:
            grad_fn = self.dynamic_scale.value_and_grad(loss_fn, has_aux=True)
            dynamic_scale, is_fin, (_, info), grads = grad_fn(self.params)
        else:
            grad_fn = jax.grad(loss_fn, has_aux=True)
            grads, info = grad_fn(self.params)
            dynamic_scale = None
            is_fin = True
        grad_norm = tree_norm(grads)
        info["grad_norm"] = grad_norm

        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        network = self.replace(
            params=jax.tree_util.tree_map(
                partial(jnp.where, is_fin), new_params, self.params
            ),
            opt_state=jax.tree_util.tree_map(
                partial(jnp.where, is_fin), new_opt_state, self.opt_state
            ),
            update_step=self.update_step + 1,
            dynamic_scale=dynamic_scale,
        )

        return network, info
