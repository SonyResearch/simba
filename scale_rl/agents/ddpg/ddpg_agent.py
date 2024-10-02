import functools
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import dynamic_scale

from scale_rl.agents.base_agent import BaseAgent
from scale_rl.agents.ddpg.ddpg_network import (
    DDPGActor,
    DDPGClippedDoubleCritic,
    DDPGCritic,
)
from scale_rl.agents.ddpg.ddpg_update import (
    update_actor,
    update_critic,
    update_target_network,
)
from scale_rl.buffers.base_buffer import Batch
from scale_rl.common.colored_noise import ColoredNoiseProcess
from scale_rl.common.scheduler import linear_decay_scheduler
from scale_rl.networks.trainer import PRNGKey, Trainer

"""
The @dataclass decorator must have `frozen=True` to ensure the instance is immutable,
allowing it to be treated as a static variable in JAX.
"""


@dataclass(frozen=True)
class DDPGConfig:
    seed: int
    num_train_envs: int
    max_episode_steps: int
    normalize_observation: bool

    actor_block_type: str
    actor_num_blocks: int
    actor_hidden_dim: int
    actor_learning_rate: float
    actor_weight_decay: float

    critic_block_type: str
    critic_num_blocks: int
    critic_hidden_dim: int
    critic_learning_rate: float
    critic_weight_decay: float
    critic_use_cdq: bool

    target_tau: float
    gamma: float
    n_step: int

    exp_noise_color: float
    exp_noise_scheduler: str
    exp_noise_decay_period: int
    exp_noise_std_init: float
    exp_noise_std_final: float

    mixed_precision: bool


@functools.partial(
    jax.jit,
    static_argnames=(
        "observation_dim",
        "action_dim",
        "cfg",
    ),
)
def _init_ddpg_networks(
    observation_dim: int,
    action_dim: int,
    cfg: DDPGConfig,
) -> Tuple[PRNGKey, Trainer, Trainer, Trainer]:
    fake_observations = jnp.zeros((1, observation_dim))
    fake_actions = jnp.zeros((1, action_dim))

    rng = jax.random.PRNGKey(cfg.seed)
    rng, actor_key, critic_key = jax.random.split(rng, 3)
    compute_dtype = jnp.float16 if cfg.mixed_precision else jnp.float32

    # When initializing the network in the flax.nn.Module class, rng_key should be passed as rngs.
    actor = Trainer.create(
        network_def=DDPGActor(
            block_type=cfg.actor_block_type,
            num_blocks=cfg.actor_num_blocks,
            hidden_dim=cfg.actor_hidden_dim,
            action_dim=action_dim,
            dtype=compute_dtype,
        ),
        network_inputs={"rngs": actor_key, "observations": fake_observations},
        tx=optax.adamw(
            learning_rate=cfg.actor_learning_rate,
            weight_decay=cfg.actor_weight_decay,
        ),
        dynamic_scale=dynamic_scale.DynamicScale() if cfg.mixed_precision else None,
    )

    if cfg.critic_use_cdq:
        critic_network_def = DDPGClippedDoubleCritic(
            block_type=cfg.critic_block_type,
            num_blocks=cfg.critic_num_blocks,
            hidden_dim=cfg.critic_hidden_dim,
            dtype=compute_dtype,
        )
    else:
        critic_network_def = DDPGCritic(
            block_type=cfg.critic_block_type,
            num_blocks=cfg.critic_num_blocks,
            hidden_dim=cfg.critic_hidden_dim,
            dtype=compute_dtype,
        )

    critic = Trainer.create(
        network_def=critic_network_def,
        network_inputs={
            "rngs": critic_key,
            "observations": fake_observations,
            "actions": fake_actions,
        },
        tx=optax.adamw(
            learning_rate=cfg.critic_learning_rate,
            weight_decay=cfg.critic_weight_decay,
        ),
        dynamic_scale=dynamic_scale.DynamicScale() if cfg.mixed_precision else None,
    )

    # we set target critic's parameters identical to critic by using same rng.
    target_network_def = critic_network_def
    target_critic = Trainer.create(
        network_def=target_network_def,
        network_inputs={
            "rngs": critic_key,
            "observations": fake_observations,
            "actions": fake_actions,
        },
        tx=None,
    )

    return rng, actor, critic, target_critic


@jax.jit
def _sample_ddpg_actions(
    rng: PRNGKey,
    actor: Trainer,
    observations: jnp.ndarray,
) -> Tuple[PRNGKey, jnp.ndarray]:
    rng, key = jax.random.split(rng)
    actions = actor(observations=observations)

    return rng, actions


@functools.partial(
    jax.jit,
    static_argnames=(
        "gamma",
        "n_step",
        "critic_use_cdq",
        "target_tau",
        "noise_std",
    ),
)
def _update_ddpg_networks(
    rng: PRNGKey,
    actor: Trainer,
    critic: Trainer,
    target_critic: Trainer,
    batch: Batch,
    gamma: float,
    n_step: int,
    critic_use_cdq: bool,
    target_tau: float,
    noise_std: float,
) -> Tuple[PRNGKey, Trainer, Trainer, Trainer, Trainer, Dict[str, float]]:
    rng, actor_key, critic_key = jax.random.split(rng, 3)

    new_actor, actor_info = update_actor(
        key=actor_key,
        actor=actor,
        critic=critic,
        batch=batch,
        critic_use_cdq=critic_use_cdq,
        noise_std=noise_std,
    )

    new_critic, critic_info = update_critic(
        key=critic_key,
        actor=new_actor,
        critic=critic,
        target_critic=target_critic,
        batch=batch,
        gamma=gamma,
        n_step=n_step,
        critic_use_cdq=critic_use_cdq,
        noise_std=noise_std,
    )

    new_target_critic, target_critic_info = update_target_network(
        network=new_critic,
        target_network=target_critic,
        target_tau=target_tau,
    )

    info = {
        **actor_info,
        **critic_info,
        **target_critic_info,
    }

    return (rng, new_actor, new_critic, new_target_critic, info)


class DDPGAgent(BaseAgent):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        cfg: DDPGConfig,
    ):
        """
        An agent that randomly selects actions without training.
        Useful for collecting baseline results and for debugging purposes.
        """

        super(DDPGAgent, self).__init__(
            observation_space,
            action_space,
            cfg,
        )

        self._observation_dim = observation_space.shape[-1]
        self._action_dim = action_space.shape[-1]

        # map dictionary to dataclass
        self._cfg = DDPGConfig(**cfg)

        self._init_network()
        self._init_exp_scheduler()
        self._init_action_noise()

    def _init_network(self):
        (
            self._rng,
            self._actor,
            self._critic,
            self._target_critic,
        ) = _init_ddpg_networks(self._observation_dim, self._action_dim, self._cfg)

    def _init_exp_scheduler(self):
        if self._cfg.exp_noise_scheduler == "linear":
            self._exp_scheduler = linear_decay_scheduler(
                decay_period=self._cfg.exp_noise_decay_period,
                initial_value=self._cfg.exp_noise_std_init,
                final_value=self._cfg.exp_noise_std_final,
            )
        else:
            raise NotImplemented

    def _init_action_noise(self):
        self._action_noise = []

        # each train environment has a separate noise schedule.
        for _ in range(self._cfg.num_train_envs):
            self._action_noise.append(
                ColoredNoiseProcess(
                    beta=self._cfg.exp_noise_color,
                    size=(self._action_dim, self._cfg.max_episode_steps),
                )
            )

    def sample_actions(
        self,
        interaction_step: int,
        prev_timestep: Dict[str, np.ndarray],
        training: bool,
    ) -> np.ndarray:
        if training:
            # reinitialize the noise if env was reinitialized
            prev_terminated = prev_timestep["terminated"]
            prev_truncated = prev_timestep["truncated"]
            for env_idx in range(self._cfg.num_train_envs):
                done = prev_terminated[env_idx] or prev_truncated[env_idx]
                if done:
                    self._action_noise[env_idx].reset()

            action_noise = np.array(
                [noise_sampler.sample() for noise_sampler in self._action_noise]
            )

            # scale the action noise with exp_noise_std
            self._noise_std = noise_std = self._exp_scheduler(interaction_step)
            action_noise = action_noise * noise_std

        else:
            action_noise = 0.0

        # current timestep observation is "next" observations from the previous timestep
        observations = jnp.asarray(prev_timestep["next_observation"])
        self._rng, actions = _sample_ddpg_actions(self._rng, self._actor, observations)
        actions = np.array(actions)
        actions = np.clip(actions + action_noise, -1, 1)

        return actions

    def update(self, update_step: int, batch: Dict[str, np.ndarray]) -> Dict:
        for key, value in batch.items():
            batch[key] = jnp.asarray(value)

        (
            self._rng,
            self._actor,
            self._critic,
            self._target_critic,
            update_info,
        ) = _update_ddpg_networks(
            rng=self._rng,
            actor=self._actor,
            critic=self._critic,
            target_critic=self._target_critic,
            batch=batch,
            gamma=self._cfg.gamma,
            n_step=self._cfg.n_step,
            target_tau=self._cfg.target_tau,
            critic_use_cdq=self._cfg.critic_use_cdq,
            noise_std=self._noise_std,
        )

        for key, value in update_info.items():
            update_info[key] = float(value)

        return update_info
