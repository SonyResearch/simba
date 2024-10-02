import argparse
import random

import hydra
import numpy as np
import omegaconf
import tqdm
from dotmap import DotMap

from scale_rl.agents import create_agent
from scale_rl.buffers import create_buffer
from scale_rl.common import WandbTrainerLogger
from scale_rl.envs import create_envs
from scale_rl.evaluation import evaluate, record_video


def run(args):
    ###############################
    # configs
    ###############################

    args = DotMap(args)
    config_path = args.config_path
    config_name = args.config_name
    overrides = args.overrides

    hydra.initialize(version_base=None, config_path=config_path)
    cfg = hydra.compose(config_name=config_name, overrides=overrides)

    def eval_resolver(s: str):
        return eval(s)

    omegaconf.OmegaConf.register_new_resolver("eval", eval_resolver)
    omegaconf.OmegaConf.resolve(cfg)

    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    #############################
    # envs
    #############################
    train_env, eval_env = create_envs(**cfg.env)

    observation_space = train_env.observation_space
    action_space = train_env.action_space

    #############################
    # buffer
    #############################
    buffer = create_buffer(
        observation_space=observation_space, action_space=action_space, **cfg.buffer
    )
    buffer.reset()

    #############################
    # agent
    #############################

    # Since the network architecture is typically tied to the learning algorithm,
    #   we opted not to fully modularize the network for the sake of readability.
    # Therefore, for each algorithm, the network is implemented within its respective directory.

    agent = create_agent(
        observation_space=observation_space,
        action_space=action_space,
        cfg=cfg.agent,
    )

    #############################
    # train
    #############################

    logger = WandbTrainerLogger(cfg)

    # initial evaluation
    eval_info = evaluate(agent, eval_env, cfg.num_eval_episodes)
    logger.update_metric(**eval_info)
    logger.log_metric(step=0)
    logger.reset()

    # start training
    update_step = 0
    update_counter = 0
    observations, env_infos = train_env.reset()
    timestep = None
    for interaction_step in tqdm.tqdm(
        range(1, int(cfg.num_interaction_steps + 1)), smoothing=0.1
    ):
        # collect data
        # While using random actions until buffer.can_sample(),
        # we feed data into agent to compute statistics within a wrapper.
        if timestep:
            actions = agent.sample_actions(
                interaction_step, prev_timestep=timestep, training=True
            )
        if buffer.can_sample() is False:
            actions = train_env.action_space.sample()

        next_observations, rewards, terminateds, truncateds, env_infos = train_env.step(
            actions
        )

        next_buffer_observations = next_observations.copy()
        for env_idx in range(cfg.num_train_envs):
            if terminateds[env_idx] or truncateds[env_idx]:
                next_buffer_observations[env_idx] = env_infos["final_observation"][
                    env_idx
                ]

        timestep = {
            "observation": observations,
            "action": actions,
            "reward": rewards,
            "terminated": terminateds,
            "truncated": truncateds,
            "next_observation": next_buffer_observations,
        }

        buffer.add(timestep)
        timestep["next_observation"] = next_observations
        observations = next_observations

        if buffer.can_sample():
            # update network
            # updates_per_interaction_step can be below 1.0
            update_counter += cfg.updates_per_interaction_step
            while update_counter >= 1:
                batch = buffer.sample()
                update_info = agent.update(update_step, batch)
                logger.update_metric(**update_info)
                update_counter -= 1
                update_step += 1

            # evaluation
            if interaction_step % cfg.evaluation_per_interaction_step == 0:
                eval_info = evaluate(agent, eval_env, cfg.num_eval_episodes)
                logger.update_metric(**eval_info)

            # video recording
            if interaction_step % cfg.recording_per_interaction_step == 0:
                video_info = record_video(agent, eval_env, cfg.num_record_episodes)
                logger.update_metric(**video_info)

            # logging
            if interaction_step % cfg.logging_per_interaction_step == 0:
                # using env steps simplifies the comparison with the performance reported in the paper.
                env_step = interaction_step * cfg.action_repeat * cfg.num_train_envs
                logger.log_metric(step=env_step)
                logger.reset()

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config_path", type=str, default="./configs")
    parser.add_argument("--config_name", type=str, default="base")
    parser.add_argument("--overrides", action="append", default=[])
    args = parser.parse_args()

    run(vars(args))
