python run_parallel.py \
    --group_name test \
    --exp_name sac_simba \
    --agent_config sac_simba \
    --env_type myosuite \
    --num_seeds 10 \
    --num_exp_per_device 3 \
    --device_ids 0 \