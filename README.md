# Simba

Official implementation of 


Simba: Simplicity Bias for Scaling Up Parameters in Deep Reinforcement Learning




## Overview



## Getting strated

### Docker

We provide a `Dockerfile` for easy installation. You can build the docker image by running

```
cd deps
docker build -t scale_rl .
cd ../
docker run -i -d --gpus all -v .:/home/user/scale_rl -t scale_rl
docker exec -it <image_id> /bin/bash
```

### Conda

If you prefer to install dependencies manually, start by installing dependencies via conda by following the guidelines.

Create conda environment
```
cd deps
conda env create -f environment.yaml
```

Install Jax-GPU
```
RUN pip install -U "jax[cuda12]==0.4.25" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

#### Mujoco

```
Please refer to https://github.com/google-deepmind/mujoco

# Headless rendering
export MUJOCO_GL="egl"
export MUJOCO_EGL_DEVICE_ID="0"
export MKL_SERVICE_FORCE_INTEL="0"
```

#### Humanoid Bench

```
git clone https://github.com/joonleesky/humanoid-bench
cd humanoid-bench
pip install -e .
```

#### Myosuite

```
git clone --recursive https://github.com/joonleesky/myosuite
cd myosuite
pip install -e .
```


##  Example usage

We provide examples on how to train SAC agents with SimBa architecture.  

To run a single experiment
```
python run.py
```

To benchmark a simba with running all environments
```
python run_parallel.py \
    --task all \
    --device_ids <list of gpu devices to use> \
    --num_seeds <num_seeds> \
    --num_exp_per_device <number>  
```

### Scripts

An example script to collect DMC results using SAC with Simba:
```
bash scripts/sac_simba_dmc_em.sh
bash scripts/sac_simba_dmc_hard.sh
bash scripts/sac_simba_hbench.sh
bash scripts/sac_simba_myosuite.sh
```

## Analysis

Please refer to `analysis/benchmark.ipynb` to analyze the exprimental results provided in the paper.


## Citation

If you find our work useful, please consider citing our paper as follows:

```
@article{lee2024simba,
  title={Simba: Simplicity Bias for Scaling Up Parameters in Deep Reinforcement Learning}, 
  author={Hojoon Lee and Dongyoon Hwang and Donghu Kim and Hyunseung Kim and Jun Jet Tai and Kaushik Subramanian and Peter R.Wurman and Jaegul Choo and Peter Stone and Takuma Seno},
  year={2024}
}
```