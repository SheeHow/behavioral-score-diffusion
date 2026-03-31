# Behavioral Score Diffusion (BSD)

Official implementation for **"Behavioral Score Diffusion: Model-Free Trajectory Planning via Kernel-Based Score Estimation from Data"** (CDC 2026).

[[Project Page]](https://sheehow.github.io/behavioral-score-diffusion/) [[Paper]](#) [[Video]](#)

## Overview

BSD is a **training-free** and **model-free** diffusion planner that computes the diffusion score function directly from a library of trajectory data via kernel-weighted estimation. At each denoising step, BSD retrieves relevant trajectories using a triple-kernel weighting scheme (diffusion proximity, state context, goal relevance) and computes a Nadaraya-Watson estimate of the denoised trajectory.

**Key results:**
- **98.5%** of model-based baseline reward without any dynamics model
- **18--63%** improvement over nearest-neighbor retrieval
- Safety shielding transfers directly from model-based methods

## Installation

1. Install Docker and NVIDIA Container Toolkit (CUDA):

```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu20.04 nvidia-smi
```

2. Build and start the container:

```bash
docker compose build
xhost +local:docker
docker compose up -d
```

3. Enter the container:

```bash
docker exec -it jax_dev bash
```

## Quick Start

### Run BSD planner

```python
from mbd.planners.bsd_planner import BSDPlannerConfig, run_bsd_diffusion
from mbd.planners.mbd_planner import MBDConfig
import mbd

# Configure
config = MBDConfig(env_name="tt2d", case="parking", num_trailers=1)
env = mbd.envs.get_env(config.env_name, case=config.case, dt=config.dt,
                        H=config.Hsample, num_trailers=config.num_trailers)

# Run BSD
run_bsd_diffusion(args=config, env=env)
```

### Reproduce experiments

```bash
# Run all BSD experiments (4 systems x 4 conditions x 50 trials)
python experiments/run_bsd_experiments.py

# Run specific system
python experiments/run_bsd_experiments.py --env_name tt2d
```

### Collect trajectory data

```bash
python -m mbd.data.collect_data --env_name tt2d --n_trajectories 1000
```

## Supported Systems

| System | State Dim | Description |
|--------|-----------|-------------|
| `kinematic_bicycle2d` | 3D | Kinematic bicycle (x, y, theta) |
| `tt2d` | 4D | Tractor-trailer (x, y, theta1, theta2) |
| `n_trailer2d` | 5D | N-trailer with additional joint |
| `acc_tt2d` | 6D | Acceleration-controlled tractor-trailer |

## Project Structure

```
mbd/
├── planners/
│   ├── mbd_planner.py       # Model-based diffusion (baseline)
│   └── bsd_planner.py       # Behavioral score diffusion (ours)
├── scorers/
│   ├── behavioral_score.py   # Kernel-weighted score estimation
│   └── bandwidth_schedule.py # Adaptive/fixed bandwidth
├── data/
│   ├── trajectory_dataset.py # Dataset loading and management
│   └── collect_data.py       # Data collection from MBD
├── robots/                   # Vehicle dynamics models
├── envs/                     # Environment setup & visualization
└── utils.py                  # JAX utilities, rollout, animation
```

## Citation

If you find this repository useful, please consider citing our paper:

```bibtex
@inproceedings{bsd2026,
  title     = {Behavioral Score Diffusion: Model-Free Trajectory Planning
               via Kernel-Based Score Estimation from Data},
  author    = {Li, Shihao and Li, Jiachen and Xu, Jiamin and Chen, Dongmei},
  booktitle = {IEEE Conference on Decision and Control (CDC)},
  year      = {2026},
}
```

## Acknowledgments

This implementation builds upon the [Safe-MPD](https://www.taekyung.me/safe-mpd) codebase by Kim et al. and the [Model-Based Diffusion](https://github.com/LeCAR-Lab/model-based-diffusion) framework. We thank the authors for making their code publicly available.
