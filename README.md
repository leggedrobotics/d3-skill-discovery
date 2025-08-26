# Unsupervised Skill Discovery Environments

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.2.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

## Overview

This repository contains the code for the [Divide Discover Deploy](https://leggedrobotics.github.io/d3-skill-discovery/) paper.
It contains environments and algorithms for unsupervised skill discovery.

```
d3-skill-discovery/
├── exts/unsupervised_RL/   # IsaacLab Environments
├── scripts/                # Training scripts
└── rsl_rl/                 # Algorithms
```


## Installation

### Prerequisites

- Install Isaac Lab, see the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html#).
- Create a new conda environment:

  ```bash
  ./isaaclab.sh --conda d3_env
  conda activate d3_env
  ```

- Install IsaacLab extensions:

  ```bash
  ./isaaclab.sh --install none
  ```

### Install this extension

Navigate to the project directory and run the installation script:

```bash
cd d3-skill-discovery
./install.sh
```

## Training

Train the unsupervised skill discovery model:

```bash
python scripts/rsl_rl/train.py --task Isaac-USD-Anymal-D-v0 --num_envs 2048 --headless --logger wandb
```
