# D3: Divide, Discover, Deploy
### Factorized Skill Learning with Symmetry and Style Priors

<p align="center">
  <img src="docs/overview_fig.png" width="100%" alt="Main figure of D3 paper"/>
</p>

<p align="center">
  <a href="https://leggedrobotics.github.io/d3-skill-discovery/"><strong>🌐 Project Page</strong></a> •
  <a href="#-overview"><strong>📖 Overview</strong></a> •
  <a href="#-installation"><strong>📦 Installation</strong></a> •
  <a href="#-usage"><strong>💻 Usage</strong></a> •
  <a href="#-citation"><strong>📝 Citation</strong></a>
</p>

---

## 📖 Overview

This repository contains the official implementation of **D3: Divide, Discover, Deploy**, presented at **CoRL 2025**. D3 is a framework for learning diverse and reusable robotic skills through factorized unsupervised skill discovery with symmetry and style priors.

### ✨ Key Features

- 🔀 **Factorized USD Algorithms**: Modular implementation supporting DIAYN, METRA, and extensible to custom algorithms
- 🤖 **IsaacLab Integration**: High-performance simulation environments for quadrupedal robots
- 📊 **Hierarchical Skill Learning**: Support for both low-level and high-level skill discovery
- 🎯 **Downstream Task Evaluation**: Pre-configured environments for goal tracking, pedipulation, and velocity tracking

---

## 📋 Table of Contents

- [📖 Overview](#-overview)
- [📦 Installation](#-installation)
- [🏗️ Repository Structure](#️-repository-structure)
- [🌍 Environments](#-environments)
  - [🔬 Unsupervised Skill Discovery (USD)](#-unsupervised-skill-discovery-usd)
  - [🎮 Downstream Tasks](#-downstream-tasks)
- [💻 Usage](#-usage)
  - [🎓 Training Unsupervised Skills](#-training-unsupervised-skills)
  - [🏗️ Training High-Level Skills](#️-training-high-level-skills)
  - [🎯 Evaluation on Downstream Tasks](#-evaluation-on-downstream-tasks)
  - [🎨 Interactive Skill Control](#-interactive-skill-control)
  - [🔬 Hyperparameter Sweeps](#-hyperparameter-sweeps)
  - [📈 Monitoring Training](#-monitoring-training)
- [🧠 Algorithm Details](#-algorithm-details)
  - [📚 Supported USD Algorithms](#-supported-usd-algorithms)
  - [🔀 Factorized USD](#-factorized-usd)
- [📊 Performance & Training Tips](#-performance--training-tips)
- [🐛 Troubleshooting](#-troubleshooting)
- [📝 Citation](#-citation)
- [⚖️ License](#️-license)

---

## 📦 Installation

<details>
<summary><b>📋 Prerequisites</b></summary>

<br>

| Requirement | Version/Details |
|------------|-----------------|
| **Operating System** | Linux (tested on Ubuntu 20.04+) |
| **Python** | 3.10+ |
| **CUDA** | 11.8+ (for GPU acceleration) |
| **GPU Memory** | 16GB+ VRAM recommended |
| **Disk Space** | ~50GB for Isaac Sim + dependencies |
| **Isaac Sim** | 4.5.0+ (included with Isaac Lab 2.2) |

</details>

### 1️⃣ Install Isaac Lab

Follow the official [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html) to install Isaac Lab 2.2.


### 2️⃣ Create Conda Environment

From your Isaac Lab installation directory:

```bash
./isaaclab.sh --conda d3_env
conda activate d3_env
```

### 3️⃣ Install Isaac Lab Extensions

```bash
./isaaclab.sh --install
```

### 4️⃣ Install D3 Extension

Clone this repository and install:

```bash
git clone https://github.com/leggedrobotics/d3-skill-discovery.git
cd d3-skill-discovery
./install.sh
```

<details>
<summary>What does <a href="install.sh"><code>install.sh</code></a> do?</summary>

<br>

The installation script will:
- ✅ Install the [`d3_rsl_rl`](d3_rsl_rl/) package with USD algorithms
- ✅ Register the [`d3_skill_discovery`](exts/d3_skill_discovery/d3_skill_discovery/) extension with Isaac Lab
- ✅ Set up all Python dependencies
- ✅ Verify the installation

</details>

---

## 🏗️ Repository Structure

<details>
<summary>Click to expand directory tree</summary>

```
d3-skill-discovery/
├── exts/d3_skill_discovery/           # IsaacLab extension with environments
│   └── d3_skill_discovery/
│       ├── tasks/                      # Environment implementations
│       │   ├── unsupervised_skill_discovery/  # USD environments
│       │   └── downstream/             # Evaluation tasks
│       └── d3_rsl_rl/                     # Configuration utilities
├── d3_rsl_rl/                             # Reinforcement learning algorithms
│   └── d3_rsl_rl/
│       ├── algorithms/                 # PPO implementation
│       ├── intrinsic_motivation/       # USD algorithms (DIAYN, METRA, etc.)
│       ├── modules/                    # Neural network architectures
│       ├── runners/                    # Training orchestration
│       └── storage/                    # Rollout buffer management
└── scripts/                            # Training and evaluation scripts
    └── d3_rsl_rl/
        ├── train.py                    # Main training script
        ├── play.py                     # Policy visualization
        └── skill_gui.py                # Interactive skill control GUI
```

</details>

---

## 🌍 Environments

The tasks are implemented inside [`exts/d3_skill_discovery/d3_skill_discovery/tasks`](exts/d3_skill_discovery/d3_skill_discovery/tasks) directory.

### 🔬 Unsupervised Skill Discovery (USD)

ANYmal-D environments for learning diverse skills without task-specific rewards:

| Environment | Description | Task ID | Config File |
|------------|-------------|---------|-------------|
| **🦿 Low-Level USD** | Basic skill learning on rough terrain (as described in paper) | `Isaac-USD-Anymal-D-v0` | [`anymal_usd_env_cfg.py`](exts/d3_skill_discovery/d3_skill_discovery/tasks/unsupervised_skill_discovery/anymal_usd/anymal_usd_env_cfg.py) |
| **🎯 High-Level USD** | Hierarchical skill learning (requires pretrained low-level policy) | `Isaac-HL-USD-Anymal-D-v0` | [`anymal_hl_usd_env_cfg.py`](exts/d3_skill_discovery/d3_skill_discovery/tasks/unsupervised_skill_discovery/anymal_usd/anymal_hl_usd_env_cfg.py) |
| **📦 USD with Box** | Skill learning with interactive movable box for manipulation | `Isaac-HL-USD-Box-Anymal-D-v0` | [`anymal_hl_usd_box_env_cfg.py`](exts/d3_skill_discovery/d3_skill_discovery/tasks/unsupervised_skill_discovery/anymal_usd/anymal_hl_usd_box_env_cfg.py) |

<details>
<summary>💡 Which environment should I start with?</summary>

<br>

For reproducing paper results, start with **Low-Level USD** (`Isaac-USD-Anymal-D-v0`). Once you have a trained low-level policy, you can proceed to high-level skill learning.

</details>

### 🎮 Downstream Tasks

Evaluation environments for testing learned skills on goal-directed tasks:

| Task Category | Description | Directory |
|--------------|-------------|-----------|
| **🎯 Goal Tracking** | Goal-reaching navigation on rough terrain | [`goal_tracking/`](exts/d3_skill_discovery/d3_skill_discovery/tasks/downstream/goal_tracking/) |
| **🦾 Pedipulation** | Precise foot positioning and object manipulation | [`pedipulation/`](exts/d3_skill_discovery/d3_skill_discovery/tasks/downstream/pedipulation/) |
| **🏃 Velocity Tracking** | Velocity tracking and locomotion control | [`velocity_tracking/`](exts/d3_skill_discovery/d3_skill_discovery/tasks/downstream/velocity_tracking/) |

---

## 💻 Usage

### 🎓 Training Unsupervised Skills

Train a low-level skill discovery model on ANYmal-D using [`scripts/d3_rsl_rl/train.py`](scripts/d3_rsl_rl/train.py):

```bash
python scripts/d3_rsl_rl/train.py \
  --task Isaac-USD-Anymal-D-v0 \
  --num_envs 2048 \
  --headless \
  --logger wandb \
  --run_name my_experiment
```

<details>
<summary><b>⚙️ Command Line Arguments</b></summary>

<br>

| Argument | Description | Default |
|----------|-------------|---------|
| `--task` | Environment task ID (see [Environments](#-environments)) | Required |
| `--num_envs` | Number of parallel simulation environments | `2048` |
| `--headless` | Run without GUI for faster training | `False` |
| `--logger` | Logging backend: `wandb` or `tensorboard` | `tensorboard` |
| `--run_name` | Experiment name for logging | Auto-generated |
| `--max_iterations` | Maximum training iterations | `10000` |
| `--device` | Compute device: `cuda` or `cpu` | `cuda` |

**See all available arguments:** [`scripts/d3_rsl_rl/cli_args.py`](scripts/d3_rsl_rl/cli_args.py)

</details>

> **💡 Tip**: For fastest training, use `--headless` mode and increase `--num_envs` based on your GPU memory (e.g., 4096 for 24GB+ VRAM).

### 🏗️ Training High-Level Skills

For hierarchical skill learning, first train a low-level policy, then:

```bash
python scripts/d3_rsl_rl/train.py \
  --task Isaac-HL-USD-Anymal-D-v0 \
  --num_envs 2048 \
  --headless \
  --logger wandb \
  --load_run path/to/low_level/checkpoint
```

> **⚠️ Important**: High-level training requires a pretrained low-level policy. Use `--load_run` to specify the checkpoint directory.

### 🎯 Evaluation on Downstream Tasks

Evaluate learned skills on downstream tasks using [`scripts/d3_rsl_rl/play.py`](scripts/d3_rsl_rl/play.py):

```bash
python scripts/d3_rsl_rl/play.py \
  --task Isaac-Goal-Tracking-Anymal-D-v0 \
  --num_envs 64 \
  --load_run path/to/trained/checkpoint
```

<details>
<summary><b>🎮 Available Evaluation Tasks</b></summary>

<br>

```bash
# Goal tracking on rough terrain
python scripts/d3_rsl_rl/play.py --task Isaac-Goal-Tracking-Anymal-D-v0 --load_run <checkpoint>

# Foot positioning for manipulation
python scripts/d3_rsl_rl/play.py --task Isaac-Foot-Tracking-Anymal-D-v0 --load_run <checkpoint>

# Velocity tracking locomotion
python scripts/d3_rsl_rl/play.py --task Isaac-Velocity-Tracking-Anymal-D-v0 --load_run <checkpoint>
```

</details>

### 🎨 Interactive Skill Control

Launch the skill GUI to manually control and visualize learned skills using [`scripts/d3_rsl_rl/skill_gui.py`](scripts/d3_rsl_rl/skill_gui.py):

```bash
python scripts/d3_rsl_rl/skill_gui.py \
  --checkpoint path/to/trained/model
```

### 🔬 Hyperparameter Sweeps

Run hyperparameter optimization sweeps using WandB:

<details>
<summary><b>Running WandB Sweeps</b></summary>

<br>

#### 1️⃣ Configure Sweep

Edit [`scripts/d3_rsl_rl/sweep/sweep.yaml`](scripts/d3_rsl_rl/sweep/sweep.yaml) to define parameters to optimize:

```yaml
program: scripts/d3_rsl_rl/train.py
method: bayes
metric:
  name: train/episode_reward
  goal: maximize
parameters:
  learning_rate:
    min: 1e-4
    max: 1e-2
  num_envs:
    values: [1024, 2048, 4096]
```

#### 2️⃣ Initialize Sweep (Once)

Run [`scripts/d3_rsl_rl/sweep/initialize_sweep.py`](scripts/d3_rsl_rl/sweep/initialize_sweep.py):

```bash
python scripts/d3_rsl_rl/sweep/initialize_sweep.py --project_name my_sweep
```

This writes the sweep ID to [`scripts/d3_rsl_rl/sweep/sweep_ids.json`](scripts/d3_rsl_rl/sweep/sweep_ids.json).

#### 3️⃣ Run Sweep Agents

Run [`scripts/d3_rsl_rl/sweep/sweep.py`](scripts/d3_rsl_rl/sweep/sweep.py):

```bash
# Run on single machine
python scripts/d3_rsl_rl/sweep/sweep.py --project_name my_sweep

# Run on multiple machines (same sweep_id)
python scripts/d3_rsl_rl/sweep/sweep.py --project_name my_sweep
```

> **💡 Tip**: You can run multiple agents in parallel on different machines to speed up the sweep.

#### 🖥️ Running on Cluster

To run sweeps on a cluster with Isaac Sim, you need to configure the sweep **before** initializing it. Update your [`sweep.yaml`](scripts/d3_rsl_rl/sweep/sweep.yaml) to use the Isaac Sim Python interpreter:

```yaml
command:
  - /isaac-sim/python.sh
  - ${program}
  # rest
```

> **⚠️ Important**: This configuration must be set **before** running `initialize_sweep.py`. Once initialized, you cannot run the same sweep on both cluster and local machines due to different Python interpreters.

</details>

---

### 📈 Monitoring Training

Track your training progress using built-in logging:

<details>
<summary><b>WandB (Recommended)</b></summary>

<br>

```bash
# Login to WandB
wandb login

# Train with WandB logging
python scripts/d3_rsl_rl/train.py \
  --task Isaac-USD-Anymal-D-v0 \
  --logger wandb \
  --wandb_project d3-skill-discovery
```

**Logged Metrics:**

- Episode rewards and lengths
- Policy/value loss
- Discriminator accuracy (USD algorithms)
- Skill diversity metrics
- Learning rates

</details>

<details>
<summary><b>TensorBoard</b></summary>

<br>

```bash
# Train with TensorBoard logging (default)
python scripts/d3_rsl_rl/train.py \
  --task Isaac-USD-Anymal-D-v0 \
  --logger tensorboard

# View logs
tensorboard --logdir logs/
```

</details>

---

## 🧠 Algorithm Details

### 📚 Supported USD Algorithms

The framework builds upon [d3_rsl_rl](https://github.com/leggedrobotics/d3_rsl_rl) (v2.2.0) and uses the [`OnPolicyRunnerUSD`](d3_rsl_rl/d3_rsl_rl/runners/on_policy_runner_usd.py) for training. Currently implemented USD algorithms:

| Algorithm | Description | Implementation |
|-----------|-------------|----------------|
| **DIAYN** | Diversity is All You Need | [`diayn.py`](d3_rsl_rl/d3_rsl_rl/intrinsic_motivation/diayn.py) |
| **METRA** | Meta-Reinforcement Learning with Task Abstraction | [`metra.py`](d3_rsl_rl/d3_rsl_rl/intrinsic_motivation/metra.py) |

**Base RL Algorithm:**

- **PPO**: Proximal Policy Optimization - [`ppo.py`](d3_rsl_rl/d3_rsl_rl/algorithms/ppo.py)

**Neural Network Modules:**

- **Actor-Critic**: Standard policy network - [`actor_critic.py`](d3_rsl_rl/d3_rsl_rl/modules/actor_critic.py)
- **Recurrent AC**: LSTM-based policy - [`actor_critic_recurrent.py`](d3_rsl_rl/d3_rsl_rl/modules/actor_critic_recurrent.py)
- More architectures available in [`d3_rsl_rl/d3_rsl_rl/modules/`](d3_rsl_rl/d3_rsl_rl/modules/)

### 🔀 Factorized USD

The [`FACTOR_USD`](d3_rsl_rl/d3_rsl_rl/intrinsic_motivation/factoized_unsupervised_skill_discovery.py) class manages multiple USD algorithms simultaneously, enabling factorized skill discovery across different observation spaces.

> **🔬 Research Note**: Factorized USD allows decomposing skill learning into multiple factors (e.g., gait style, navigation behavior), each learned by a separate USD algorithm.

<details>
<summary><b>🛠️ Adding Custom USD Algorithms</b></summary>

<br>

To integrate a new unsupervised skill discovery algorithm, follow these steps:

#### 1️⃣ Implement the Algorithm

Subclass [`BaseSkillDiscovery`](d3_rsl_rl/d3_rsl_rl/intrinsic_motivation/base_skill_discovery.py) in `d3_rsl_rl/intrinsic_motivation/`:

```python
import torch
from d3_rsl_rl.intrinsic_motivation.base_skill_discovery import BaseSkillDiscovery

class MyUSDAlgorithm(BaseSkillDiscovery):
    def reward(self, usd_observations, skill: torch.Tensor, **kwargs) -> torch.Tensor:
        """Calculate the intrinsic reward for the underlying RL algorithm."""
        # Your reward computation logic
        pass

    def sample_skill(self, envs_to_sample: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample a skill z."""
        # Your skill sampling logic
        pass

    def update(self, observation_batch, **kwargs) -> dict:
        """Update the intrinsic motivation algorithm (e.g., train discriminator)."""
        # Your update logic (e.g., discriminator training)
        return {"loss": loss_value}

    def get_save_dict(self) -> dict:
        """Return state dict for saving."""
        return {"model_state": self.model.state_dict()}

    def load(self, state_dict: dict, **kwargs) -> None:
        """Load the algorithm state."""
        self.model.load_state_dict(state_dict["model_state"])

    @property
    def performance_metric(self) -> float:
        """Return performance metric between 0 and 1."""
        # Your performance metric (e.g., discriminator accuracy)
        return 0.5
```

#### 2️⃣ Create Configuration

Add a config class to [`exts/d3_skill_discovery/d3_skill_discovery/d3_rsl_rl/rl_cfg.py`](exts/d3_skill_discovery/d3_skill_discovery/d3_rsl_rl/rl_cfg.py):

```python
@configclass
class MyUSDAlgorithmCfg:
    """Configuration for MyUSDAlgorithm."""
    learning_rate: float = 3e-4
    # Your config parameters
```

#### 3️⃣ Update FACTOR_USD

Extend the factory class in [`d3_rsl_rl/intrinsic_motivation/factoized_unsupervised_skill_discovery.py`](d3_rsl_rl/d3_rsl_rl/intrinsic_motivation/factoized_unsupervised_skill_discovery.py) to initialize your algorithm.

#### 4️⃣ Configure Environment

Update the environment's USD configuration:

```python
# In your environment config file
factors: dict[str, tuple[list[str], Literal["metra", "diayn", "my_algorithm"]]]
skill_dims: dict[str, int]
resampling_intervals: dict[str, int]
usd_alg_extra_cfg: dict[str, dict]
```

</details>

---

## 📊 Performance & Training Tips

<details>
<summary><b>Expected Training Performance</b></summary>

<br>

### Hardware Requirements

| Configuration | Recommended Specs |
|--------------|------------------|
| **GPU** | NVIDIA RTX 3090 / 4090 or better |
| **VRAM** | 16GB+ (24GB for 4096 envs) |
| **CPU** | Modern multi-core processor |
| **RAM** | 32GB+ |

### Training Time Estimates

| Task | Envs | Iterations | Time (RTX 4090) |
|------|------|-----------|-----------------|
| Low-Level USD | 2048 | 10,000 | ~3-5 hours |
| High-Level USD | 2048 | 5,000 | ~2-3 hours |
| Downstream Tasks | 2048 | 5,000 | ~2-3 hours |

</details>

<details>
<summary><b>🎯 Hyperparameter Tuning Tips</b></summary>

<br>

### Key Hyperparameters

Most hyperparameters can be adjusted in the configuration files:

**Environment Configs:**

- [`exts/d3_skill_discovery/d3_skill_discovery/tasks/unsupervised_skill_discovery/anymal_usd/`](exts/d3_skill_discovery/d3_skill_discovery/tasks/unsupervised_skill_discovery/anymal_usd/)

**Agent Configs:**

- USD Algorithm: [`rsl_rl_usd_cfg.py`](exts/d3_skill_discovery/d3_skill_discovery/tasks/unsupervised_skill_discovery/anymal_usd/config/anymal_d/agents/rsl_rl_usd_cfg.py)
- PPO Hyperparameters: [`rsl_rl_ppo_cfg.py`](exts/d3_skill_discovery/d3_skill_discovery/tasks/unsupervised_skill_discovery/anymal_usd/config/anymal_d/agents/rsl_rl_ppo_cfg.py)

**Common adjustments:**

```python
# PPO hyperparameters
num_learning_epochs: int = 5      # Number of epochs per iteration
num_mini_batches: int = 4         # Mini-batches per epoch
learning_rate: float = 1e-3       # Actor-critic learning rate

# USD-specific
skill_dim: int = 8                # Dimension of skill space
resampling_interval: int = 1000   # Steps before resampling skills
```

### Tips for Better Performance

1. **Start with default parameters** from the paper
2. **Increase num_envs** if you have more GPU memory
3. **Adjust learning_rate** if training is unstable
4. **Monitor discriminator accuracy** in WandB/TensorBoard
5. **Use curriculum learning** for complex terrains

</details>

---

## 🐛 Troubleshooting

<details>
<summary><b>Common Issues and Solutions</b></summary>

<br>

### Installation Issues

**Problem**: `ImportError: No module named 'isaaclab'`

**Solution**: Ensure Isaac Lab is properly installed and the conda environment is activated:

```bash
conda activate d3_env
python -c "import isaaclab; print(isaaclab.__version__)"
```

---

**Problem**: `CUDA out of memory` during training

**Solution**: Reduce the number of parallel environments:
```bash
python scripts/d3_rsl_rl/train.py --task Isaac-USD-Anymal-D-v0 --num_envs 1024  # or lower
```

---

### Training Issues

**Problem**: Training is very slow on my GPU

**Solution**:

- Use `--headless` mode to disable rendering
- Ensure you're using CUDA: check `nvidia-smi` shows GPU usage
- Close other GPU-intensive applications

---

**Problem**: WandB login required

**Solution**: Either login to WandB or use a different logger:

```bash
wandb login  # Enter your API key
# OR use tensorboard instead
python scripts/d3_rsl_rl/train.py --logger tensorboard
```

</details>

---

## 📝 Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{cathomen2025d3,
  author    = {Cathomen, Rafael and Mittal, Mayank and Vlastelica, Marin and Hutter, Marco},
  title     = {Divide, Discover, Deploy: Factorized Skill Learning with Symmetry and Style Priors},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2025},
}
```

---

## ⚖️ License

This project is licensed under the **BSD-3-Clause License**. See [LICENSE](LICENCE) for details.

<details>
<summary><b>Third-Party Licenses</b></summary>

<br>

This project incorporates code from the following open-source projects:

| Project | License | Details |
|---------|---------|---------|
| **Isaac Lab** | BSD-3-Clause | [View License](docs/licenses/isaaclab-license.txt) |
| **d3_rsl_rl** | BSD-3-Clause | [View License](docs/licenses/d3_rsl_rl-license.txt) |

</details>
