# RSL-RL + Unsupervised Skill Discovery

<p align="center">
  <strong>Extended reinforcement learning library for unsupervised skill discovery</strong>
</p>

---

## 📖 Overview

This directory contains an **extended version of [rsl_rl](https://github.com/leggedrobotics/rsl_rl)** (v[2.2.0](https://github.com/leggedrobotics/rsl_rl/releases/tag/v2.2.0), renamed to `d3_rsl_rl`), the Robotic Systems Lab's reinforcement learning library, with added support for **unsupervised skill discovery (USD) algorithms**.

### 🔗 Based on RSL-RL

[RSL-RL](https://github.com/leggedrobotics/rsl_rl) is a lightweight and modular library for on-policy reinforcement learning, originally developed at ETH Zurich's Robotic Systems Lab. It provides efficient implementations of PPO and related algorithms optimized for robot learning.

### ✨ What We Extend

This fork adds the following key capabilities to the original RSL-RL:

| Component | Description | Location |
|-----------|-------------|----------|
| 🧠 **USD Framework** | Abstract base class for implementing skill discovery algorithms | [`base_skill_discovery.py`](d3_rsl_rl/intrinsic_motivation/base_skill_discovery.py) |
| 🎯 **DIAYN** | Diversity is All You Need implementation | [`diayn.py`](d3_rsl_rl/intrinsic_motivation/diayn.py) |
| 🎨 **METRA** | Meta-Reinforcement Learning with Task Abstraction | [`metra.py`](d3_rsl_rl/intrinsic_motivation/metra.py) |
| 🔀 **Factorized USD** | Multi-algorithm manager for factorized skill learning | [`factoized_unsupervised_skill_discovery.py`](d3_rsl_rl/intrinsic_motivation/factoized_unsupervised_skill_discovery.py) |
| 🏃 **USD Runner** | Extended on-policy runner with USD integration | [`on_policy_runner_usd.py`](d3_rsl_rl/runners/on_policy_runner_usd.py) |

All original RSL-RL functionality (PPO, actor-critic modules, storage, etc.) remains intact and unchanged.

---

## 🏗️ Architecture

### Skill Discovery Framework

The USD framework is designed to be **modular and extensible**. Any USD algorithm that can:

- ✅ Provide an **intrinsic reward function** for state-action-skill tuples
- ✅ Be **trained with on-policy data** from PPO rollouts

can be integrated into this framework.

### Implemented Algorithms

| Algorithm | Paper | Use Case |
|-----------|-------|----------|
| **DIAYN** | [Eysenbach et al., 2018](https://arxiv.org/abs/1802.06070) | Discovering diverse behaviors without reward |
| **METRA** | [Pertsch et al., 2024](https://arxiv.org/abs/2310.08887) | Metric-aware task representation learning |

### Base Class: `BaseSkillDiscovery`

All USD algorithms inherit from [`BaseSkillDiscovery`](d3_rsl_rl/intrinsic_motivation/base_skill_discovery.py), which defines:

**Required Methods:**

- `reward()` — Compute intrinsic reward for skill-conditioned policy
- `sample_skill()` — Sample skill vector **z** for the policy
- `update()` — Update USD components (e.g., discriminator training)
- `get_save_dict()` / `load()` — Checkpoint management
- `performance_metric` — Return training progress metric (0-1)

**Optional Methods:**

- `visualize()` — Debug visualizations
- `update_skill_distribution()` — Curriculum learning over skills
- `symmetry_augmentation()` — Augment skills with symmetries

### Factorized Learning: `FACTOR_USD`

The [`FACTOR_USD`](d3_rsl_rl/intrinsic_motivation/factoized_unsupervised_skill_discovery.py) class enables **simultaneous training of multiple USD algorithms** on different observation subspaces. This allows decomposing skill learning into factors (e.g., separate gait style from navigation behavior).

---

## 🔧 Environment Requirements

To use USD algorithms, your Isaac Lab environment must define:

```python
# Environment configuration attributes

factors: dict[str, tuple[list[str], Literal["metra", "diayn"]]]
# Maps factor names to (observation keys, algorithm type)

skill_dims: dict[str, int]
# Dimensionality of each skill factor

resampling_intervals: dict[str, int]
# How often to resample each skill factor (in timesteps)

usd_alg_extra_cfg: dict[str, dict]
# Algorithm-specific hyperparameters per factor
```

**Algorithm Configuration:**

Your agent config must include a `usd` field of type `RslRlFactorizedUSDAlgorithmCfg` (see [`rl_cfg.py`](../exts/d3_skill_discovery/d3_skill_discovery/d3_rsl_rl/rl_cfg.py)).

---

## 🛠️ Adding a New USD Algorithm

Want to implement your own skill discovery method? Follow these steps:

### Step 1: Implement the Algorithm

Create a new file in [`d3_rsl_rl/intrinsic_motivation/`](d3_rsl_rl/intrinsic_motivation/) and subclass `BaseSkillDiscovery`:

```python
import torch
from d3_rsl_rl.intrinsic_motivation.base_skill_discovery import BaseSkillDiscovery

class MyUSDAlgorithm(BaseSkillDiscovery):
    def reward(self, usd_observations, skill: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute intrinsic reward for (observation, skill) pair."""
        # Your reward logic here
        pass

    def sample_skill(self, envs_to_sample: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample skill vector z ~ p(z)."""
        # Your sampling logic here
        pass

    def update(self, observation_batch, **kwargs) -> dict:
        """Update algorithm (e.g., train discriminator)."""
        # Your training logic here
        return {"loss": loss_value, "accuracy": acc}

    # ... implement other required methods
```

### Step 2: Add Configuration Class

In [`rl_cfg.py`](../exts/d3_skill_discovery/d3_skill_discovery/d3_rsl_rl/rl_cfg.py), define your algorithm's config:

```python
@configclass
class MyUSDAlgorithmCfg:
    """Configuration for MyUSDAlgorithm."""
    learning_rate: float = 3e-4
    hidden_dims: list[int] = [256, 256]
    # ... your hyperparameters
```

Then add it to `RslRlFactorizedUSDAlgorithmCfg`.

### Step 3: Register in `FACTOR_USD`

Extend [`factoized_unsupervised_skill_discovery.py`](d3_rsl_rl/intrinsic_motivation/factoized_unsupervised_skill_discovery.py) to instantiate your algorithm when specified in the environment config.

### Step 4: Update Environment Config

In your environment's configuration:

```python
factors = {
    "my_factor": (["joint_pos", "joint_vel"], "my_algorithm")
}
skill_dims = {"my_factor": 8}
resampling_intervals = {"my_factor": 1000}
usd_alg_extra_cfg = {"my_factor": {"learning_rate": 1e-3}}
```

---

## 📚 Additional Resources

- **Original RSL-RL Repository**: [github.com/leggedrobotics/d3_rsl_rl](https://github.com/leggedrobotics/d3_rsl_rl)
- **Main D3 README**: [../README.md](../README.md) — Full project documentation
- **Example Environments**: [`exts/d3_skill_discovery/d3_skill_discovery/tasks/`](../exts/d3_skill_discovery/d3_skill_discovery/tasks/)
- **Training Scripts**: [`scripts/d3_rsl_rl/`](../scripts/d3_rsl_rl/)

---

## ⚖️ License

This extended version maintains the **BSD-3-Clause License** of the original RSL-RL. See [LICENSE](LICENCE) for details.
