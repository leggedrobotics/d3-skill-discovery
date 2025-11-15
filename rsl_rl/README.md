# Unsupervised Skill Discovery (USD)

This directory contains **unsupervised skill discovery algorithms** built on top of
[rsl_rl](https://github.com/leggedrobotics/rsl_rl) (version [2.2.0](https://github.com/leggedrobotics/rsl_rl/releases/tag/v2.2.0)).

The framework supports any USD algorithm that:

* Provides a **reward function** for a transition tuple
* Can be **trained with on-policy** data

Currently implemented algorithms:

* **DIAYN** — `rsl_rl/intrinsic_motivation/diayn.py`
* **METRA** — `rsl_rl/intrinsic_motivation/metra.py`

New algorithms can be added by subclassing `BaseSkillDiscovery` located at:
`rsl_rl/intrinsic_motivation/base_skill_discovery.py`

Multiple algorithms are managed by the `FACTOR_USD` class:
`rsl_rl/intrinsic_motivation/factoized_unsupervised_skill_discovery.py`

---

## Environment Requirements

To enable factorized USD, the environment must define the following attributes:

```python
factors: dict[str, tuple[list[str], Literal["metra", "diayn"]]]
skill_dims: dict[str, int]
resampling_intervals: dict[str, int]
usd_alg_extra_cfg: dict[str, dict]
```

Additionally, the algorithm configuration must contain a `usd` field of type
`RslRlFactorizedUSDAlgorithmCfg`, located in:

```
IsaacLab/source/extensions/omni.isaac.lab_tasks/
    omni/isaac/lab_tasks/utils/wrappers/rsl_rl/rl_cfg.py
```

---

## Adding a New USD Algorithm

Follow these steps to integrate a new unsupervised skill discovery algorithm:

### 1. Implement the Algorithm

Create a class that subclasses `BaseSkillDiscovery`.

### 2. Add a Configuration Class

Define and register a configuration class for your algorithm in:

```
IsaacLab/source/extensions/omni.isaac.lab_tasks/
    omni/isaac/lab_tasks/utils/wrappers/rsl_rl/rl_cfg.py
```

Add it to the `RslRlFactorizedUSDAlgorithmCfg` class.

### 3. Update `FACTOR_USD`

Extend the `FACTOR_USD` class so it can initialize and manage your new algorithm.

### 4. Update the Environment Configuration

Modify the following fields to reflect your new algorithm:

* `factors`
* `skill_dims`
* `resampling_intervals`
* `usd_alg_extra_cfg`

---
