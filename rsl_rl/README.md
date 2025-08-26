## Unsupervised Skill Discovery (USD)
This directory contains unsupervised skill discovery algorithms based on [rsl_rl](https://github.com/leggedrobotics/rsl_rl) (version [2.2.0](https://github.com/leggedrobotics/rsl_rl/releases/tag/v2.2.0)).


This framework is compatible with any USD algorithm that 
- Provides a Reward function given a transition tuple
- Can be trained with on-policy data

Currently, DIAYN and METRA are implemented in `rsl_rl/intrinsic_motivation/diayn.py` and `rsl_rl/intrinsic_motivation/metra.py`.
Additional algorithms can be implemented by subclassing from `BaseSkillDiscovery` in `rsl_rl/intrinsic_motivation/base_skill_discovery.py`.
The algorithms are combined in the `FACTOR_USD` class in `/rsl_rl/intrinsic_motivation/factoized_unsupervised_skill_discovery.py`.

For this algorithm to work, the environment additionally needs to provide the attributes:
```python
factors: dict[str, tuple[list[str], Literal["metra", "diayn"]]]
skill_dims: dict[str, int] 
resampling_intervals: dict[str, int] 
usd_alg_extra_cfg: dict[str, dict]
```

The algorithm configuration needs an additional field `usd` (as in unsupervised skill discovery, not to be confused with universal scene description) of type `RslRlFactorizedUSDAlgorithmCfg` defined in
`IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/utils/wrappers/rsl_rl/rl_cfg.py`


### How to add new USD algorithm
1. Implement the algorithm by subclassing from `BaseSkillDiscovery`.
2. Add a configuration class for this algorithm in 
`IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/utils/wrappers/rsl_rl/rl_cfg.py`
and include it as a field in `RslRlFactorizedUSDAlgorithmCfg`
3. Update the `FACTOR_USD` class to also handle your new algorithm
4. In the environment configuration, update the `factors`, `skill_dims`, `resampling_intervals` and `usd_alg_extra_cfg` configs accordingly


