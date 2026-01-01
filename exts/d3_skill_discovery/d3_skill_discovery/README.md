## Environments

The `tasks` directory provides two categories of environments: **unsupervised skill discovery** and **downstream tasks**.

### Unsupervised Skill Discovery

ANYmal-D environments for learning unsupervised skill discovery:

* **`anymal_usd_env_cfg.py`** - Basic low-level skill learning setup with robot on rough terrain (as described in the paper)
* **`anymal_hl_usd_env_cfg.py`** - High-level skill learning environment (requires a pretrained low-level skill policy)
* **`anymal_hl_usd_box_env_cfg.py`** - High-level skill learning with an interactive movable box

### Downstream Tasks

Evaluation environments for testing learned skills:

* **`goal_tracking/`** - Goal-reaching navigation tasks on rough terrain
* **`pedipulation/`** - Precise foot positioning and manipulation tasks
* **`velocity_tracking/`** - Velocity tracking and locomotion control tasks
