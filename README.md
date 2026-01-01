# Divide, Discover, Deploy: Factorized Skill Learning with Symmetry and Style Priors

<p align="center">
  <img src="docs/overview_fig.png" width="100%" alt="Main figure of D3 paper"/>
</p>

This repository contains the implementation accompanying the CoRL 2025 paper **Divide, Discover, Deploy: Factorized Skill Learning with Symmetry and Style Priors**. The project page is available at: [https://leggedrobotics.github.io/d3-skill-discovery/](https://leggedrobotics.github.io/d3-skill-discovery/)

---


## Repository structure

```
d3-skill-discovery/
├── exts/d3_skill_discovery/    # IsaacLab Environments
├── scripts/                    # Training scripts
└── rsl_rl/                     # Algorithms
```


See [exts/d3_skill_discovery/d3_skill_discovery/README.md](exts/d3_skill_discovery/d3_skill_discovery/README.md) for details on the environments.
See [rsl_rl/README.md](rsl_rl/README.md) for details on the usd algorithm.

---

## Installation

### Prerequisites

1. Install Isaac Lab 2.2 following the official [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html#).

2. Create a new conda environment:

   ```bash
   ./isaaclab.sh --conda d3_env
   conda activate d3_env
   ```

3. Install IsaacLab extensions:

   ```bash
   ./isaaclab.sh --install none
   ```

---

## Install this Extension

Navigate to the project directory and run:

```bash
cd d3-skill-discovery
./install.sh
```

---

## Training

To train an unsupervised skill discovery model:

```bash
python scripts/train.py --task Isaac-USD-Anymal-D-v0 --num_envs 2048 --headless --logger wandb
```

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@inproceedings{cathomen2025d3,
  author    = {Cathomen, Rafael and Mittal, Mayank and Vlastelica, Marin and Hutter, Marco},
  title     = {Divide, Discover, Deploy: Factorized Skill Learning with Symmetry and Style Priors},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2025},
}
```

---

## License

This project is released under the [BSD-3-Clause License](LICENSE).
