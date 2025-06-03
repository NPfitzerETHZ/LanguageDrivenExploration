# Language‑Driven Exploration

A modular research framework for training reinforcement‑learning agents that follow natural‑language instructions and can be deployed on physical **DJI RoboMaster EP** robots.

## Table of Contents

* [Overview](#overview)
* [Prerequisites](#prerequisites)
* [Install & Training](#install--training)
* [Deployment](#deployment)

  * [ROS 2 Workspace Setup](#ros-2-workspace-setup)
  * [Launch Nodes](#launch-nodes)
* [Citing](#citing)
* [License](#license)

## Overview

Language‑Driven Exploration (LDE) provides:

* **BenchMARL‑ or TorchRL-based training** pipelines with language‑conditioned policies.
* Ready‑to‑use **ROS 2 launch files** to run trained policies on RoboMaster platforms.

<p align="center">
  <img src="docs/images/lde_architecture.svg" width="600" alt="LDE architecture diagram"/>
</p>

## Prerequisites

| Component | Version               |
| --------- | --------------------- |
| Python    | ≥ 3.9                 |
| pip       | ≥ 23                  |
| ROS 2     | Humble (Ubuntu 22.04) |
| CMake     | ≥ 3.22                |

> **Note**
> Deployment is tested on ROS 2 Humble. For ROS 2 Foxy you may need to adjust message definitions.

---

## Install & Training

### 1 · Clone and install

```bash
# Get the source
git clone https://github.com/NPfitzerETHZ/LanguageDrivenExploration.git
cd LanguageDrivenExploration

# Editable install (adds console script `my-deployment`)
pip install -e .
```

### 2 · Start a training run

```bash
python trainers/benchmarl_train.py
```

By default this launches a MAPPO experiment defined in `configs/benchmarl_mappo.yaml`; logs and checkpoints are saved under `outputs/`.

---

## Deployment

### ROS 2 Workspace Setup

```bash
# Pick / create your workspace folder
mkdir -p ~/robomaster_ws/src
cd ~/robomaster_ws/src

# Clone dependencies
git clone --branch ros2-devel --single-branch git@github.com:unl-nimbus-lab/Freyja.git
git clone https://github.com/proroklab/ros2_robomaster_msgs.git
git clone https://github.com/NPfitzerETHZ/LanguageDrivenExploration.git

# Build the workspace
cd ..
python -m colcon build --symlink-install --cmake-args -DNO_PIXHAWK=True
source install/setup.bash

# Install Python deps for LDE
cd src/LanguageDrivenExploration
pip install -e .
```

### Launch Nodes

```bash
# 1) Start the robot drivers (choose the correct TF frame)
ros2 launch src/Freyja/freyja_robomaster.launch.yaml tf_myframe:=robomaster_2

# 2) Run the deployment script
python deployment/my_deployment.py \
    config_path=/path/to/deployment_checkpoint_folder \
    config_name=benchmarl_mappo.yaml

#  —or— using the console script installed earlier
my-deployment \
    config_path=/path/to/deployment_checkpoint_folder \
    config_name=benchmarl_mappo.yaml
```

---

## Citing

If you use LDE in your research, please cite:

```text
@misc{fitzer2025language,
  title        = {Language‑Driven Exploration},
  author       = {N. Pfitzer *et al.*},
  howpublished = {GitHub},
  year         = {2025},
  url          = {https://github.com/NPfitzerETHZ/LanguageDrivenExploration}
}
```

## License

This repository is released under the **MIT License**. See [LICENSE](LICENSE) for details.

  
