# Real Robot Deployment Setup

Quick start guide for deploying π0 on Dobot E6.

## Prerequisites

- Jetson Orin 64GB
- Dobot E6 robot
- Overhead camera (USB or CSI)
- Python 3.10+

## Installation

```bash
# 1. Install dependencies
pip install -r requirements_deployment.txt

# 2. Download π0 model
python scripts/deployment/download_pi0_model.py \
    --model pi0-bridge \
    --output checkpoints

# 3. Test camera
python scripts/deployment/test_camera.py
```

## Data Collection

```bash
# Start teleoperation interface
python scripts/deployment/collect_data.py \
    --task pushing \
    --episodes 50

# View collected episodes
python scripts/deployment/visualize_episodes.py \
    --data data/pushing
```

## Fine-tuning

```bash
# Fine-tune π0 on Dobot E6 data
python scripts/deployment/finetune_pi0.py \
    --checkpoint checkpoints/pi0-bridge \
    --data data/pushing \
    --task pushing \
    --epochs 10 \
    --output checkpoints/pushing_dobot_e6
```

## Deployment

```bash
# Deploy on robot
python scripts/deployment/ robot.py \
    --checkpoint checkpoints/pushing_dobot_e6 \
    --task pushing \
    --eval-episodes 20
```

## Directory Structure

```
motion-vla/
├── checkpoints/
│   ├── pi0-bridge/          # Pre-trained from HuggingFace
│   └── pushing_dobot_e6/    # Fine-tuned for Dobot
├── data/
│   ├── pushing/             # Collected episodes
│   └── pick_place/
├── scripts/deployment/
│   ├── download_pi0_model.py
│   ├── collect_data.py
│   ├── finetune_pi0.py
│   └── deploy_robot.py
└── src/motion_vla/
    ├── data_collection/
    │   └── episode_recorder.py
    └── deployment/
        └── robot_interface.py
```

## Troubleshooting

See `docs/planning/2026-01-10/pi0_adaptation_plan.md` for detailed setup instructions.
