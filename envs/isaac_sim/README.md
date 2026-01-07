# Isaac Sim Wiping Environment

This directory contains the Isaac Sim simulation environment for the Wiping with Adverb Control task.

## Directory Structure

```
isaac_sim/
├── wiping_env.py          # Main wiping task environment
├── dobot_e6_robot.py      # Dobot E6 robot wrapper
├── dirt_simulator.py      # Particle-based dirt simulation
├── vision_metrics.py      # Vision-based cleaning rate evaluation
└── README.md              # This file
```

## Quick Start

```python
from envs.isaac_sim.wiping_env import WipingEnv

env = WipingEnv()
obs = env.reset()

# Execute adverb-conditioned wiping
action = policy.predict(obs, instruction="Wipe the table gently")
obs, reward, done, info = env.step(action)
```

## Features

- **Particle-based dirt simulation** using PhysX
- **Vision-based metrics** for cleaning rate evaluation
- **Adverb-conditioned control** integration
- **Domain randomization** for Sim2Real transfer
