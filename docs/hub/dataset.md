---
license: cc-by-4.0
task_categories:
- robotics
size_categories:
- 100K<n<1M
tags:
- LeRobot
- so101
- mujoco
- simulation
- drawing
- goal-image
configs:
- config_name: default
  data_files: data/*/*.parquet
---

# Oekaki goal v1: an SO-101 drawing goal images with a pen (simulation)

999 expert episodes (421k frames, 30 fps) of a simulated SO-101 arm with a spring-loaded pen. In each episode
it draws one Quick, Draw! drawing at a random place, size and angle on the paper. LeRobot v3.0 format.

Used to train [{{NAMESPACE}}/smolvla-oekaki-goal](https://huggingface.co/{{NAMESPACE}}/smolvla-oekaki-goal).
Code and generator: {{CODE_URL}} (`scripts/generate_goal_dataset.py`, `configs/datagen_goal_v1.yaml`)

## Features

| key | shape | content |
|---|---|---|
| `observation.images.top` | 256×256 video (AV1) | overhead camera |
| `observation.images.wrist` | 256×256 video (AV1) | wrist camera |
| `observation.images.goal` | 256×256 video (AV1) | goal image: the drawing on the paper (constant within an episode) |
| `observation.state` | 8 | `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper, contact, compression_mm` |
| `action` | 6 | joint targets (the clean expert action) |
| task | text | 24 instructions; 30 % do not name the category (e.g. "reproduce the reference picture") |

## How it was made

- Scene: MuJoCo Menagerie SO-101 with a pen holder. The pen slides in the holder (0–10 mm, 300 N/m spring), and
  ink is drawn into the paper texture from the tip contacts. The paper height is random (±20 mm).
- Goals: Quick, Draw! training-split drawings of circle, square, triangle, star and smiley face (recognized,
  ≤ 8 strokes). Random size (40–90 % of the drawable area), rotation (±15°) and mirroring.
- Expert: IK, with contact-driven height estimation from the pen compression (it never reads the true paper
  height). DART-style height noise is added to the executed action; the clean action is recorded.

## Use

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("{{NAMESPACE}}/oekaki-goal-v1")
```

## License

CC BY 4.0. The goal drawings are derived from the Quick, Draw! dataset (Google Creative Lab, CC BY 4.0,
https://github.com/googlecreativelab/quickdraw-dataset). The SO-101 model is from MuJoCo Menagerie (Apache-2.0).
