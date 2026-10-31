---
license: apache-2.0
base_model: lerobot/smolvla_base
library_name: lerobot
pipeline_tag: robotics
datasets:
- {{NAMESPACE}}/oekaki-goal-v1
tags:
- lerobot
- smolvla
- so101
- mujoco
- simulation
- drawing
- goal-image
---

# SmolVLA Oekaki G: draw a goal image with a pen

SmolVLA fine-tuned to draw on paper with a simulated SO-101 arm holding a spring-loaded pen (MuJoCo). Besides
the instruction (e.g. `"draw a star"`), it sees the **goal image** (the drawing placed on the paper) as a third
camera, and copies it.

Code, simulation and evaluation: {{CODE_URL}}

![SO-101 drawing a smiley face in MuJoCo](robot_drawing.png)

## Inputs and outputs

| key | shape | content |
|---|---|---|
| `observation.images.top` | 3×256×256 | overhead camera |
| `observation.images.wrist` | 3×256×256 | wrist camera |
| `observation.images.goal` | 3×256×256 | goal image (black lines on white, in paper coordinates) |
| `observation.state` | 8 | 5 joints + gripper, pen contact (0/1), pen compression (mm) |
| task | text | instruction; 30 % of the training instructions do not name the category |
| `action` | 50×6 | chunk of joint targets at 30 Hz |

## Training

- Base model: [lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base).
- Data: [{{NAMESPACE}}/oekaki-goal-v1](https://huggingface.co/datasets/{{NAMESPACE}}/oekaki-goal-v1), 999 expert episodes (421k frames).
- 15k steps, effective batch 64.
- Two extensions from the code repository, stored in `oekaki_extras.json`:
  - training-time real-time chunking;
  - an elapsed-time input.

  **The checkpoint therefore needs the code repository** (`oekaki/policy/`); plain LeRobot cannot load it as is.

## Results (simulation)

- Test goals: held-out Quick, Draw! drawings, 250 episodes per setting.
- Inference: σ = 0, replan every 30 steps, latency 3, real-time chunking.

| setting | this model | same recipe without the goal image |
|---|---|---|
| goal IoU, true goal | **0.51** | 0.19 |
| position error, true goal | **5.1 mm** | 24.9 mm |
| goal IoU, goal imagined by [the diffusion model](https://huggingface.co/{{NAMESPACE}}/oekaki-sketch-diffusion) | **0.53** | — |

![generated picture, goal on the paper, robot drawing](goal_demo_strip.png)

## Use

```bash
git clone {{CODE_URL}} oekaki-vla && cd oekaki-vla
uv venv --python 3.12 && uv sync
export MUJOCO_GL=egl
.venv/bin/python scripts/imagine_and_draw.py --prompt "draw a star" --seed 1 --out outputs/demo_star \
    --policy {{NAMESPACE}}/smolvla-oekaki-goal --generator {{NAMESPACE}}/oekaki-sketch-diffusion
```

The script downloads both models, imagines a star, and lets this model draw it.

## Limitations

- Simulation only, five categories (circle, square, triangle, star, smiley face), one drawing per goal image.
- Shape details such as thin star tips or smiley eyes are often broken.

## License

Apache-2.0, like the base model. The training goals come from the Quick, Draw! dataset
(Google Creative Lab, CC BY 4.0).
