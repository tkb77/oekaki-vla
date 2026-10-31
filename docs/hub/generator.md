---
license: apache-2.0
library_name: pytorch
tags:
- diffusion
- ddpm
- sketch
- quickdraw
- class-conditional
---

# Oekaki sketch diffusion: a tiny class-conditional line-drawing generator

A 5.0 M-parameter DDPM that draws 64×64 line drawings of five categories: circle, square, triangle, star and
smiley face. In the Oekaki VLA project, it *imagines* the picture that the robot then draws
([{{NAMESPACE}}/smolvla-oekaki-goal](https://huggingface.co/{{NAMESPACE}}/smolvla-oekaki-goal)).

Code: {{CODE_URL}} (`oekaki/generator/`)

![samples after clean-up, one row per category](samples_post.png)

## Model

- U-Net DDPM, 200 timesteps, base width 64.
- Classifier-free guidance (scale 3.0); weights are the EMA weights.
- Sampling: DDIM, 50 steps; the intermediate images can be returned for animation.
- Trained for 100k steps, batch 256, on 60,000 Quick, Draw! drawings per category: recognized drawings, at most
  8 strokes, rendered with the project's common line renderer. A 10 % split by `key_id` is held out.
- Adapted from [lcccluck/sketch-model](https://huggingface.co/lcccluck/sketch-model) (Apache-2.0).

## Quality

- No empty images in 1000 samples.
- A separate judge CNN (99.1 % test accuracy on real drawings) recognises 99.9 % of the generated drawings.

![DDIM trajectory of a star](trajectory_star.png)

## Files

- `last.pt`: a PyTorch checkpoint, a pickle. It holds the EMA weights, the class names and the sampling and
  clean-up settings. Load it only with the code repository: `oekaki.generator.imagine.Imaginer`.
- `config.json`: the training configuration.

```python
from oekaki.config import resolve_ckpt
from oekaki.generator.imagine import Imaginer

im = Imaginer(resolve_ckpt("{{NAMESPACE}}/oekaki-sketch-diffusion", "last.pt"))
out = im.generate("star", seed=0)  # out["image"]: 64x64 uint8, out["segments"]: strokes for the robot
```

## License

Apache-2.0. Trained on the Quick, Draw! dataset (Google Creative Lab, CC BY 4.0,
https://github.com/googlecreativelab/quickdraw-dataset).
