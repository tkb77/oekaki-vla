---
license: apache-2.0
tags:
  - quickdraw
  - sketch-generation
  - diffusion
  - ddpm
  - class-conditional
  - classifier-free-guidance
library_name: pytorch
pipeline_tag: image-to-image
---

# QuickDraw Sketch Diffusion

This repository contains a small class-conditional DDPM trained on rasterized Google QuickDraw sketches. The model generates 64x64 grayscale sketches with a black background and white strokes.

It also includes a local FastAPI web app for selecting a class, generating sketches with classifier-free guidance, and optionally recognizing the generated image with the bundled 100-class CNN classifier.

## Files

- `models/diffusion/checkpoint_step_500000.pt`: 500k-step class-conditional diffusion checkpoint.
- `train_quickdraw_ddpm.py`: training, sampling, QuickDraw download, rasterization, and model definitions.
- `quickdraw_app/server.py`: FastAPI generation and recognition API.
- `quickdraw_app/static/`: browser UI.
- `quickdraw_app/cnn_classifier.py`: optional CNN recognition helper.
- `quickdraw_app/models/cnn_residual_100cls_64/best_model.pt`: bundled 100-class 64x64 CNN classifier.
- `assets/samples_step_500000.png`: sample grid from the diffusion checkpoint.

## Model Details

- Dataset: Google QuickDraw `full/simplified` drawings.
- Classes: 100 built-in QuickDraw classes.
- Training set used: 50,000 samples per class, about 5 million sketches total.
- Image size: 64x64.
- Channels: 1 grayscale channel.
- Diffusion steps: 200.
- Architecture: compact conditional U-Net.
- Conditioning: class embedding plus classifier-free guidance.
- Checkpoint step: 500,000.
- Default guidance scale: 3.0.
- Output convention: black background, white sketch strokes.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the local web app:

```bash
uvicorn quickdraw_app.server:app --host 127.0.0.1 --port 7860
```

Then open:

```text
http://127.0.0.1:7860
```

The first generation loads the checkpoint and may take longer. Later requests reuse the loaded model.

## Python Sampling

```python
import torch
from torchvision.utils import save_image

from train_quickdraw_ddpm import SmallConditionalUNet, make_schedule, pick_device, sample

checkpoint_path = "models/diffusion/checkpoint_step_500000.pt"
device = pick_device()
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

classes = checkpoint["classes"]
image_size = int(checkpoint["image_size"])
timesteps = int(checkpoint["timesteps"])
base_channels = int(checkpoint["base_channels"])

model = SmallConditionalUNet(len(classes), base_channels=base_channels).to(device)
model.load_state_dict(checkpoint.get("model_unwrapped") or checkpoint["model"])
model.eval()

schedule = make_schedule(timesteps, device)
class_name = "cat"
label = torch.tensor([classes.index(class_name)], device=device)

with torch.no_grad():
    image = sample(
        model,
        label,
        image_size,
        schedule,
        timesteps,
        device,
        guidance_scale=3.0,
    )

save_image((image + 1) / 2, "cat.png")
```

## Training Example

The final run used 100 classes, 50,000 samples per class, 64x64 images, CFG dropout, and 500k optimization steps. A similar run can be started with:

```bash
python train_quickdraw_ddpm.py \
  --num-classes 100 \
  --samples-per-class 50000 \
  --image-size 64 \
  --line-width 2 \
  --batch-size 512 \
  --steps 500000 \
  --timesteps 200 \
  --base-channels 64 \
  --cfg-drop-prob 0.1 \
  --guidance-scale 3.0 \
  --data-parallel
```

To resume:

```bash
python train_quickdraw_ddpm.py \
  --resume models/diffusion/checkpoint_step_500000.pt \
  --steps 550000 \
  --batch-size 512 \
  --data-parallel
```

## Evaluation Note

Using the bundled 100-class 64x64 CNN classifier on 400 generated samples, the diffusion checkpoint reached about 36.00% top-1 and 71.75% top-5 agreement. This is a rough model-to-model sanity check, not a human preference score.

## Limitations

This is a compact 64x64 sketch model. It is useful as a QuickDraw-style benchmark and interactive demo, but it is not a high-resolution image generator. Some classes are visually ambiguous, and CNN agreement can be poor for categories with similar silhouettes.

## Data

The training script downloads examples from the public Google QuickDraw dataset:

```text
https://storage.googleapis.com/quickdraw_dataset/full/simplified/{word}.ndjson
```
