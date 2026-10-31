#!/usr/bin/env python3
"""Train a small class-conditional DDPM on rasterized QuickDraw sketches."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from tqdm import tqdm


QUICKDRAW_URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/{word}.ndjson"
QUICKDRAW_100_CLASSES = [
    "aircraft carrier", "airplane", "alarm clock", "ambulance", "angel",
    "animal migration", "ant", "anvil", "apple", "arm", "asparagus", "axe",
    "backpack", "banana", "bandage", "barn", "baseball", "baseball bat",
    "basket", "basketball", "bat", "bathtub", "beach", "bear", "beard",
    "bed", "bee", "belt", "bench", "bicycle", "binoculars", "bird",
    "birthday cake", "blackberry", "blueberry", "book", "boomerang",
    "bottlecap", "bowtie", "bracelet", "brain", "bread", "bridge",
    "broccoli", "broom", "bucket", "bulldozer", "bus", "bush", "butterfly",
    "cactus", "cake", "calculator", "calendar", "camel", "camera",
    "camouflage", "campfire", "candle", "cannon", "canoe", "car", "carrot",
    "castle", "cat", "ceiling fan", "cello", "cell phone", "chair",
    "chandelier", "church", "circle", "clarinet", "clock", "cloud",
    "coffee cup", "compass", "computer", "cookie", "cooler", "couch",
    "cow", "crab", "crayon", "crocodile", "crown", "cruise ship", "cup",
    "diamond", "dishwasher", "diving board", "dog", "dolphin", "donut",
    "door", "dragon", "dresser", "drill", "drums", "duck",
]


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def render_drawing(drawing: list, image_size: int, line_width: int) -> torch.Tensor:
    image = Image.new("L", (image_size, image_size), 255)
    draw = ImageDraw.Draw(image)
    scale = image_size / 256.0

    for stroke in drawing:
        xs, ys = stroke
        points = [(round(x * scale), round(y * scale)) for x, y in zip(xs, ys)]
        if len(points) >= 2:
            draw.line(points, fill=0, width=line_width)
        elif len(points) == 1:
            x, y = points[0]
            r = max(1, line_width // 2)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=0)

    data = torch.tensor(list(image.tobytes()), dtype=torch.uint8).view(1, image_size, image_size)
    return 255 - data


class QuickDrawSketches(Dataset):
    def __init__(
        self,
        classes: list[str],
        samples_per_class: int,
        image_size: int,
        line_width: int,
        recognized_only: bool = True,
        download_retries: int = 5,
    ) -> None:
        self.classes = classes
        total_samples = len(classes) * samples_per_class
        images = torch.empty(total_samples, 1, image_size, image_size, dtype=torch.uint8)
        labels = torch.empty(total_samples, dtype=torch.long)

        for label, word in enumerate(classes):
            quoted = urllib.parse.quote(word, safe="")
            url = QUICKDRAW_URL.format(word=quoted)
            for attempt in range(1, download_retries + 1):
                loaded = 0
                try:
                    with urllib.request.urlopen(url, timeout=60) as response:
                        for raw_line in response:
                            item = json.loads(raw_line)
                            if recognized_only and not item.get("recognized", False):
                                continue
                            index = label * samples_per_class + loaded
                            images[index] = render_drawing(item["drawing"], image_size, line_width)
                            labels[index] = label
                            loaded += 1
                            if loaded >= samples_per_class:
                                break
                    if loaded >= samples_per_class:
                        print(f"loaded class {label + 1}/{len(classes)}: {word} ({loaded})", flush=True)
                        break
                    raise RuntimeError(f"Only loaded {loaded} samples for class {word!r}")
                except Exception:
                    if attempt == download_retries:
                        raise
                    time.sleep(min(2 ** attempt, 30))
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index].float() / 127.5 - 1.0
        return image, self.labels[index]


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SmallConditionalUNet(nn.Module):
    def __init__(self, num_classes: int, base_channels: int = 64, emb_dim: int = 256) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.null_label = num_classes
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.class_emb = nn.Embedding(num_classes + 1, emb_dim)

        c = base_channels
        self.in_conv = nn.Conv2d(1, c, 3, padding=1)
        self.down1 = ResBlock(c, c, emb_dim)
        self.downsample1 = nn.Conv2d(c, c * 2, 4, stride=2, padding=1)
        self.down2 = ResBlock(c * 2, c * 2, emb_dim)
        self.downsample2 = nn.Conv2d(c * 2, c * 4, 4, stride=2, padding=1)
        self.mid1 = ResBlock(c * 4, c * 4, emb_dim)
        self.mid2 = ResBlock(c * 4, c * 4, emb_dim)
        self.upsample2 = nn.ConvTranspose2d(c * 4, c * 2, 4, stride=2, padding=1)
        self.up2 = ResBlock(c * 4, c * 2, emb_dim)
        self.upsample1 = nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1)
        self.up1 = ResBlock(c * 2, c, emb_dim)
        self.out_norm = nn.GroupNorm(min(8, c), c)
        self.out_conv = nn.Conv2d(c, 1, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(t) + self.class_emb(y)
        x0 = self.in_conv(x)
        x1 = self.down1(x0, emb)
        x2 = self.down2(self.downsample1(x1), emb)
        x3 = self.mid2(self.mid1(self.downsample2(x2), emb), emb)
        x = self.upsample2(x3)
        x = self.up2(torch.cat([x, x2], dim=1), emb)
        x = self.upsample1(x)
        x = self.up1(torch.cat([x, x1], dim=1), emb)
        return self.out_conv(F.silu(self.out_norm(x)))


@dataclass
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    posterior_variance: torch.Tensor


def make_schedule(timesteps: int, device: torch.device) -> DiffusionSchedule:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, device=device)
    alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = betas.clamp(1e-4, 0.999)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        alphas_cumprod_prev=alphas_cumprod_prev,
        sqrt_alphas_cumprod=torch.sqrt(alphas_cumprod),
        sqrt_one_minus_alphas_cumprod=torch.sqrt(1.0 - alphas_cumprod),
        posterior_variance=posterior_variance,
    )


def extract(values: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    return values.gather(0, t).view(t.shape[0], *((1,) * (len(x_shape) - 1)))


def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, schedule: DiffusionSchedule) -> torch.Tensor:
    return (
        extract(schedule.sqrt_alphas_cumprod, t, x0.shape) * x0
        + extract(schedule.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
    )


@torch.no_grad()
def sample(
    model: nn.Module,
    labels: torch.Tensor,
    image_size: int,
    schedule: DiffusionSchedule,
    timesteps: int,
    device: torch.device,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    model.eval()
    x = torch.randn(labels.shape[0], 1, image_size, image_size, device=device)
    null_labels = torch.full_like(labels, unwrap_model(model).null_label)
    for step in tqdm(reversed(range(timesteps)), total=timesteps, desc="sample"):
        t = torch.full((labels.shape[0],), step, device=device, dtype=torch.long)
        if guidance_scale == 1.0:
            pred_noise = model(x, t, labels)
        else:
            pred_uncond = model(x, t, null_labels)
            pred_cond = model(x, t, labels)
            pred_noise = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        alpha_bar_t = extract(schedule.alphas_cumprod, t, x.shape)
        alpha_bar_prev = extract(schedule.alphas_cumprod_prev, t, x.shape)
        beta_t = extract(schedule.betas, t, x.shape)
        alpha_t = extract(schedule.alphas, t, x.shape)
        pred_x0 = (x - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
        pred_x0 = pred_x0.clamp(-1, 1)
        coef_x0 = beta_t * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar_t)
        coef_xt = (1.0 - alpha_bar_prev) * torch.sqrt(alpha_t) / (1.0 - alpha_bar_t)
        mean = coef_x0 * pred_x0 + coef_xt * x
        if step > 0:
            variance = extract(schedule.posterior_variance, t, x.shape)
            x = mean + torch.sqrt(variance.clamp_min(1e-20)) * torch.randn_like(x)
        else:
            x = mean
    return x.clamp(-1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", nargs="+", default=["cat", "dog", "house", "airplane"])
    parser.add_argument("--num-classes", type=int, default=0)
    parser.add_argument("--samples-per-class", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/quickdraw-ddpm"))
    parser.add_argument("--sample-every", type=int, default=250)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--cfg-drop-prob", type=float, default=0.1)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--download-retries", type=int, default=5)
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--sample-num-classes", type=int, default=16)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.num_classes:
        if args.num_classes > len(QUICKDRAW_100_CLASSES):
            raise ValueError(f"--num-classes supports at most {len(QUICKDRAW_100_CLASSES)} built-in classes")
        args.classes = QUICKDRAW_100_CLASSES[: args.num_classes]
    resume_checkpoint = None
    if args.resume is not None:
        resume_checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        args.classes = list(resume_checkpoint["classes"])
        args.image_size = int(resume_checkpoint["image_size"])
        args.timesteps = int(resume_checkpoint["timesteps"])
        args.base_channels = int(resume_checkpoint["base_channels"])

    run_dir = args.out_dir / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device()

    print(f"device: {device}")
    print(f"classes: {args.classes}")
    print("loading and rasterizing QuickDraw samples...")
    dataset = QuickDrawSketches(
        classes=args.classes,
        samples_per_class=args.samples_per_class,
        image_size=args.image_size,
        line_width=args.line_width,
        download_retries=args.download_retries,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = SmallConditionalUNet(len(args.classes), base_channels=args.base_channels).to(device)
    if args.data_parallel:
        if device.type != "cuda" or torch.cuda.device_count() < 2:
            raise RuntimeError("--data-parallel requires at least two visible CUDA devices")
        model = nn.DataParallel(model)
        print(f"data_parallel_devices: {torch.cuda.device_count()}")
    schedule = make_schedule(args.timesteps, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_step = 0
    if resume_checkpoint is not None:
        state_dict = resume_checkpoint.get("model_unwrapped") or resume_checkpoint["model"]
        unwrap_model(model).load_state_dict(state_dict)
        opt.load_state_dict(resume_checkpoint["optimizer"])
        start_step = int(resume_checkpoint["step"])
        print(f"resumed checkpoint: {args.resume} at step {start_step}", flush=True)

    with (run_dir / "config.json").open("w") as f:
        json.dump(
            vars(args) | {"device": str(device), "run_dir": str(run_dir), "start_step": start_step},
            f,
            indent=2,
            default=str,
        )

    data_iter = iter(loader)
    pbar = tqdm(range(start_step + 1, args.steps + 1), desc="train")
    last_loss = None
    for step in pbar:
        try:
            x0, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x0, labels = next(data_iter)

        x0 = x0.to(device)
        labels = labels.to(device)
        if args.cfg_drop_prob > 0:
            drop_mask = torch.rand(labels.shape, device=device) < args.cfg_drop_prob
            labels_for_model = labels.masked_fill(drop_mask, unwrap_model(model).null_label)
        else:
            labels_for_model = labels
        t = torch.randint(0, args.timesteps, (x0.shape[0],), device=device)
        noise = torch.randn_like(x0)
        xt = q_sample(x0, t, noise, schedule)
        pred_noise = model(xt, t, labels_for_model)
        loss = F.mse_loss(pred_noise, noise)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        last_loss = float(loss.item())
        pbar.set_postfix(loss=f"{last_loss:.4f}")

        if step % args.sample_every == 0 or step == args.steps:
            sample_class_count = min(args.sample_num_classes, len(args.classes))
            sample_labels = torch.arange(sample_class_count, device=device).repeat_interleave(4)
            images = sample(
                model,
                sample_labels,
                args.image_size,
                schedule,
                args.timesteps,
                device,
                guidance_scale=args.guidance_scale,
            )
            save_image((images + 1) / 2, run_dir / f"samples_step_{step:06d}.png", nrow=4)
            model.train()

        if step % args.save_every == 0 or step == args.steps:
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_unwrapped": unwrap_model(model).state_dict(),
                    "optimizer": opt.state_dict(),
                    "step": step,
                    "classes": args.classes,
                    "image_size": args.image_size,
                    "timesteps": args.timesteps,
                    "base_channels": args.base_channels,
                    "cfg_drop_prob": args.cfg_drop_prob,
                    "guidance_scale": args.guidance_scale,
                    "loss": last_loss,
                },
                run_dir / f"checkpoint_step_{step:06d}.pt",
            )

    print(f"done: {run_dir}")


if __name__ == "__main__":
    main()
