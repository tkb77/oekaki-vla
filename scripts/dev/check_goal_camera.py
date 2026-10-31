"""Pre-check (inference only): the goal image as a third SmolVLA camera -- feature order, resize,
value range, tokens per image, a forward pass. Saves the three prepared input images."""
import sys, torch
sys.path.insert(0, "scripts")
from train_smolvla import build_policy
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from oekaki.config import resolve
cfg = dict(base_model="lerobot/smolvla_base", chunk_size=50, n_action_steps=5, lr=1e-4, warmup_steps=10, decay_steps=100, decay_lr=1e-6)
root = resolve("data/goal_v1/G")
meta = LeRobotDatasetMetadata("local/oekaki_goal_v1", root=root)
policy, pre, post, pcfg = build_policy(cfg, meta, "cuda")
print("image features:", list(pcfg.image_features))
ds = LeRobotDataset("local/oekaki_goal_v1", root=root, delta_timestamps=resolve_delta_timestamps(pcfg, meta))
b = torch.utils.data.default_collate([ds[i] for i in (0, 200)])
print({k: tuple(v.shape) for k, v in b.items() if hasattr(v, "shape")})
print("goal range", b["observation.images.goal"].min().item(), b["observation.images.goal"].max().item())
bb = pre(b)
imgs, masks = policy.prepare_images(bb)
print("prepared images:", [tuple(i.shape) for i in imgs])
m = policy.model
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    toks = [m.vlm_with_expert.embed_image(i) for i in imgs]
    print("tokens per image:", [tuple(t.shape) for t in toks])
    loss, _ = policy.forward(bb)
print("loss", float(loss))
import torchvision; torchvision.utils.save_image(torch.cat([(i[:1] + 1) / 2 for i in imgs]).float(), "results/goal/precheck/precheck_inputs.png", nrow=3)
print("peak mem GiB", torch.cuda.max_memory_allocated() / 2**30)
