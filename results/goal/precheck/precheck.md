# Pre-check part 1: the goal image as a third SmolVLA input (inference only, no training)

`scripts/dev/check_goal_camera.py`, run with smolvla_base on a small goal_v1 dataset:

- Input features are taken from the dataset: `observation.images.top`, `observation.images.wrist`,
  `observation.images.goal`.
  - SmolVLA has no limit on the number of cameras (`empty_cameras = 0`, so no padding cameras).
  - The images go to the VLM in this order.
- **Resize:** every image (256×256) is resized to 512×512 with aspect-preserving padding. Pixels go from
  [0, 1] to [-1, 1] for SigLIP.
  - The visual normalisation mode is identity, so dataset statistics are not applied to images.
- **Tokens:** each image becomes 64 visual tokens of width 960 (SigLIP 16-px patches, then a 4×4 pixel
  shuffle).
  - One token covers 1/8 × 1/8 of the paper (about 19 mm).
  - The goal line is 8 px wide at 256 px (1/32 of the paper, about 4.7 mm).
  - A dot is a disc of the same diameter.
  - Dots stay visible in the prepared input (`precheck_inputs.png`, right).
- **Forward pass:** loss 0.47 at initialisation; peak memory 1.0 GiB for batch 2.

## Part 2: overfitting on 30 episodes (2026-10-28)
Setup:
- Model G was trained on the first 6 episodes of each category (`configs/train_goal_overfit_G.yaml`, 4k steps).
- Training ran on an AWS g6e instance (1 L40S) and took 33 min. Final loss 0.023, val 0.031.
- `scripts/eval_goal_overfit.py` replays each of the 30 episodes twice: with its own goal, and with the goal of
  another training episode of the same category (`overfit_summary.md`, `overfit_sheet.png`).

| goal | IoU with the goal shown | IoU with the other goal | Chamfer (mm) | centroid offset (mm) | size ratio |
|---|---|---|---|---|---|
| own | 0.60 | 0.17 | 4.1 | 6.3 | 0.91 |
| swapped | 0.21 | 0.15 | 31.3 | 34.6 | 0.68 |

- **Own goal: passes.** G draws the shape at the goal's position (offset 6 mm) and size (ratio 0.91).
- **Swapped goal: G does not follow it.**
  - The drawings are small and broken, and 5 of 30 are empty.
  - The drawings do not repeat the episode's original drawing either: IoU with the original goal is only 0.15.
  - So the goal image changes what the policy does; it is read, not ignored.
  - With 30 episodes, though, G has memorised each (goal, start state, instruction) combination instead of
    learning to copy. An unseen combination is out of distribution, which is the expected result of an
    overfitting test.
- **Consequence:** generalisation to new goals can only be judged on the full-data model G (949 episodes), in
  evaluation E1. Watch for the same failure there: small, broken or empty drawings on unseen goals.
- **Schedule deviation:** to keep the deadline, the full N and G runs were started at the same time as this
  check instead of after it.
