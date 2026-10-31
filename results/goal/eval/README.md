# Evaluation E1–E5

## Setup
- **Episodes:** 50 per category and setting. Goals come from the held-out Quick, Draw! test split.
  Layout, paper height and instruction come from one seed per (setting, category, i).
- **Inference:** ttrtc, replan 30, delay 3, noise 0.
- **Compute:** run on an AWS g6e.24xlarge node (`scripts/cloud_eval_goal.sh`).
  - 25 shards, one per (setting, category), 6 processes per GPU, 31 min in total.
  - `scripts/merge_eval_shards.py` merges the shards.
  - `scripts/report_goal.py` writes the files listed below.

## Files

| file | content |
|---|---|
| `summary.md` | Means with bootstrap 95 % CIs. |
| `per_category.md` | Means per category. |
| `failures.md` | Failure classes. |
| `metrics.png` | Bar charts. |
| `grid_<setting>_<agent>.png` | The 6 worst and 6 best episodes, as goal \| drawn pairs. |
| `episodes.csv` | Every episode. |

## Paired G vs N (same episodes, Wilcoxon signed-rank)

| setting | IoU G − N | G better | position error G − N | G better |
|---|---|---|---|---|
| E1 | +0.31 | 96 % | −19.7 mm | 98 % |
| E3 | +0.38 | 98 % | −32.9 mm | 99 % |
| E4 | +0.32 | 96 % | −18.8 mm | 96 % |

All p < 1e-40 (n = 250 each).

## G, IoU per category

| setting | circle | smiley face | square | star | triangle |
|---|---|---|---|---|---|
| E1 | 0.61 | 0.46 | 0.47 | 0.52 | 0.48 |
| E2 | 0.61 | 0.55 | 0.49 | 0.52 | 0.47 |
| E3 | 0.58 | 0.43 | 0.51 | 0.49 | 0.49 |
| E4 | 0.61 | 0.48 | 0.50 | 0.50 | 0.47 |
| E5 | 0.36 | 0.38 | 0.41 | 0.39 | 0.35 |

## Findings
- **The goal image is used.**
  - G follows position, size and shape. N draws a typical shape near the centre.
  - Without the category name (E4), G is unchanged. N's judge accuracy drops to chance (0.19).
- **Generated goals (E2) work as well as true goals.** Generated drawings are cleaner, so G scores 0.53 on them
  versus 0.51 on true goals.
- **Failures are mostly broken shapes** (92 of 250 episodes in E1). They concentrate on goals far from the
  category's typical form: open shapes, tiny drawings, two squares.
- **E5 (two drawings) fails.** G merges both drawings into one shape, a composition it never saw in training.

## Follow-up models (E1, E2, E5; paired with G, Wilcoxon)

| model | training | E1 IoU | E2 IoU | E5 IoU | E5 halves correct |
|---|---|---|---|---|---|
| G | goal_v1, 999 single, 15k steps | 0.507 | 0.528 | 0.376 | 0.33 |
| G_2x | goal_v2, 1997 single, 15k steps | 0.530 (p = 0.007) | 0.551 (p = 0.002) | 0.383 | 0.39 |
| G_long | goal_v1, 999 single, 30k steps | **0.561** (p = 1e-11) | **0.580** (p = 2e-12) | 0.390 | 0.39 |
| G_pair | goal_v2, 1000 single + 496 pairs, 15k steps | 0.500 (n.s.) | 0.509 (p = 0.01, worse) | **0.523** (p = 3e-35) | 0.31 |

- **E5 halves correct:** the judge CNN classifies the left and right halves of each canvas (`e5_halves.md`).
  - Reference classes come from the goal halves; the judge reads 241 of the 250 goal pairs correctly.
- **Longer training helps the shape detail most.**
- **Two-drawing episodes teach the layout but not the shapes.** G_pair draws two shapes at the two places, but
  each shape is no more often correct (`e5_G_vs_G_pair.png`).
