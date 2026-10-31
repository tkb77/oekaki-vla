# Expert and layout

`scripts/validate_goal_expert.py --n 20`: 20 episodes per category, Quick, Draw! train-split drawings.

Randomised per episode:
- size 0.41–0.90 of the drawable area;
- rotation ±15°;
- mirroring;
- position;
- paper height ±19 mm;
- disturbance in about 25 % of episodes;
- DART noise.

Results:
- **Episodes drawn:** 100 / 100 (`expert.json`, `expert_summary.md`).
- **Goal vs canvas, all categories:** coverage 1.00, dilated IoU 0.95–0.97, Chamfer 0.17–0.34 px at 128 px (≈ 0.2–0.4 mm).
- **Smiley-face dots:** drawn as taps.
- `expert_sheet.png` shows goal vs drawn canvas.
