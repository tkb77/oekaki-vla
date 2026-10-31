# E5 per-half check (judge CNN on the left / right halves)

both_drawn: ink in both halves; halves_correct: fraction of halves whose class matches the goal half; both_correct: both halves match.

| agent   |   n |   both_drawn |   halves_correct |   both_correct |
|:--------|----:|-------------:|-----------------:|---------------:|
| G       | 250 |        1     |            0.334 |          0.048 |
| G_2x    | 250 |        0.98  |            0.388 |          0.108 |
| G_pair  | 250 |        0.928 |            0.312 |          0.088 |
| G_long  | 250 |        1     |            0.388 |          0.104 |
