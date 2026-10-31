# Dataset goal_v1 (data/goal_v1/G)

999 successful expert episodes of 1000 (one smiley face with a disturbance stalled in PRELOAD), 421,134 frames at 30 fps, 3.5 GB; video/data frame counts verified for the three streams; goal stream decodes to the goal PNG (IoU 0.999).

| category    |   episodes |   seconds_mean |   seconds_max |   strokes_mean |   named |   disturbance |
|:------------|-----------:|---------------:|--------------:|---------------:|--------:|--------------:|
| circle      |        200 |          11.94 |         20.07 |           1.11 |    0.7  |          0.26 |
| smiley face |        199 |          18.57 |         40    |           4.42 |    0.7  |          0.26 |
| square      |        200 |          12.93 |         22.57 |           1.3  |    0.72 |          0.28 |
| star        |        200 |          14.45 |         25.2  |           1.38 |    0.74 |          0.21 |
| triangle    |        200 |          12.5  |         27.73 |           1.53 |    0.7  |          0.28 |

`dataset_examples.png`: goal | expert canvas, label = instruction type and size fraction.
