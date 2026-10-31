# Common renderer and Quick, Draw! preprocessing

- `quickdraw_{train,test}_{64,128}.png` show random drawings per category, drawn by `oekaki.render.render_strokes`
  (centred, longer side 0.8 of the image, line width = size/32). Dots, such as smiley-face eyes, are drawn as discs.
- `quickdraw_stats.json` gives, per category:
  - drawings read;
  - drawings kept after the filter (recognized, ≤ 8 strokes, ≤ 160 points, extent ≥ 40 / 256);
  - split sizes (train capped at 60k, test at 5k; test = `key_id % 10 == 0`).
