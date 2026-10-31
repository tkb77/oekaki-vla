"""Course video (1-3 min, MP4, 1280x720 @ 30 fps): starts from a white frame and fades in, then title, concept,
imagine-then-draw demos, G vs N, results, the smoothing story, limitations and credits.

Inputs are the outputs of the pipeline (demo videos, evaluation videos and figures). Sped-up clips carry a
"2x speed" badge, as the course requires. Text is rendered with Noto Sans CJK.

Usage: python scripts/make_course_video.py [--out artifacts/course_video/oekaki_vla.mp4] [--pair-grid PNG]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from oekaki.config import resolve

W, H, FPS = 1280, 720, 30
INK = (27, 33, 64)
SUB = (90, 96, 120)
ACCENT = (43, 58, 158)
FONT_R = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def font(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT_R, size, index=0)  # index 0 = JP


def text_img(img: np.ndarray, items) -> np.ndarray:
    """items: (x, y, text, size, bold, color, anchor)"""
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    for x, y, t, size, bold, color, anchor in items:
        d.text((x, y), t, font=font(size, bold), fill=color, anchor=anchor)
    return np.array(im)


def blank():
    return np.full((H, W, 3), 255, np.uint8)


def card(title, lines=(), secs=5.0, title_size=56):
    items = [(W // 2, H // 2 - 40 - 30 * len(lines), title, title_size, True, INK, "mm")]
    for k, l in enumerate(lines):
        items.append((W // 2, H // 2 + 30 + 52 * k - 30 * len(lines) + 20, l, 30, False, SUB, "mm"))
    f = text_img(blank(), items)
    return [f] * int(secs * FPS)


def caption_bar(f: np.ndarray, caption: str, badge: str | None) -> np.ndarray:
    items = [(W // 2, H - 42, caption, 30, False, INK, "mm")]
    if badge:
        cv2.rectangle(f, (W - 170, 18), (W - 20, 62), ACCENT, -1)
        items.append((W - 95, 40, badge, 26, True, (255, 255, 255), "mm"))
    return text_img(f, items)


def fit(frame: np.ndarray, box_h: int = H - 100) -> np.ndarray:
    """Place a frame (RGB) centred in the area above the caption bar."""
    h, w = frame.shape[:2]
    s = min(W / w, box_h / h)
    fr = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    out = blank()
    y0, x0 = (box_h - fr.shape[0]) // 2 + 10, (W - fr.shape[1]) // 2
    out[y0:y0 + fr.shape[0], x0:x0 + fr.shape[1]] = fr
    return out


def read_frames(path):
    r = iio.get_reader(str(path))
    k = 0
    while True:
        try:
            yield r.get_data(k)
        except (IndexError, RuntimeError):
            return
        k += 1


def clip(path, caption, speed=2, hold_s=1.5, max_s=None):
    frames = []
    for k, f in enumerate(read_frames(path)):
        if k % speed:
            continue
        frames.append(caption_bar(fit(f), caption, f"{speed}倍速" if speed > 1 else None))
        if max_s and len(frames) >= max_s * FPS:
            break
    return frames + [frames[-1]] * int(hold_s * FPS)


def still(path, caption, secs=6.0):
    im = cv2.imread(str(path))[..., ::-1]
    return [caption_bar(fit(im), caption, None)] * int(secs * FPS)


def g_vs_n(ev: Path, key: str, cat: str):
    """Stack the G and N evaluation videos of one E1 episode (overhead | goal | canvas)."""
    def res(agent):
        for l in open(ev / "E1" / f"results_{agent}.jsonl"):
            r = json.loads(l)
            if r["category"] == cat and r["i"] == 0:
                return r
    rg, rn = res("G"), res("N")
    out, lg, ln = [], None, None
    ga, na = read_frames(ev / "E1" / "G" / f"{key}.mp4"), read_frames(ev / "E1" / "N" / f"{key}.mp4")
    while True:
        a, b = next(ga, None), next(na, None)
        if a is None and b is None:
            break
        lg = a if a is not None else lg
        ln = b if b is not None else ln
        out.append((lg, ln))
    frames = []
    for k, (a, b) in enumerate(out):
        if k % 2:
            continue
        top = text_img(np.vstack([np.full((44, a.shape[1], 3), 255, np.uint8), a]),
                       [(8, 22, f"G：お手本画像あり　 俯瞰カメラ｜お手本｜描いた絵　 IoU {rg['dilated_iou']:.2f}", 22, True, ACCENT, "lm")])
        bot = text_img(np.vstack([np.full((44, b.shape[1], 3), 255, np.uint8), b]),
                       [(8, 22, f"N：文字の指示だけ　 俯瞰カメラ｜お手本｜描いた絵　 IoU {rn['dilated_iou']:.2f}", 22, True, SUB, "lm")])
        frames.append(caption_bar(fit(np.vstack([top, bot])), "同じお手本・同じ紙で比べる。N はお手本を見られないので中央に典型的な星を描く", "2倍速"))
    return frames + [frames[-1]] * 45


def results_card(summary_csv: Path, secs=9.0):
    import pandas as pd
    s = pd.read_csv(summary_csv).set_index(["setting", "agent"])
    g = lambda st, a, m: s.loc[(st, a), m]
    rows = [("お手本との一致（IoU）", f"{g('E1', 'G', 'dilated_iou'):.2f}", f"{g('E1', 'N', 'dilated_iou'):.2f}"),
            ("位置のずれ", f"{g('E1', 'G', 'centroid_offset_mm'):.1f} mm", f"{g('E1', 'N', 'centroid_offset_mm'):.1f} mm"),
            ("生成したお手本で描く（IoU）", f"{g('E2', 'G', 'dilated_iou'):.2f}", "—"),
            ("カテゴリ名なしの指示（IoU）", f"{g('E4', 'G', 'dilated_iou'):.2f}", f"{g('E4', 'N', 'dilated_iou'):.2f}")]
    items = [(W // 2, 90, "評価：学習に使っていない絵 250 枚 × 設定", 40, True, INK, "mm"),
             (760, 170, "G（お手本あり）", 28, True, ACCENT, "mm"), (1040, 170, "N（文字だけ）", 28, True, SUB, "mm")]
    for k, (name, a, b) in enumerate(rows):
        y = 250 + 90 * k
        items += [(140, y, name, 32, False, INK, "lm"), (760, y, a, 44, True, ACCENT, "mm"), (1040, y, b, 44, True, SUB, "mm")]
    items.append((W // 2, 640, "同じエピソードの対応付き比較で G が良いのは 96%（p < 1e-40）", 26, False, SUB, "mm"))
    f = text_img(blank(), items)
    for k in range(len(rows)):
        y = 250 + 90 * k + 45
        cv2.line(f, (120, y), (1160, y), (225, 228, 236), 1)
    return [f] * int(secs * FPS)


def fade(frames, n_in=0, n_out=0):
    """Fade from / to white over the first n_in / last n_out frames."""
    out = list(frames)
    for k in range(min(n_in, len(out))):
        a = (k + 1) / (n_in + 1)
        out[k] = (255 * (1 - a) + out[k] * a).astype(np.uint8)
    for k in range(min(n_out, len(out))):
        a = (k + 1) / (n_out + 1)
        out[-1 - k] = (255 * (1 - a) + out[-1 - k] * a).astype(np.uint8)
    return out


def e1_examples(ev: Path) -> np.ndarray:
    """Rows goal / G / N, one column per category: episode 0 of E1 (not cherry-picked), RGB, with row labels."""
    rows = []
    for d in ["goal", "G", "N"]:
        ims = [cv2.resize(cv2.imread(str(ev / "E1" / d / f"{c}_000.png")), (190, 190), interpolation=cv2.INTER_AREA)
               for c in ["circle", "square", "triangle", "star", "smiley_face"]]
        rows.append(np.hstack([cv2.copyMakeBorder(x, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(200, 200, 200))
                               for x in ims]))
    grid = np.vstack(rows)[..., ::-1]
    grid = np.hstack([np.full((grid.shape[0], 150, 3), 255, np.uint8), grid])
    return text_img(grid, [(140, 99 + 198 * k, t, 26, True, c, "rm")
                           for k, (t, c) in enumerate([("お手本", INK), ("G", ACCENT), ("N", SUB)])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/course_video/oekaki_vla.mp4")
    ap.add_argument("--eval", default="outputs/eval_goal")
    ap.add_argument("--pair-grid", default="results/goal/eval/e5_G_vs_G_pair.png",
                    help="figure for the two-drawing (E5) segment")
    ap.add_argument("--pair-caption", default="左からお手本・G・G_pair。G_pair は 2 か所に描き分けるが、形はまだ崩れる")
    args = ap.parse_args()
    ev, out = resolve(args.eval), resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # each segment is built, faded from / to white and written before the next one (keeps memory small)
    segments = [
        (lambda: card("お絵かきVLA", ["絵を想像して、ペンで描くロボット",
                                             "SO-101 × SmolVLA × 画像生成モデル（MuJoCo シミュレーション）"], 6.0, 72), 45, 10),
        (lambda: card("しくみ", ["① 指示文「draw a star」からカテゴリを取り出す",
                                 "② 小さな拡散モデル（5M）が線画を「想像」する",
                                 "③ 紙の上に置いた絵を「お手本画像」として SmolVLA に見せる",
                                 "④ SmolVLA がばね式ペンの SO-101 を動かして写し描く"], 8.0), 10, 10),
        (lambda: clip(resolve("outputs/demo_goal/smiley/demo.mp4"),
                      "デモ「draw a smiley face」：左で拡散モデルがお手本を描き、右でロボットがそれを写す"), 10, 10),
        (lambda: clip(resolve("outputs/demo_goal/circle/demo.mp4"), "デモ「please draw a circle」"), 10, 10),
        (lambda: card("お手本画像は効くのか？", ["G：お手本画像を見るモデル　／　N：文字の指示だけのモデル",
                                                 "同じデータ・同じ設定で学習し、同じエピソードで比べる"], 5.0), 10, 10),
        (lambda: g_vs_n(ev, "star_000", "star"), 10, 10),
        (lambda: still_from_array(e1_examples(ev), "各カテゴリの 0 番（選別なし）。G はお手本の位置・大きさ・傾きまで写す",
                                  7.0), 10, 10),
        (lambda: results_card(resolve("results/goal/eval/summary.csv")), 10, 10),
        (lambda: card("工夫：なめらかに描くための推論", ["そのままの SmolVLA は腕が震えて線がギザギザになる",
                                                       "→ サンプリングのノイズを 0 にし、長めに実行してから作り直す",
                                                       "→ チャンクのつなぎ目を RTC でならす（品質成功率 2% → 54%）"], 6.0), 10, 10),
        (lambda: clip(resolve("artifacts/videos_smooth/standard_star_01_naive_vs_smooth_vs_final.mp4"),
                      "左：そのまま　中：なめらかな推論　右：最終形（同じ星のエピソード）"), 10, 10),
        (lambda: card("追加実験：2 つの絵", ["学習データは 1 枚に 1 つの絵だけだったので、2 つ並べたお手本は苦手だった",
                                             "→ 2 つの絵のエピソードを 500 本足して学習し直す（IoU 0.38 → 0.52）"], 5.0), 10, 10),
        (lambda: still(resolve(args.pair_grid), args.pair_caption, 8.0), 10, 10),
        (lambda: card("まとめ", ["画像生成モデルの「想像」を SmolVLA が見て描く流れが閉ループで動いた",
                                 "お手本画像で位置・大きさ・形の一致が大きく向上（IoU 0.19 → 0.51）",
                                 "追加学習：学習を 2 倍に延ばすと 0.56、2 つの絵のデータで E5 が 0.38 → 0.52",
                                 "",
                                 "Quick, Draw! (CC BY 4.0) / SmolVLA, LeRobot (Apache-2.0) / MuJoCo Menagerie SO-101"],
                      7.0), 10, 45),
    ]
    wr = iio.get_writer(out, fps=FPS, quality=8, macro_block_size=1)
    n = 0
    for make, n_in, n_out in segments:
        for f in fade(make(), n_in, n_out):
            wr.append_data(f)
            n += 1
    wr.close()
    print(f"{n / FPS:.1f} s -> {out}")


def still_from_array(im_rgb, caption, secs):
    return [caption_bar(fit(im_rgb), caption, None)] * int(secs * FPS)


if __name__ == "__main__":
    main()
