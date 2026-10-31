"""README figures: a rendered photo of model G drawing in MuJoCo, and the system diagram (English and Japanese)
built from the real images of one imagine-then-draw episode.

  python scripts/make_readme_figures.py sim       # run the episode (GPU), cache images in outputs/readme_figs/
  python scripts/make_readme_figures.py figures   # -> docs/img/robot_drawing.png, docs/img/system_{en,ja}.png
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from oekaki.config import resolve

CACHE = "outputs/readme_figs"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
PHOTO_VIEW = dict(lookat=(0.17, 0.0, 0.10), azimuth=125, elevation=-36, distance=0.46)


def run_sim(args):
    import json

    from oekaki.config import load_config
    from oekaki.data.goal import goal_image, load_goal_config
    from oekaki.env.drawing_env import DrawingEnv, EpisodeParams
    from oekaki.eval.goal_metrics import goal_metrics
    from oekaki.eval.policy import SmolVLAAgent
    from oekaki.generator.imagine import Imaginer, parse_category
    from oekaki.layout import sample_layout
    from oekaki.render import apply_affine

    out = resolve(CACHE); out.mkdir(parents=True, exist_ok=True)
    env_cfg, lcfg, gcfg = load_config(), load_config("configs/layout.yaml"), load_goal_config()
    inf = load_config("configs/eval_goal.yaml")["inference"]
    rng = np.random.default_rng(args.seed)
    im = Imaginer(resolve("outputs/generator/last.pt"))
    gen = im.generate(parse_category(args.prompt, im.classes), seed=args.seed, frames=True)
    T, _ = sample_layout(gen["unit"], rng, lcfg, env_cfg["canvas"]["size"])
    goal = goal_image([apply_affine(s, T) for s in gen["unit"]], gcfg["goal"]["size"])

    agent = SmolVLAAgent(resolve("outputs/train/goal_G/checkpoints/last"), mode=inf["mode"], replan=inf["replan"],
                         delay=inf["delay"], noise_scale=inf["noise_scale"])
    env = DrawingEnv(env_cfg, render=True)
    r = env_cfg["randomization"]
    obs = env.reset(EpisodeParams(height_offset=float(rng.uniform(-r["height_range"], r["height_range"])),
                                  xy_offset=(0.0, 0.0)))
    agent.reset(args.prompt, obs, goal=goal)
    n = int(args.max_seconds * env.fps)
    snaps = []  # simulator state while the pen is on the paper, re-rendered as photos later
    for t in range(n):
        a = agent.act(obs)
        obs = env.step(a)
        if t % 30 == 0 and obs["contact"]:
            snaps.append(dict(t=t, qpos=env.data.qpos.copy(), mocap=env.data.mocap_pos.copy(),
                              paper=env.canvas.img.copy(), top=obs["images"]["top"].copy(),
                              wrist=obs["images"]["wrist"].copy()))
    canvas = env.canvas.img.copy()
    env.close()
    np.savez_compressed(out / "episode.npz", goal=goal, generated=gen["image"], canvas=canvas,
                        traj=np.stack([x for x, _ in gen["trajectory"]]),
                        **{k: np.stack([s[k] for s in snaps]) for k in snaps[0]})
    print(json.dumps(dict(prompt=args.prompt, n_snaps=len(snaps), **goal_metrics(goal, canvas))))


def render_photo(ep, k, view=PHOTO_VIEW, size=(720, 1280)):
    """Re-render snapshot k of the cached episode from a free camera."""
    import mujoco

    from oekaki.config import load_config
    from oekaki.env.drawing_env import DrawingEnv

    env = DrawingEnv(load_config(), render=False)
    m, d = env.model, env.data
    d.qpos[:], d.mocap_pos[:] = ep["qpos"][k], ep["mocap"][k]
    mujoco.mj_forward(m, d)
    n = env.canvas.res * env.canvas.res * 3
    m.tex_data[env.tex_adr:env.tex_adr + n] = ep["paper"][k].reshape(-1)
    r = mujoco.Renderer(m, *size)
    mujoco.mjr_uploadTexture(m, r._mjr_context, env.tex_id)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:], cam.azimuth, cam.elevation, cam.distance = (view["lookat"], view["azimuth"], view["elevation"],
                                                               view["distance"])
    r.update_scene(d, camera=cam)
    return r.render().copy()


# ---- diagram -------------------------------------------------------------------------------------------------

TXT = {
    "en": dict(
        imagine="1  Imagine", draw="2  Draw (closed loop, 30 Hz)",
        instr="instruction", parse="category\nfrom the text",
        diffusion="class-conditional diffusion model\n(DDPM 64×64, DDIM sampling)",
        layout="clean-up +\nrandom layout", goal="goal image\n(whole paper)",
        smolvla="SmolVLA (fine-tuned)", vlm="vision-language model", expert="action expert\n(flow matching)",
        top="top camera", wrist="wrist camera", goalcam="goal image\n(3rd camera)",
        state="joints (6) + pen contact\n+ pen compression", text="instruction text",
        chunk="action chunk\n50 steps of joint targets",
        infer="smooth inference\nσ = 0, execute 30 steps,\nreal-time chunking",
        sim="MuJoCo: SO-101 + spring pen", simsub="paper height random, live ink texture",
        obs="observations", result="drawn result", ex='"draw a smiley face"'),
    "ja": dict(
        imagine="1  想像する", draw="2  描く（閉ループ、30 Hz）",
        instr="指示文", parse="文から\nカテゴリを判定",
        diffusion="クラス条件付き拡散モデル\n（DDPM 64×64、DDIM で生成）",
        layout="後処理 +\nランダム配置", goal="お手本画像\n（紙全体）",
        smolvla="SmolVLA（ファインチューニング）", vlm="視覚言語モデル（VLM）", expert="行動エキスパート\n（flow matching）",
        top="俯瞰カメラ", wrist="手首カメラ", goalcam="お手本画像\n（3 台目のカメラ）",
        state="関節角 6 + ペン接触\n+ ペンの押し込み量", text="指示文",
        chunk="行動チャンク\n関節目標 50 ステップ分",
        infer="なめらかな推論\nσ = 0、30 ステップ実行、\nreal-time chunking",
        sim="MuJoCo：SO-101 + ばね式ペン", simsub="紙の高さはランダム、インクを描画",
        obs="観測", result="描いた結果", ex='"draw a smiley face"'),
}
C = dict(bg=(255, 255, 255), ink=(33, 37, 41), sub=(95, 99, 104), line=(120, 124, 130),
         imag=(232, 240, 254), imag_b=(66, 133, 244), draw=(230, 244, 234), draw_b=(52, 168, 83),
         vla=(254, 247, 224), vla_b=(242, 153, 0), box=(248, 249, 250), box_b=(200, 204, 208))


def font(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, size, index=0)


class Canvas:
    def __init__(self, w, h):
        self.im = Image.new("RGB", (w, h), C["bg"])
        self.d = ImageDraw.Draw(self.im)

    def box(self, xy, fill, outline, r=18, width=3):
        self.d.rounded_rectangle(xy, r, fill=fill, outline=outline, width=width)

    def text(self, xy, s, size=24, bold=False, color=None, anchor="mm"):
        self.d.multiline_text(xy, s, font=font(size, bold), fill=color or C["ink"], anchor=anchor, align="center",
                              spacing=6)

    def img(self, arr, xy, size, border=True):
        a = arr if arr.ndim == 3 else np.repeat(arr[..., None], 3, -1)
        interp = cv2.INTER_NEAREST if a.shape[0] <= 128 else cv2.INTER_AREA
        p = Image.fromarray(cv2.resize(a, (size, size), interpolation=interp))
        self.im.paste(p, xy)
        if border:
            self.d.rectangle([xy[0], xy[1], xy[0] + size, xy[1] + size], outline=C["box_b"], width=2)

    def arrow(self, pts, color=None, width=4, head=14):
        color = color or C["line"]
        self.d.line(pts, fill=color, width=width, joint="curve")
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        ang = np.arctan2(y1 - y0, x1 - x0)
        tip = [(x1, y1), (x1 - head * np.cos(ang - 0.45), y1 - head * np.sin(ang - 0.45)),
               (x1 - head * np.cos(ang + 0.45), y1 - head * np.sin(ang + 0.45))]
        self.d.polygon(tip, fill=color)


def diagram(ep, photo, lang):
    t = TXT[lang]
    W, H = 1800, 1060
    c = Canvas(W, H)
    # ---- 1 imagine: instruction -> category -> diffusion -> layout -> goal image -----------------------
    c.box([20, 20, W - 20, 330], C["imag"], C["imag_b"])
    c.text((44, 44), t["imagine"], 30, True, C["imag_b"], "la")
    y = 195
    c.box([50, y - 60, 300, y + 60], C["box"], C["box_b"])
    c.text((175, y - 25), t["instr"], 22, color=C["sub"])
    c.text((175, y + 15), t["ex"], 24, True)
    c.arrow([(300, y), (350, y)])
    c.box([350, y - 60, 530, y + 60], C["box"], C["box_b"])
    c.text((440, y - 12), t["parse"], 22)
    c.text((440, y + 38), "smiley face", 20, True, C["imag_b"])
    c.arrow([(530, y), (580, y)])
    traj = ep["traj"]
    x = 600
    c.text((x + (5 * 110 - 14) // 2, y - 100), t["diffusion"], 22)
    for k, i in enumerate(np.linspace(0, len(traj) - 1, 5).round().astype(int)):
        c.img(traj[i], (x + k * 110, y - 48), 96)
        if k < 4:
            c.arrow([(x + k * 110 + 97, y), (x + k * 110 + 109, y)], width=3, head=8)
    x += 5 * 110
    c.arrow([(x - 6, y), (x + 50, y)])
    c.box([x + 50, y - 60, x + 250, y + 60], C["box"], C["box_b"])
    c.text((x + 150, y), t["layout"], 22)
    c.arrow([(x + 250, y), (x + 300, y)])
    gx, gs = x + 310, 150
    c.img(ep["goal"], (gx, y - gs // 2), gs)
    c.text((gx + gs + 12, y), t["goal"], 20, anchor="lm")
    # ---- 2 draw: inputs -> SmolVLA -> chunk -> smooth inference -> MuJoCo -> observations ------------
    top2 = 360
    c.box([20, top2, W - 20, H - 20], C["draw"], C["draw_b"])
    c.text((175, top2 + 24), t["draw"], 30, True, C["draw_b"], "la")
    ix, iy, isz, gap = 80, top2 + 110, 110, 128
    k = len(ep["top"]) - 1
    rows = [(ep["goal"], t["goalcam"]), (ep["top"][k], t["top"]), (ep["wrist"][k], t["wrist"])]
    ys = []
    for j, (a, lab) in enumerate(rows):
        yy = iy + j * gap
        c.img(a, (ix, yy), isz)
        c.text((ix + isz + 16, yy + isz // 2), lab, 21, anchor="lm")
        ys.append(yy + isz // 2)
    sy = iy + 3 * gap + 6
    c.box([ix, sy, ix + 330, sy + 70], C["box"], C["box_b"], r=10)
    c.text((ix + 165, sy + 35), t["state"], 19)
    ys.append(sy + 35)
    c.box([ix, sy + 86, ix + 330, sy + 126], C["box"], C["box_b"], r=10)
    c.text((ix + 165, sy + 106), t["text"], 19)
    ys.append(sy + 106)
    # the goal image of part 1 becomes the third camera
    c.arrow([(gx + gs // 2, y + gs // 2), (gx + gs // 2, top2 - 15), (ix + isz // 2, top2 - 15), (ix + isz // 2, iy - 4)],
            color=C["imag_b"], width=4)
    # SmolVLA
    vx0, vx1, vy0, vy1 = 480, 930, iy - 10, sy + 136
    c.box([vx0, vy0, vx1, vy1], C["vla"], C["vla_b"])
    mx = (vx0 + vx1) // 2
    c.text((mx, vy0 + 40), t["smolvla"], 26, True, C["vla_b"])
    c.box([vx0 + 40, vy0 + 100, vx1 - 40, vy0 + 210], C["box"], C["box_b"], r=12)
    c.text((mx, vy0 + 155), t["vlm"] + "\nSmolVLM2", 21)
    c.arrow([(mx, vy0 + 210), (mx, vy0 + 270)])
    c.box([vx0 + 40, vy0 + 270, vx1 - 40, vy0 + 380], C["box"], C["box_b"], r=12)
    c.text((mx, vy0 + 325), t["expert"], 21)
    for yy in ys:
        c.arrow([(ix + 345, yy), (vx0 - 4, yy)], width=3, head=10)
    # action chunk -> smooth inference -> simulator
    cx, cy = vx1 + 45, (vy0 + vy1) // 2
    c.arrow([(vx1, cy), (cx, cy)])
    c.box([cx, cy - 130, cx + 270, cy - 10], C["box"], C["box_b"])
    c.text((cx + 135, cy - 70), t["chunk"], 20)
    c.arrow([(cx + 135, cy - 10), (cx + 135, cy + 20)], width=3, head=8)
    c.box([cx, cy + 20, cx + 270, cy + 150], C["box"], C["box_b"])
    c.text((cx + 135, cy + 85), t["infer"], 19)
    sx = cx + 320
    pw = W - 60 - sx
    ph = int(pw * photo.shape[0] / photo.shape[1])
    py = cy - ph // 2 - 40
    c.arrow([(cx + 270, cy + 85), (sx - 4, cy + 85)])
    c.im.paste(Image.fromarray(cv2.resize(photo, (pw, ph), interpolation=cv2.INTER_AREA)), (sx, py))
    c.d.rectangle([sx, py, sx + pw, py + ph], outline=C["box_b"], width=2)
    c.text((sx + pw // 2, py + ph + 28), t["sim"], 22, True)
    c.text((sx + pw // 2, py + ph + 60), t["simsub"], 19, color=C["sub"])
    # observations: back to the cameras and the state
    lx, by = 46, H - 48
    c.arrow([(sx + pw // 2, py + ph + 84), (sx + pw // 2, by), (lx, by), (lx, ys[1])], color=C["draw_b"], width=4,
            head=1)
    for yy in ys[1:4]:
        c.arrow([(lx, yy), (ix - 4, yy)], color=C["draw_b"], width=4, head=12)
    c.text(((sx + pw // 2 + vx1) // 2, by - 22), t["obs"], 21, True, C["draw_b"])
    return c.im


def make_figures(_):
    ep = dict(np.load(resolve(CACHE) / "episode.npz"))
    out = resolve("docs/img"); out.mkdir(parents=True, exist_ok=True)
    photo = render_photo(ep, len(ep["t"]) - 1)
    Image.fromarray(photo).save(out / "robot_drawing.png", optimize=True)
    for lang in ("en", "ja"):
        diagram(ep, photo, lang).save(out / f"system_{lang}.png", optimize=True)
    print("->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sim", "figures"])
    ap.add_argument("--prompt", default="draw a smiley face")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--max-seconds", type=float, default=40.0)
    args = ap.parse_args()
    (run_sim if args.stage == "sim" else make_figures)(args)


if __name__ == "__main__":
    main()
