# お手本画像まわりの学習を別の GPU マシンで回す手順

お手本画像まわりの学習（判定 CNN、画像生成モデル、SmolVLA の過学習テスト、モデル N・G）は、シミュレーション用マシンとは別の
GPU マシンで行う。データ生成・評価・デモ（MuJoCo）はシミュレーション用マシンで行う。

## 必要なもの
- NVIDIA GPU 1 枚（VRAM 24 GB 以上推奨。SmolVLA は micro-batch 32 で約 10 GB、足りなければ自動で micro-batch を下げ、勾配累積で実効 batch を保つ）
- Python 3.12 を入れられる Linux（`uv` が自動で入れる）、インターネット接続（Quick, Draw! と `lerobot/smolvla_base` の取得）
- ディスク 30 GB 程度

## 手順
```bash
git clone <repo> && cd <repo>
bash scripts/goal_pipeline.sh setup        # uv sync
bash scripts/goal_pipeline.sh quickdraw    # Quick, Draw! の取得と前処理（シミュレーション機と同じ分割になる）
bash scripts/goal_pipeline.sh classifier   # 判定 CNN      -> outputs/classifier/judge.pt
bash scripts/goal_pipeline.sh generator    # 生成モデル     -> outputs/generator/last.pt（samples_*.png で途中確認）
# シミュレーション機で: bash scripts/pack_goal_data.sh -> goal_v1.tar を学習機へ転送し、リポジトリ直下で tar xf
bash scripts/goal_pipeline.sh overfit      # 過学習テスト（30 エピソード, 4k step）
bash scripts/goal_pipeline.sh G            # モデル G（15k step）
bash scripts/goal_pipeline.sh N            # モデル N（15k step。G と同じ設定・シード）
bash scripts/goal_pipeline.sh pack         # outputs_goal.tar（重みとログ）をシミュレーション機へ戻す
```
- classifier と generator は独立なので、GPU が 2 枚あれば並行してよい。G と N も同様。
- 各段階のログは `logs/goal_<stage>.log`。SmolVLA は `outputs/train/goal_*/train_log.csv` に loss を記録し、5k step ごとにチェックポイントを保存する。
- `--dedicated` を付けて学習するので、共有マシン用の待機（GPU / RAM の空き待ち）は無効になる。

## 学習後（シミュレーション機）
```bash
tar xf outputs_goal.tar
.venv/bin/python scripts/sample_generator.py                       # 生成画像の一覧と判定 CNN での正解率
.venv/bin/python scripts/eval_goal_overfit.py                        # 過学習テスト（お手本どおりの位置・大きさか、差し替えたお手本に従うか）
.venv/bin/python scripts/evaluate_goal.py --agent G=outputs/train/goal_G/checkpoints/last \
    --agent N=outputs/train/goal_N/checkpoints/last --agent expert=expert --settings E1 E2 E3 E4 E5
.venv/bin/python scripts/report_goal.py                          # 表・グラフ・画像グリッド
.venv/bin/python scripts/imagine_and_draw.py --prompt "draw a star" --seed 0 --out outputs/demo_star   # デモ動画
```

## 目安時間（A100 / L40S クラス 1 枚）
| 段階 | 目安 |
|---|---|
| quickdraw | 5 分 |
| classifier | 10〜20 分 |
| generator | 2〜4 時間（100k step, batch 256） |
| overfit | 約 1 時間 |
| G, N | 各 5〜7 時間（15k step, 実効 batch 64, 3 カメラ） |

## クラウドの GPU ノード（例：AWS g6e）で回す場合
コードは `git archive` の tar、データは tar のまま S3 などのオブジェクトストレージに置き、タスクごとに
`scripts/cloud_task.sh` を動かす（`docker/Dockerfile.train` のイメージで動作確認済み）。GPU が複数あるノードでは
タスクごとに `CUDA_VISIBLE_DEVICES` で GPU を 1 枚ずつ割り当てる。

```bash
# シミュレーション機: 置く
git archive --format=tar.gz -o code_<TAG>.tar.gz HEAD
tar cf quickdraw_processed.tar data/quickdraw/processed     # 判定 CNN・生成モデル用（ノードから Quick, Draw! を取らずに済む）
aws s3 cp code_<TAG>.tar.gz <code の置き場>; aws s3 cp goal_v1.tar <DATA>/; aws s3 cp quickdraw_processed.tar <DATA>/
# ノード（1 タスク = GPU 1 枚）: コードを展開した根で
DV_DATA=<DATA> DV_OUT=<結果の置き場>/<タスク名> bash scripts/cloud_task.sh classifier generator   # または overfit / G / N
```
- `cloud_task.sh` は必要なデータだけを落とし、`uv sync` の後に GPU が見えるかを確かめる。CUDA 13 の torch が使えない
  （ドライバ 580 未満）ときは同じ版の CUDA 12.8 の wheel に入れ替え、FFmpeg が無く torchcodec が動かないときは pyav で動画を読む
  （`OEKAKI_VIDEO_BACKEND=pyav`）。
- `outputs/` と `logs/` は 15 分ごとと終了時（失敗時も）に `DV_OUT` へ同期される。学習の再開には対応していないので、
  ノードが入れ替わったタスクは最初からやり直しになる。

## 評価 E1〜E5 もレンタル GPU ノードで回す場合
共有マシンの GPU が混んでいると、閉ループ評価（約 2000 エピソード）は 10 時間を超える。イメージ 0.1.1 以降には MuJoCo の
EGL 描画が入っているので、評価もノードで回せる。
```bash
tar cf models_goal.tar outputs/train/goal_G/checkpoints/last outputs/train/goal_N/checkpoints/last \
    outputs/generator/last.pt outputs/generator/config.json outputs/classifier/judge.pt      # S3 の <DATA>/ に置く
# ノード（GPU 全部を 1 タスクで使う）
DV_DATA=<DATA> DV_MODELS=<DATA>/models_goal.tar DV_OUT=<結果の置き場>/goal_eval bash scripts/cloud_eval_goal.sh 6
# シミュレーション機: <結果の置き場>/goal_eval/eval_goal_shards/ を outputs/eval_goal_shards/ に落として
.venv/bin/python scripts/merge_eval_shards.py && .venv/bin/python scripts/report_goal.py
```
- (設定, カテゴリ) ごとに 1 プロセス、GPU 1 枚あたり 6 プロセス。エピソードは (設定, カテゴリ, 番号) ごとに種が決まるので、
  分けて回しても 1 プロセスで回したときと同じエピソードになる。
