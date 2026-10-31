# Third-party: lcccluck/sketch-model

Source: https://huggingface.co/lcccluck/sketch-model (model card license: Apache-2.0,
https://www.apache.org/licenses/LICENSE-2.0).

Kept here for reference: `train_quickdraw_ddpm.py` and `README.md`, unmodified copies.
`oekaki/generator/model.py` adapts the U-Net (`SmallConditionalUNet`, `ResBlock`, `SinusoidalTimeEmbedding`),
the cosine noise schedule and the noise-prediction training objective.

Changes we made:
- a DDIM sampler that also returns the intermediate images;
- EMA weights;
- a single-pass classifier-free guidance batch;
- training data rendered by our common renderer (`oekaki/render.py`).

None of the original checkpoints are used. The generator of this project is trained from scratch on five
Quick, Draw! categories.
