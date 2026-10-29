# Compressor: checkpoint and training

## Released model

`models/compressor.safetensors` is the exact checkpoint used in the paper.

- SHA256: `66be2c4b8af0d9e92f1915bfda55d7f37cce4c9d28253c7e46c07b875dd07a1e`
- Size: 43,050,836 bytes; safe tensor state dictionary, no pickle payload.
- Architecture: full 4096-dimensional conditioning → width 512; two residual
  LayerNorm/MLP blocks, expansion 4096, SiLU; a residual output around the
  encoder's native 512-dimensional prefix, followed by normalization.
- The encoder is frozen. Queries use its normalized first 512 dimensions;
  only skill keys pass through the learned compressor.
- Primary seed: 17. Seeds 29 and 43 were robustness runs and are not bundled.

## Training data and split

The model was trained on [SKILLRET](https://huggingface.co/datasets/ThakiCloud/SKILLRET)
**TRAIN only**, revision `a050ad233a504a43135bafe8cdf45574052b5729`:
63,259 queries, 10,123 skills, and 127,190 positive associations. TRAIN-file
hashes are checked by the preparation code. No official SKILLRET test qrels or
ToolQA answers are inputs to training.

The released `configs/compressor_split.json` gives the exact identity split:
40,984 fit queries, 5,025 skill-held-out validation queries, and 17,250 mixed
queries. The holdout consists of 2,025 skills, grouped by normalized document
body. Duplicate normalized query texts share their positive-skill union when
assigning folds. Other known positives from duplicate queries are masked out of
the negative denominator. Mixed queries enter only the final all-TRAIN refit.

Features use frozen **Qwen/Qwen3-Embedding-8B, BF16, 4096 dimensions**. Skill text
joins nonempty `name`, `description`, and `skill_md` fields with newlines. This is
not SKILLRET's pipe-separated reference serialization. The first 8192-token
window was used; 168 training skills exceeded it. Query text uses the prefix
stored in `training/compressor.py`. Neither the encoder weights nor the large
feature cache is bundled.

## Objective and schedule

The loss averages log-softmax over each query's positive skills at temperature
0.05, with a 0.1 cosine anchoring term to the native 512-dimensional prefix.
Training uses AdamW, batch 256, learning rate 1e-4, weight decay 0.01, gradient
norm clipping at 1, and 5% warmup followed by cosine decay over 20 epochs.

Screening used seed 17 and checkpoints 1, 2, 5, 10, and 20. The TRAIN-only
skill-held-out choice was epoch **5**, ordered by macro Recall@1, then MRR,
Recall@8, then the earlier epoch. The published checkpoint was refitted from
scratch on all TRAIN queries for 5 epochs, **keeping the first 5 epochs of the
20-epoch schedule**. It did not compress that schedule into 5 epochs.

## Commands

Create the environment once with `conda env create -f environment.yml` from
the repository root, then install the training extras into it:

```bash
conda activate compskill
python -m pip install -e '.[train]'
python -m skill_experience.training.compressor fetch --output work/skillret
# Download Qwen/Qwen3-Embedding-8B separately under its upstream license.
python -m skill_experience.training.compressor encode \
  --data work/skillret --encoder /path/to/local/encoder --output work/skillret-features
python -m skill_experience.training.compressor fit \
  --inputs work/skillret-features --output work/compressor-primary --seed 17
```

The fit command uses the already selected 5-epoch recipe. `--screen` performs
the original 20-epoch candidate-training schedule and saves the five candidates;
it does not silently select a new checkpoint or consult target outcomes.
Each fit has a 3600-second bound and requires CUDA with at least 8 GiB free.
It refuses an existing output directory and does not resume/retry automatically.

To apply the **released** checkpoint to a frozen feature file:

```bash
python -m skill_experience.training.compressor compress \
  --features work/skill-features.safetensors --tensor skills --output work/keys.safetensors
```

`recipe.py` retains the original objective, masks, split function, and schedule.
The portable driver replaces machine paths, cache recovery, and operational
bookkeeping. Fresh encoding/training may differ numerically with hardware and
library versions. The included checkpoint provides the exact deployed weights used in the paper.
