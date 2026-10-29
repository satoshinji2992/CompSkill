#!/usr/bin/env bash
set -euo pipefail
: "${DATA:?Set DATA to the pinned SKILLRET snapshot}"
: "${ENCODER:?Set ENCODER to a local Qwen3-Embedding-8B checkpoint}"
: "${CACHE:?Set CACHE to a new feature directory}"
: "${OUTPUT:?Set OUTPUT to a new training directory}"
python -m skill_experience.training.compressor encode --data "$DATA" --encoder "$ENCODER" --output "$CACHE"
python -m skill_experience.training.compressor fit --inputs "$CACHE" --output "$OUTPUT" --seed 17
