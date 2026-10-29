#!/usr/bin/env bash
set -euo pipefail
# Explicit sequential stages. See docs/toolqa.md for prepared input contracts.
: "${INPUTS:?Set INPUTS to a prepared ToolQA Full input directory}"
: "${RUN:?Set RUN to a new output directory}"
: "${SKILLROUTER_SOURCE:?Set SKILLROUTER_SOURCE}"
: "${RERANKER_MODEL:?Set RERANKER_MODEL to a local frozen checkpoint}"
python -m skill_experience.toolqa.full prepare --inputs "$INPUTS" --output "$RUN"
python -m skill_experience.toolqa.full rerank --run "$RUN" --skillrouter "$SKILLROUTER_SOURCE" --model "$RERANKER_MODEL"
python -m skill_experience.toolqa.full select --run "$RUN"
python -m skill_experience.toolqa.full review --run "$RUN"
# Execution and scoring are intentionally explicit separate commands in the guide.
