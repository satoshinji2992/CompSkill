# Release scope

## Contents and paper alignment

The release includes the method modules, deployed compressor checkpoint,
TRAIN-only training recipe and split, ToolQA Full selection/review/execution
scripts, public task/document identities, executor-specific support ledgers,
final KeyOpt fits, the submitted fixed-strength calibration, the support-arm
reranker scores needed to refit KeyOpt, and offline tests.

| Setting | Paper configuration |
|---|---|
| `calibration: keyopt` (default) | Final KeyOpt with support-selected strengths: the paper's main results |
| `calibration: fixed_strength` | Submitted configuration: fixed-strength calibration plus memory correction ([archived appendix](docs/archive/README.md)) |
| `review_mode: exact_quote` (default) | Cited review with verbatim quotes |
| `review_mode: source_line` (optional) | Citation-interface review with document/source-line IDs ([archived appendix](docs/archive/README.md)) |

The default review uses the original full-document prompt and exact-substring
quote validator. The optional mode assigns stable document/source-line IDs and
validates citations against the selected document. Both retain the proposed
document on invalid review. Each configuration has its own result provenance and
must be reported under its own name. Reported counts are those of the paper; this
release does not restate them.

## Execution profiles

Selected DeepSeek cases in the paper use a **65,536-token reply limit** and a
**192,000-token cumulative pre-request output gate** (default: 16,384 / 24,000);
both profiles retain 20 steps and 900 seconds. The gate is checked before a
request. Unknown executions remain in every denominator.

## Released support and calibration

`data/toolqa/gpt/` and `data/toolqa/deepseek/` each provide 318 support tasks with
two separately executed alternatives (636 branches), matching settings, and two
calibrations:

- `keyopt.safetensors`: the final KeyOpt fit (`beta`, `delta`, `mu`), with the
  cross-validated `alpha`, `score_cap` and `use_residual` stored as metadata;
- `fitted.safetensors`: the submitted fixed-strength calibration (`bias`, `mean`,
  `delta`).

GPT has 633 known and 3 unknown branches; DeepSeek has 626 known and 10 unknown.
Null outcomes remain unknown and are never used for fitting. Both calibration
files are byte-identical to the frozen research files; `provenance.json` records
their hashes. `support_reranker_scores.json` holds the frozen reranker scores of
the support arms. Together with the ledgers it suffices to refit KeyOpt with
`scripts/fit_keyopt.py`, which reproduces the released cross-validated selection
exactly and the fitted tensors to float32 rounding.

The ledger exports public task context, query vectors, selection/document hashes,
execution identities and binary/null support outcomes. It excludes target-task
outcomes, generated answer text, raw replies and native traces. KeyOpt reads only
these support outcomes; the learned compressor stays frozen.

One historical prompt-metadata detail is preserved for exact reconstruction:
the old support-history formatter printed "GPT-5.5 low" in the event display
label for both executors. The exported `history_executor_label` preserves that
string; `executor_id`, `executor_label`, settings and calibration record the
actual executor. DeepSeek uses its own support outcomes, never GPT outcomes.

## Verification and execution boundary

CPU reconstruction with frozen historical reranker scores produced exact matches
for all 332 GPT and 332 DeepSeek complete review prompts of the final KeyOpt
configuration, and likewise for the submitted fixed-strength configuration. This
checks the released support export, routing, calibration, history formatting and
prompt construction. It does not rerun the reranker or model responses. Offline
tests additionally cover KeyOpt fitting, quote attribution, both review modes,
both calibrations, input integrity, unknown handling, executor isolation, the
checkpoint, and scoring gates.

The portable package uses an OpenAI-compatible endpoint and disposable Docker
workers around the upstream ToolQA agent and scoring function. Historical GPT
execution used RPC and remote workers. End-to-end measurements follow the paper's
original runtime; the portable packaging has been checked offline. Provider-side
model updates and sampling may cause run-to-run differences.

The evaluation setting follows the paper's development protocol. The 4096-to-512
ratio measures stored key coordinates at equal precision; cached lookup timings
are reported separately.

## External assets and exclusions

Obtain upstream ToolQA data/native databases, public target-question text,
encoder/reranker checkpoints and feature caches separately as described in the
[run guide](docs/toolqa.md). External assets retain their upstream licenses.
Other benchmarks' experiment adapters, historical queues, operational archives,
server/account configuration, credentials, internal monitoring/review logs,
raw model traces and private Git history are excluded.
