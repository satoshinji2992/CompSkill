# ToolQA Full

## Scope and prerequisites

The default is the paper’s Full profile: trained compressor → frozen semantic
Top-10 → conditional support-memory expansion (protect 6, add at most 4) →
frozen SkillRouter reranking plus the executor's fitted KeyOpt correction →
Top-8 exact-quote review with at most 6 history events → one complete skill
document → official ToolQA ReAct → official scoring after complete generation
and cleanup. The submitted fixed-strength calibration with memory correction
remains available as an option.

The bank contains 785 public skill documents; the task manifests identify 318
support tasks and 332 target tasks. Support feedback must be collected with the
same executor, environment, document versions, and budget as the target run.
Unknown alternatives are omitted from learning, not converted to failures.
Target task/source-group history is excluded during conditional reading.

Create the environment once from the repository root, then install the ToolQA extras:

```bash
conda env create -f environment.yml
conda activate compskill
python -m pip install -e '.[toolqa]'
```

If the environment already exists, skip its creation and activate it directly.
Obtain the upstream
[SR-Agents source and native assets](https://github.com/oneal2000/SR-Agents),
[SkillRouter source](https://github.com/zhengyanzhao1997/SkillRouter), and frozen
encoder/reranker checkpoints separately. The ToolQA corpus, native databases,
public target-question text and large feature caches are obtained separately.
**The support ledgers, fitted KeyOpt parameters and submitted calibration
tensors are included for both executors** under `data/toolqa/`. [Paper alignment and verification](../RELEASE_SCOPE.md)
are documented in the release scope.

Compatible source checkouts used for the published integration hashes:

```bash
git clone https://github.com/oneal2000/SR-Agents third_party/SR-Agents
git -C third_party/SR-Agents checkout 277fd8d2bbd7d3b81a5cf4ffa6e87e18c7906e4f
git clone https://github.com/zhengyanzhao1997/SkillRouter third_party/SkillRouter
git -C third_party/SkillRouter checkout 2f0c69fe6786bfee6312a1ab5d5f69abdc6bd245
```

Follow upstream installation and native-data setup instructions. The Docker
image installs only the dependencies required by its ToolQA worker; it does not
include downloaded databases or pretrained retriever caches.

## Input directory

The following files are copied and sealed by `full prepare`:

- `tasks.json`: ordered records with exactly `task_id`, `source_group`, and
  `question`. No answer, evaluation, or reward fields are accepted.
- `skills.json`: ordered records with exactly `skill_id`, `name` (display name),
  `description`, `body`, `document`, and `document_sha256`. `body` is the full
  document used for reranking. Bank order and full-document hashes must match
  `configs/toolqa_bank.json`.
- `features.safetensors`: detached float tensors `keys` (785 × 512) and `queries`
  (332 × 512), in manifest order. Both use the same embedding coordinate system.
  Keys come from the bundled compressor applied to frozen full 4096-dimensional
  skill representations. Queries use normalized first-512 encoder dimensions.
  Historical ToolQA query encoding used NF4 Qwen3-Embedding-8B with BF16 compute;
  document conditioning came from the BF16 cache. These precision choices must
  not be silently interchanged when claiming historical reproduction.
- `keyopt.safetensors`: the executor's final KeyOpt fit, `beta` (785), `delta`
  (785 × 512) and `mu` (512), with the cross-validated `alpha`, `score_cap` and
  `use_residual` as metadata. These are support-derived, not the general
  compressor weights. Use the included executor-specific file;
  `retrieval/keyopt.py` contains the fitting code and `scripts/fit_keyopt.py`
  refits it from the released support data.
- `fitted.safetensors` (only for `calibration: fixed_strength`): the submitted
  configuration's `bias` (785), `mean` (512), and `delta` (785 × 512).
  `retrieval/reranker_calibration.py` contains its support fitting functions.
- `support.json`: 318 records in the schema below, including unknowns.
- `settings.json`: provider/model/runtime identities, `review_mode` and the
  optional `calibration` (`keyopt` when absent, or `fixed_strength`).
  Use the matching included settings: GPT review uses 4000 output tokens;
  DeepSeek uses 8000. Both request low reasoning and default to `exact_quote`.

Support schema (illustrative identities; not experimental data):

```json
{
  "task_id": "support-task", "source_group": "support-group",
  "question": "Public question", "query": [0.6, 0.8],
  "split": "train", "executor_id": "same-as-settings",
  "executor_label": "GPT-5.5 (low reasoning)",
  "environment_id": "toolqa-official-react-785",
  "selection_sha256": "64-character hash of the frozen pre-feedback selection",
  "arms": [
    {"skill_id": "a", "document_sha256": "64-character document hash",
     "physical_id": "original-execution-a", "outcome": 1},
    {"skill_id": "b", "document_sha256": "64-character document hash",
     "physical_id": "original-execution-b", "outcome": null}
  ]
}
```

Real queries have 512 dimensions and are normalized before memory retrieval.
The published records also preserve `context_sha256` for the original public
instruction and `history_executor_label` for exact historical prompt formatting.
The latter is display metadata; executor isolation uses `executor_id`. The two arms are separately
executed alternatives; their outcomes do not establish that a joint bundle
would succeed. The support ledger is executor-specific and cannot be borrowed
between models.

`scripts/prepare_toolqa_inputs.py` assembles these inputs from the original
`corpus.json.zip`, an already answer-free public task file, frozen features and
the released model-specific bundle. It validates document identities and checks
the included bundle against its published hashes:

```bash
python scripts/prepare_toolqa_inputs.py \
  --corpus /path/to/corpus.json.zip \
  --public-tasks /path/to/answer-free-public-tasks.json \
  --features /path/to/frozen-features.safetensors \
  --executor gpt --output work/inputs
```

Use `--executor deepseek` with a separate output directory for DeepSeek.
Custom frozen inputs can instead be supplied as `--support` and `--settings`
together with `--keyopt` and/or `--fitted`. The assembler does not collect
feedback or fit parameters.

## Routing, review, and prediction freezing

```bash
python -m skill_experience.toolqa.full prepare --inputs work/inputs --output work/full
python -m skill_experience.toolqa.full rerank --run work/full \
  --skillrouter third_party/SkillRouter --model /path/to/SkillRouter-Reranker-0.6B
python -m skill_experience.toolqa.full select --run work/full
# Set COMPSKILL_API_KEY and COMPSKILL_BASE_URL in your environment, never a checked-in file.
python -m skill_experience.toolqa.full review --run work/full
```

The reranker call uses the upstream `flat-full` prompt, length 4096, batch 2,
and sigmoid logits. Default `exact_quote` review uses the original prompt and
requires a nonempty verbatim quote from the selected document, with evidence IDs
restricted to the supplied history. Invalid JSON/citations retain the proposed
skill. To use the appendix’s Citation-interface Full, change `review_mode` to
`source_line` in the assembled input `settings.json` **before** `full prepare`,
and use a fresh run directory. This optional mode validates document/source-line
IDs. To run the submitted fixed-strength configuration instead of KeyOpt, set
`calibration` to `fixed_strength` in the same file before `full prepare`. Each
run seals its mode and calibration together with its inputs.
The settings below describe the default execution profile. For the paper’s
DeepSeek output budgets, see
[Execution profiles](../RELEASE_SCOPE.md#execution-profiles).
Provider errors stop the run; no retry or fallback model is hidden in the
public adapter. `frozen_guidance.json` contains complete selected documents.

## Official execution in isolated containers

The controller imports the released `sragents.infer.engines.react.ReActAgent`.
Only the text transport and tool process boundary are adapted. Agent parsing,
prompt construction, observation truncation, and finish handling stay upstream.
The native tool worker runs separately without provider credentials, network
access, or a writable root filesystem. Its source digest is checked on startup.

Build the image from the **same compatible upstream checkout** used by the
controller. The build itself downloads dependencies; generation does not.

```bash
docker buildx build --build-context sra=./third_party/SR-Agents \
  -f native.Dockerfile -t compskill-toolqa-native --load .
python -m skill_experience.toolqa.execute generate --run work/full \
  --sra third_party/SR-Agents --corpus /path/to/native-toolqa-corpus \
  --hf-cache /path/to/prepopulated/huggingface-cache \
  --native-image compskill-toolqa-native
```

The native text retriever's upstream models must already be present in the
readable cache. Provision a Linux Docker host with enough memory for the native
corpus (the container limit is 84 GiB). The loop is sequential, at most 20 model
calls/steps and 900 seconds per episode, with a 16,384 output-token request cap
and a 24,000 cumulative-output **pre-call** gate. A call started below that gate
may finish above it, as in the original protocol. Unknown usage stops execution.

All outputs are new, write-once artifacts. An exception or cleanup failure
preserves the receipt and stops; this focused release has no automatic recovery
queue. An existing directory cannot be used to silently replay failed tasks.


## Separate scoring

```bash
python -m skill_experience.toolqa.execute score --run work/full \
  --sra third_party/SR-Agents --labels /path/to/official-scoring.json \
  --labels-sha256 YOUR_PRECOMMITTED_SCORING_FILE_SHA256
```

The scorer verifies all task receipts, generation manifests, and native cleanup
before opening the answer file. It calls the upstream
`sragents.evaluate.datasets.toolqa.evaluate`; it does not reimplement matching.
The official scoring input is a JSON array keyed by `instance_id`, with the
upstream `eval_data` field. Keep it out of all preparation and generation inputs.

## Verification status

Offline tests cover the actual checkpoint, method modules, citation attribution,
unknown handling, same-executor support, source-group exclusions, manifest drift,
write-once output, complete-generation scoring gates, KeyOpt fitting and both
calibrations. Using frozen reranker scores, the exact-quote path reconstructs the
paper's final-KeyOpt review prompts (2 × 332) and the submitted fixed-strength
prompts (2 × 332) byte for byte. See [Release scope](../RELEASE_SCOPE.md) for
execution verification and provenance.
