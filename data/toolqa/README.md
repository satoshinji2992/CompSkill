# ToolQA support and calibration

These frozen, executor-specific inputs accompany the paper's ToolQA Full configuration.
Each model has its own 318 support tasks, with two separately executed document arms per task.

| Directory | Model | Tasks | Known branches | Unknown branches |
|---|---|---:|---:|---:|
| `gpt/` | GPT-5.5, low reasoning | 318 | 633 | 3 |
| `deepseek/` | DeepSeek-flash, low thinking | 318 | 626 | 10 |

Each directory contains:

- `support.json`: public task context, frozen 512-d query embeddings, task/source-group IDs,
  document hashes, execution IDs and binary or null outcomes.
- `keyopt.safetensors`: the final KeyOpt fit, `beta` (785), `delta` (785 × 512) and `mu` (512),
  with the cross-validated `alpha`, `score_cap` and `use_residual` as metadata. Default calibration.
- `fitted.safetensors`: the submitted fixed-strength calibration, `bias` (785), `mean` (512) and
  `delta` (785 × 512). Used only with `calibration: fixed_strength`.
- `settings.json`: the matching model and default `exact_quote` review mode.

`support_reranker_scores.json` (shared) holds the frozen reranker scores of the two support arms
of every support task. With a ledger it suffices to refit that executor's KeyOpt:

```bash
python scripts/fit_keyopt.py --executor gpt --output work/keyopt-gpt.safetensors
```

`provenance.json` binds the released files to the original selection, results and fit
manifests by SHA-256. Calibration files are byte-identical copies; the ledger is a
field-restricted export. Generated answers, raw model responses and target-task
outcomes are excluded. Unknown support outcomes remain null.

Use `scripts/prepare_toolqa_inputs.py --executor gpt` or `--executor deepseek` to
select and hash-check the matching bundle. See [the ToolQA guide](../../docs/toolqa.md)
for the complete command and public-data prerequisites, and [release scope](../../RELEASE_SCOPE.md)
for historical prompt metadata and evaluation scope.
