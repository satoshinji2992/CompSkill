# Upstream components

This repository contains original method code and a trained compressor. It
does not vendor the projects, datasets, or large models listed below.

| Component | Source | Use and license boundary |
|---|---|---|
| SKILLRET | [Dataset and card](https://huggingface.co/datasets/ThakiCloud/SKILLRET) | TRAIN-only compressor supervision; dataset card states Apache-2.0 for benchmark metadata/queries and MIT or Apache-2.0 for source skills. Obtain data upstream. Split files contain identities only. |
| Qwen3-Embedding-8B | [Model card](https://huggingface.co/Qwen/Qwen3-Embedding-8B) | Frozen encoder; Apache-2.0 model license. Weights are obtained separately. |
| SkillRouter | [Code](https://github.com/zhengyanzhao1997/SkillRouter) | Calls the official reranker loader and scoring routine; MIT code. |
| SkillRouter reranker | [Model](https://huggingface.co/pipizhao/SkillRouter-Reranker-0.6B) | Frozen 0.6B reranker, revision `78986e1142d12857cfd85b8005e62902cd42d858`; Apache-2.0 model license; obtained separately. |
| SR-Agents | [Code](https://github.com/oneal2000/SR-Agents) | Official released ToolQA ReAct, tools, corpus interface, and evaluator; MIT code. Imported, not reimplemented or relicensed here. |
| ToolQA | [Original benchmark](https://github.com/night-chen/ToolQA) | Repository code and benchmark questions are released under Apache-2.0. Referenced external corpora retain their source-specific terms; the ledger adds our frozen execution outcomes and embeddings. |

Preserve upstream notices when installing these components. The repository's
MIT license does not replace upstream dataset, model, or third-party dependency
licenses. Source-file hashes for the ToolQA integration are provided in configs;
a changed upstream implementation must be reviewed as a different runtime.
