"""Pure training functions extracted unchanged from the checkpoint recipe."""
import hashlib, math, unicodedata
from collections import defaultdict

REVISION = "a050ad233a504a43135bafe8cdf45574052b5729"

SOURCE_HASHES = {
    "skills": "b4978863f6c36c1d42af4a779f86b8e5809b4709c353aa698ec494e3ed663a2b",
    "queries": "e1189e402c74d11ba57af3bbc60308cc4a9b51f29f2de265e541d0599067624c",
    "qrels": "7480dbf936eeee9905c752a681b029bae8e13f67dcacdf4b716684ca21da83e1",
}

def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())

def field_text(skill):
    # Deliberately matches the existing SRA embedding cache, not official pipe rendering.
    return "\n".join(
        skill.get(k) or "" for k in ("name", "description", "skill_md") if skill.get(k)
    )

def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()

def document_groups(skills):
    """Identical normalized bodies share a fold, regardless of ID/name."""
    return {s["id"]: text_hash(normalized(s.get("skill_md") or s["description"])) for s in skills}

def make_split(skills, queries, excluded):
    groups = document_groups(skills)
    ordered = sorted(set(groups.values()), key=lambda g: text_hash("skillret-holdout-17|" + g))
    held_groups = set(ordered[: max(1, round(0.2 * len(ordered)))])
    held = {s for s, g in groups.items() if g in held_groups}
    duplicate_positive = defaultdict(set)
    for q in queries:
        if q["id"] not in excluded:
            duplicate_positive[normalized(q["query"])].update(q["skill_ids"])
    result = {"fit": [], "validation": [], "mixed": [], "excluded": sorted(excluded)}
    for q in queries:
        if q["id"] in excluded:
            continue
        p = duplicate_positive[normalized(q["query"])]
        role = "fit" if not p & held else "validation" if p <= held else "mixed"
        result[role].append(q["id"])
    if not result["fit"] or not result["validation"]:
        raise ValueError("Empty fit or skill-held-out validation")
    return {
        **result,
        "held_skills": sorted(held),
        "duplicate_document_groups": len(skills) - len(ordered),
    }

def batch_masks(rows, skill_ids):
    import torch

    positions = {s: i for i, s in enumerate(skill_ids)}
    pos = torch.zeros(len(rows), len(skill_ids), dtype=torch.bool)
    valid = torch.ones_like(pos)
    for i, row in enumerate(rows):
        if not set(row["skill_ids"]) <= positions.keys():
            raise ValueError("Training positive outside allowed skill fold")
        for s in row.get("duplicate_skills", row["skill_ids"]):
            if s in positions:
                valid[i, positions[s]] = False
        pos[i, [positions[s] for s in row["skill_ids"]]] = True
        valid[i] |= pos[i]
    return pos, valid

def objective(keys, native, queries, positive, valid):
    import torch.nn.functional as F

    logits = (queries @ keys.float().T / 0.05).masked_fill(~valid, -1e9)
    retrieval = (-(F.log_softmax(logits, -1) * positive).sum(-1) / positive.sum(-1)).mean()
    anchor = (1 - (keys.float() * native).sum(-1)).mean()
    return retrieval + 0.1 * anchor, retrieval, anchor

def lr_factor(step, total):
    warmup = max(1, round(total * 0.05))
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup - 1)))
