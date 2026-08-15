"""
Task routing — the "MoE / per-task head" question, in the only form that is
defensible at 6,270 training samples.

WHY NOT EIGHT HEADS
-------------------
Per-task heads sound right and are wrong here, for three reasons:

1. Data. Split the training set eight ways and skill assessment gets 150 rows,
   region caption 310. A LoRA adapter on 150 samples memorises; it does not
   learn a skill. The smallest viable partition is ~1,000 rows.

2. The benchmark's own numbers argue against it. MedGRPO's ablation Row A vs
   Row C: adding the CAPTIONING rewards improved GROUNDING — STG +4.7%,
   TAG@0.3 +6.9%, TAG@0.5 +9.9%. That is positive cross-task transfer, measured
   on this exact data. Task-separated parameters destroy precisely that. Routing
   is the cure for negative transfer, and the evidence here points the other way.

3. There is no "head" to give. An autoregressive VLM emits text through one
   shared LM head over the vocabulary. "Per-task head" can only mean per-task
   LoRA adapters (feasible) or MoE FFN layers with a learned router (needs far
   more data than you have to train the router itself).

WHAT IS DEFENSIBLE: THREE GROUPS, DETERMINISTIC ROUTING
-------------------------------------------------------
Group by what the OUTPUT DISTRIBUTION actually is, not by task name:

    G  grounding   tal, stg                      -> numbers, precision
    C  captioning  dvc, video_summary, region_c  -> clinical prose
    A  assessment  next_action, cvs, skill       -> short ordinal/categorical

    G ~2,480 rows    C ~2,370 rows    A ~1,420 rows

Each partition is large enough to train. Routing is a dict lookup on qa_type —
no learned router, no data cost, no inference ambiguity. Within a group the
tasks genuinely share an output format, so the transfer that Row A/C measured is
preserved where it is most likely to exist.

Total compute is roughly unchanged: three runs over a third of the data each.

MY ACTUAL RECOMMENDATION
------------------------
Train shared first. It is the primary model. Then, if GPU time remains, train
the three grouped adapters and compare on val. Ship the winner.

Note that a negative result is publishable and, on this data, likely: "we tested
deterministic task routing against shared multi-task training; at 6K samples
shared training dominates by X, contradicting the negative-transfer motivation
for task-specialised parameters in surgical VideoLLMs." Reviewers see MoE
proposed constantly and evaluated against a shared baseline almost never.
"""
from __future__ import annotations
from .sampling import normalize_task

TASK_GROUPS = {
    "tal": "g", "stg": "g",
    "dense_captioning": "c", "video_summary": "c", "region_caption": "c",
    "next_action": "a", "cvs_assessment": "a", "skill_assessment": "a",
}
GROUP_NAMES = {"g": "grounding", "c": "captioning", "a": "assessment"}


def group_of(qa_type: str) -> str:
    return TASK_GROUPS.get(normalize_task(qa_type), "c")


def filter_rows(rows, group: str | None):
    if not group or group == "all":
        return rows
    keep = {x.strip().lower() for x in group.split(",")}
    return [r for r in rows if group_of(r["qa_type"]) in keep]


def summarise(rows):
    import collections
    c = collections.Counter(group_of(r["qa_type"]) for r in rows)
    total = max(1, len(rows))
    print(f"[groups] {len(rows)} rows")
    for g, n in sorted(c.items()):
        print(f"    {g} ({GROUP_NAMES[g]:11s}) {n:6d}  {100*n/total:5.1f}%")
    return c


def parse_adapter_map(spec: str | None) -> dict[str, str]:
    """'g=runs/sft_g/final,c=runs/sft_c/final,a=runs/sft_a/final' -> dict"""
    if not spec:
        return {}
    out = {}
    for part in spec.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip().lower()] = v.strip()
    return out


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", default="data/train.json")
    a = ap.parse_args()
    rows = json.load(open(a.train_json))
    summarise(rows)
    print("\nPer-task counts (why eight heads fail):")
    import collections
    t = collections.Counter(normalize_task(r["qa_type"]) for r in rows)
    for k, v in sorted(t.items(), key=lambda x: -x[1]):
        flag = "   <- too small for its own adapter" if v < 800 else ""
        print(f"    {k:20s} {v:6d}  group={TASK_GROUPS.get(k,'?')}{flag}")
