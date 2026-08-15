"""
Parsers, metrics, and verifiable rewards.

Doubles as the local evaluation backend (evaluate.py) and the GRPO reward
function (train_grpo.py). One implementation, so what you optimise is exactly
what you measure.

Reward design differences vs MedGRPO:

  * They kept a GPT-4.1 LLM judge INSIDE the RL loop. At ~$0.003/call x 8
    rollouts x 5000 steps that is both slow and expensive, and it is the reason
    their RL run is a multi-day affair. We use a deterministic proxy
    (terminology-F1 from the ontology + vagueness penalty + embedding sim) in
    the loop, and reserve the real judge for offline validation only.

  * They excluded NAP from the reward set and NAP regressed 0.442 -> 0.405. We
    include every verifiable task (TAL, STG, NAP, CVS, SA) so nothing silently
    rots. Task coverage in the reward set is not optional in multi-task RL.

  * We keep their logistic median-centred normalisation (it is the paper's
    genuinely good idea) but recompute percentiles from the *current* policy
    every N steps instead of freezing them at the SFT checkpoint.
"""
from __future__ import annotations
import re, math, numpy as np, collections
from .ontology import terminology_f1, vagueness_penalty
from .sampling import normalize_task
from .formats import (parse_spans, merge_spans, parse_tboxes, stg_miou,
                      parse_events, parse_cvs, parse_osats, osats_score,
                      osats_mae, box_iou, fmt_spans, fmt_cvs, fmt_osats,
                      OSATS_DIMS, CVS_DIMS)

# ------------------------------------------------------------------- parsers

# Parsers live in src/formats.py — they were rewritten against the real
# ground-truth dump. Do not reintroduce local copies here.


def parse_box(text):
    """Back-compat single box = the first box of the sequence. STG scoring must
    use stg_miou(), not this."""
    tb = parse_tboxes(text)
    return tb[0][1] if tb else None


def parse_skill(text):
    """Deprecated: skill assessment is six OSATS 1-5 ratings, not a class."""
    return parse_osats(text)


# ------------------------------------------------------------------- metrics

def tiou(a, b) -> float:
    """Temporal IoU over two SETS of intervals (union-based)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    a, b = merge_spans(a), merge_spans(b)
    def total(x): return sum(e - s for s, e in x)
    inter = 0.0
    for s1, e1 in a:
        for s2, e2 in b:
            inter += max(0.0, min(e1, e2) - max(s1, s2))
    union = total(a) + total(b) - inter
    return inter / union if union > 0 else 0.0


def best_tiou(pred, gt) -> float:
    """Max pairwise IoU — matches 'mIoU@thr' style single-segment scoring."""
    if not pred or not gt:
        return 0.0
    best = 0.0
    for s1, e1 in pred:
        for s2, e2 in gt:
            i = max(0.0, min(e1, e2) - max(s1, s2))
            u = (e1 - s1) + (e2 - s2) - i
            if u > 0:
                best = max(best, i / u)
    return best


def box_iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, xb - xa) * max(0.0, yb - ya)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / ua if ua > 0 else 0.0


def dvc_f1(pred_events, gt_events, iou_thr=0.3) -> float:
    """Segment-level F1: a prediction matches if tIoU>=thr AND action agrees."""
    if not pred_events and not gt_events:
        return 1.0
    if not pred_events or not gt_events:
        return 0.0
    used, tp = set(), 0
    for p in pred_events:
        for j, g in enumerate(gt_events):
            if j in used:
                continue
            i = max(0.0, min(p["end"], g["end"]) - max(p["start"], g["start"]))
            u = (p["end"] - p["start"]) + (g["end"] - g["start"]) - i
            if u > 0 and i / u >= iou_thr:
                pa, ga = p["action"], g["action"]
                if pa in ga or ga in pa or terminology_f1(pa, ga) > 0.5:
                    used.add(j); tp += 1; break
    prec = tp / len(pred_events)
    rec = tp / len(gt_events)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def caption_proxy(pred, gt, embedder=None) -> float:
    """Deterministic stand-in for the GPT judge. 0..1."""
    tf = terminology_f1(pred, gt)
    vg = vagueness_penalty(pred)
    sim = 0.0
    if embedder is not None:
        try:
            import numpy as _np
            a, b = embedder.encode([pred or "", gt or ""], normalize_embeddings=True)
            sim = float(_np.dot(a, b))
        except Exception:
            sim = 0.0
    else:
        pa = set(re.findall(r"[a-z]{4,}", (pred or "").lower()))
        pb = set(re.findall(r"[a-z]{4,}", (gt or "").lower()))
        sim = len(pa & pb) / max(1, len(pa | pb))
    # weights mirror the paper's hybrid design (half semantic, half clinical)
    return max(0.0, 0.5 * sim + 0.5 * tf - 0.3 * vg)


# ------------------------------------------------- raw per-sample task metric

def raw_metric(pred: str, gt: str, qa_type: str, embedder=None) -> float:
    t = normalize_task(qa_type)
    if t == "tal":
        return best_tiou(parse_spans(pred), parse_spans(gt))
    if t == "stg":
        return stg_miou(pred, gt)          # sequence of boxes, not one box
    if t == "dense_captioning":
        pe, ge = parse_events(pred), parse_events(gt)
        f1 = dvc_f1(pe, ge)
        cap = caption_proxy(" ".join(e["desc"] for e in pe),
                            " ".join(e["desc"] for e in ge), embedder)
        return 0.6 * f1 + 0.4 * cap
    if t in ("video_summary", "region_caption"):
        return caption_proxy(pred, gt, embedder)
    if t == "next_action":
        p, g = (pred or "").strip().lower(), (gt or "").strip().lower()
        return 1.0 if (p == g or (g and g in p) or terminology_f1(p, g) > 0.7) else 0.0
    if t == "cvs_assessment":
        p, g = parse_cvs(pred), parse_cvs(gt)
        return sum(a == b for a, b in zip(p, g)) / 3.0
    if t == "skill_assessment":
        return osats_score(pred, gt)       # six OSATS dimensions, exact match
    return 0.0


def format_ok(pred: str, qa_type: str) -> bool:
    t = normalize_task(qa_type)
    if t == "tal":
        return bool(parse_spans(pred)) or (pred or "").strip().lower() == "none"
    if t == "stg":
        return len(parse_tboxes(pred)) > 0
    if t == "dense_captioning":
        return bool(parse_events(pred))
    if t == "cvs_assessment":
        return len(re.findall(r"\b[0-2]\b", pred or "")) >= 3
    if t == "tal":
        pass
    if t == "skill_assessment":
        return len(parse_osats(pred)) >= 3
    return bool((pred or "").strip())


# --------------------------------------------- cross-dataset normalisation

class RewardNormalizer:
    """Logistic, median-centred, IQR-scaled — MedGRPO eq.(3), but the
    percentiles are refreshable from the live policy instead of frozen at SFT.

        r = sigmoid( k * (x - p50) / IQR )

    so median performance maps to 0.5 for EVERY (dataset, task) pair, which is
    what stops easy datasets from dominating the gradient.
    """

    def __init__(self, k=3.0, min_iqr=0.02, warmup=32):
        self.k, self.min_iqr, self.warmup = k, min_iqr, warmup
        self.buf = collections.defaultdict(list)
        self.stats = {}

    @staticmethod
    def key(dataset, qa_type):
        return f"{dataset}|{normalize_task(qa_type)}"

    def observe(self, dataset, qa_type, x):
        kk = self.key(dataset, qa_type)
        b = self.buf[kk]
        b.append(float(x))
        if len(b) > 4096:
            del b[:1024]
        if len(b) >= self.warmup and len(b) % 32 == 0:
            arr = np.asarray(b)
            p25, p50, p75 = np.percentile(arr, [25, 50, 75])
            self.stats[kk] = (float(p50), max(self.min_iqr, float(p75 - p25)))

    def __call__(self, dataset, qa_type, x):
        kk = self.key(dataset, qa_type)
        if kk not in self.stats:
            return float(x)                     # identity until warmed up
        p50, iqr = self.stats[kk]
        z = self.k * (float(x) - p50) / iqr
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    def state(self):
        return dict(self.stats)

    def load(self, d):
        self.stats.update(d)


# ------------------------------------------------------------ GRPO reward fn

def boundary_bonus(pred_spans, gt_spans, tau=2.0) -> float:
    """Extra credit for tight boundaries, not just overlap.

    TAG@0.3=0.504 vs TAG@0.5=0.441 on the leaderboard says the events are found
    but the edges are loose. Plain IoU is flat-ish near the optimum; an explicit
    exponential boundary term keeps pushing once IoU is already decent.
    """
    if not pred_spans or not gt_spans:
        return 0.0
    ps, pe = pred_spans[0][0], pred_spans[-1][1]
    gs, ge = gt_spans[0][0], gt_spans[-1][1]
    return 0.5 * (math.exp(-abs(ps - gs) / tau) + math.exp(-abs(pe - ge) / tau))


def grpo_reward(pred, gt, sample, normalizer: RewardNormalizer | None = None,
                embedder=None, format_penalty=0.6, terminology_bonus=0.15):
    """Multiplicative format gate x normalised content + terminology shaping."""
    qa, ds = sample["qa_type"], sample["dataset_name"]
    x = raw_metric(pred, gt, qa, embedder)
    if normalizer is not None:
        normalizer.observe(ds, qa, x)
        r = normalizer(ds, qa, x)
    else:
        r = x
    fmt = 1.0 if format_ok(pred, qa) else (1.0 - format_penalty)
    bonus = terminology_bonus * terminology_f1(pred, gt)
    if normalize_task(qa) == "tal":
        bonus += 0.15 * boundary_bonus(parse_spans(pred), parse_spans(gt))
    # length guard: stops the model padding captions to game similarity
    n = len((pred or "").split())
    length_pen = 0.1 if n > 220 else 0.0
    return float(max(0.0, r * fmt + bonus - length_pen)), float(x)
