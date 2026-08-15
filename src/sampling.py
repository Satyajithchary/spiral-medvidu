"""
Task-Adaptive Timestamped Sampling (TATS).

The trap everybody falls into: "adaptive sampling picks better frames" is true,
but if you hand a model non-uniformly spaced frames and ask "when does cutting
happen?", it has no way to know frame 12 is at t=31s rather than t=12s. Adaptive
sampling silently destroys temporal grounding unless you *tell* the model the
timestamps. So every sampler here returns (indices, timestamps) and prompts.py
always renders the timestamp grid into the text.

Three regimes:
  grounding (tal/stg)          -> dense + uniform-ish, coverage guaranteed, high N
  captioning (dvc/vs)          -> information-scored, wide temporal span
  region/anchor (rc/stg-anchor)-> slow-fast: dense burst around the anchor frame
                                  plus sparse global context
"""
from __future__ import annotations
import os, numpy as np

# ------------------------------------------------------------------ timestamps

# Timestamps moved to src/timebase.py: this benchmark carries three different
# metadata conventions and ragged frame strides, which broke naive inference on
# 77/300 sampled rows. timebase resolves duration -> dirname fps -> stride.
from .timebase import frame_times, clip_duration, time_source  # noqa: F401


# --------------------------------------------------------------- score loading

def load_scores(sample, cache_root: str | None) -> np.ndarray | None:
    if not cache_root:
        return None
    key = f"{sample['dataset_name']}__{sample['metadata']['video_id']}.npy"
    p = os.path.join(cache_root, "_scores", key)
    if not os.path.exists(p):
        return None
    all_scores = np.load(p)
    # scores were computed over the *video's* sorted unique frames; map by rank
    # of this sample's frames within that ordering. We approximate by
    # interpolating on the frame-index axis, which is monotone.
    frames = np.asarray(sample["sampled_video_frames"], dtype=np.float64)
    if len(all_scores) == 0:
        return None
    xs = np.linspace(frames.min(), frames.max(), len(all_scores))
    return np.interp(frames, xs, all_scores)


# ------------------------------------------------------------------- samplers

def _uniform(n_total: int, k: int) -> np.ndarray:
    if k >= n_total:
        return np.arange(n_total)
    return np.unique(np.round(np.linspace(0, n_total - 1, k)).astype(int))


def _coverage_adaptive(scores: np.ndarray, k: int, n_bins: int | None = None,
                       guarantee: float = 0.5) -> np.ndarray:
    """Pick k frames: `guarantee` fraction uniformly (temporal coverage), the
    rest greedily by information score. This is the safe form of adaptive
    sampling — you never lose a region of the timeline entirely.
    """
    n = len(scores)
    if k >= n:
        return np.arange(n)
    n_uniform = max(2, int(k * guarantee))
    base = set(_uniform(n, n_uniform).tolist())
    n_bins = n_bins or n_uniform
    # remaining budget goes to top-scoring frames, but at most 2 extra per bin
    remaining = k - len(base)
    if remaining > 0:
        order = np.argsort(-scores)
        per_bin = {}
        bin_of = np.minimum((np.arange(n) * n_bins) // n, n_bins - 1)
        for i in order:
            if remaining <= 0:
                break
            if i in base:
                continue
            b = bin_of[i]
            if per_bin.get(b, 0) >= 2:
                continue
            base.add(int(i))
            per_bin[b] = per_bin.get(b, 0) + 1
            remaining -= 1
    return np.array(sorted(base))


def _slowfast(n: int, anchor: int, k_fast: int, k_slow: int,
              fast_radius: int) -> np.ndarray:
    lo, hi = max(0, anchor - fast_radius), min(n - 1, anchor + fast_radius)
    fast = np.unique(np.round(np.linspace(lo, hi, min(k_fast, hi - lo + 1))).astype(int))
    slow = _uniform(n, k_slow)
    return np.unique(np.concatenate([fast, slow, [anchor]]))


# ---------------------------------------------------------------- entry point

TASK_BUDGET = {
    # qa_type prefix -> (n_frames, mode)
    "tal":              (64, "grounding"),
    "stg":              (48, "anchor"),
    "dense_captioning": (48, "info"),
    "video_summary":    (40, "info"),
    "region_caption":   (32, "anchor"),
    "next_action":      (32, "info"),
    "cvs_assessment":   (24, "info"),
    "skill_assessment": (48, "info"),
}


def normalize_task(qa_type: str) -> str:
    q = qa_type.lower()
    for k in TASK_BUDGET:
        if q.startswith(k):
            return k
    # tolerate variants like dense_captioning_gpt / dvc / vs
    alias = {"dvc": "dense_captioning", "vs": "video_summary",
             "rc": "region_caption", "nap": "next_action",
             "cvs": "cvs_assessment", "sa": "skill_assessment",
             "tag": "tal", "temporal": "tal"}
    for a, v in alias.items():
        if q.startswith(a):
            return v
    return "video_summary"


def select_frames(sample, cache_root: str | None = None,
                  budget_scale: float = 1.0, max_frames: int | None = None):
    """Returns (idx: list[int], times: list[float], mode: str)."""
    task = normalize_task(sample["qa_type"])
    k, mode = TASK_BUDGET[task]
    k = int(k * budget_scale)
    if max_frames:
        k = min(k, max_frames)
    n = len(sample["video"])
    times = frame_times(sample)
    scores = load_scores(sample, cache_root)

    if mode == "grounding":
        # Temporal grounding needs a clean, near-uniform grid. We allow a mild
        # adaptive component only (guarantee=0.75) so the mapping from position
        # to time stays close to linear and easy for the model to internalise.
        idx = (_coverage_adaptive(scores, k, guarantee=0.75)
               if scores is not None else _uniform(n, k))
    elif mode == "anchor":
        anchor = anchor_index(sample) or 0
        idx = _slowfast(n, anchor, k_fast=int(k * 0.6), k_slow=int(k * 0.4),
                        fast_radius=max(4, n // 8))
    else:  # "info"
        idx = (_coverage_adaptive(scores, k, guarantee=0.5)
               if scores is not None else _uniform(n, k))

    idx = [int(i) for i in idx][:k]
    return idx, [float(times[i]) for i in idx], mode


def anchor_index(sample):
    """RC_info['start_frame'] is a FILE PATH in this release, not an index."""
    from .formats import anchor_index_from_rcinfo
    return anchor_index_from_rcinfo(sample)
