"""
Timestamps, done properly.

inspect_data reported 77/300 disagreements between the two naive methods. The
cause is that this benchmark carries THREE different metadata conventions:

    input_video_start_time / input_video_end_time     1395 rows  (NurViD)
    input_video_start_frame / input_video_end_frame   4275 rows
    start_frame / end_frame                            600 rows  (Cholec80_CVS)

plus ragged frame lists (strides of 15, 16, and 24 inside one sample), which
makes median-stride inference unreliable at the ends.

Two facts rescue this:

  1. Where *_time exists, the clip DURATION is known exactly, so
         t_i = (f_i - f_0) / (f_last - f_0) * duration
     needs no frame rate at all and is immune to ragged strides.

  2. Where it does not, the native frame rate is written in the directory name:
         /root/data/NurViD/frames_2fps/...     -> 2
         /root/data/AVOS/frames_15fps/...      -> 15
     Explicit beats inferred.

Resolution order: duration -> dirname fps -> median stride -> index/fps.
`frame_times()` returns seconds and `time_source()` says which rule fired, so
you can audit coverage across the whole split.
"""
from __future__ import annotations
import re, os
import numpy as np

RE_DIR_FPS = re.compile(r"frames?[_-]?(\d+(?:\.\d+)?)\s*fps", re.I)
# some dumps encode it differently; extend here as you find more
RE_DIR_FPS_ALT = re.compile(r"(\d+(?:\.\d+)?)fps", re.I)


def native_fps_from_path(p: str):
    for rx in (RE_DIR_FPS, RE_DIR_FPS_ALT):
        m = rx.search(p or "")
        if m:
            try:
                v = float(m.group(1))
                if 0 < v <= 240:
                    return v
            except ValueError:
                pass
    return None


def _md(sample, *keys):
    md = sample.get("metadata") or {}
    for k in keys:
        if k in md and md[k] not in (None, ""):
            try:
                return float(md[k])
            except (TypeError, ValueError):
                pass
    return None


def clip_duration(sample):
    """Seconds, or None. Uses whichever metadata convention this row carries."""
    s = _md(sample, "input_video_start_time", "start_time")
    e = _md(sample, "input_video_end_time", "end_time")
    if s is not None and e is not None and e > s:
        return e - s

    sf = _md(sample, "input_video_start_frame", "start_frame")
    ef = _md(sample, "input_video_end_frame", "end_frame")
    if sf is not None and ef is not None and ef > sf:
        nfps = native_fps_from_path((sample.get("video") or [""])[0])
        if nfps:
            return (ef - sf) / nfps
    return None


def frame_times(sample) -> np.ndarray:
    """Clip-relative seconds for each entry of sample['video']."""
    frames = np.asarray(sample.get("sampled_video_frames") or [], dtype=np.float64)
    n = len(frames)
    if n == 0:
        n = len(sample.get("video") or [])
        frames = np.arange(n, dtype=np.float64)
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.zeros(1)

    span = frames[-1] - frames[0]

    # 1) exact: known duration, normalise by frame-index range (stride-proof)
    dur = clip_duration(sample)
    if dur is not None and span > 0:
        return (frames - frames[0]) / span * dur

    # 2) explicit native fps from the directory name
    nfps = native_fps_from_path((sample.get("video") or [""])[0])
    if nfps:
        return (frames - frames[0]) / nfps

    # 3) median stride x sampling fps
    d = np.diff(frames)
    d = d[d > 0]
    try:
        samp = float((sample.get("metadata") or {}).get("fps", 1.0)) or 1.0
    except (TypeError, ValueError):
        samp = 1.0
    if len(d):
        native = float(np.median(d)) * samp
        if native > 0:
            return (frames - frames[0]) / native

    # 4) last resort
    return np.arange(n, dtype=np.float64) / samp


def time_source(sample) -> str:
    if clip_duration(sample) is not None and \
            len(sample.get("sampled_video_frames") or []) > 1:
        return "duration"
    if native_fps_from_path((sample.get("video") or [""])[0]):
        return "dirname_fps"
    if len(sample.get("sampled_video_frames") or []) > 1:
        return "median_stride"
    return "index_fps"


# ------------------------------------------------------------------ calibrate

def calibrate(json_path, n=1500):
    """Are ground-truth times CLIP-RELATIVE or ABSOLUTE video time?

    Everything downstream depends on this and it cannot be settled from one
    example. Discriminator: under the absolute hypothesis no GT span may begin
    before input_video_start_time. Under the clip-relative hypothesis no span
    may exceed the clip duration. Count violations of each across the split.
    """
    import json, collections, random
    from .sampling import normalize_task
    with open(json_path) as f:
        data = json.load(f)
    rng = random.Random(0)
    rows = [r for r in data if normalize_task(r.get("qa_type", "")) in
            ("tal", "dense_captioning", "stg")]
    rows = rng.sample(rows, min(n, len(rows)))

    rel_bad = abs_bad = both = neither = 0
    src = collections.Counter()
    durs, gt_max = [], []
    RE = re.compile(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)")
    RE_T = re.compile(r"(\d+(?:\.\d+)?)\s*seconds?\s*:")

    for r in rows:
        src[time_source(r)] += 1
        gt = next((m["value"] for m in r["conversations"] if m["from"] == "gpt"), "")
        spans = [(float(a), float(b)) for a, b in RE.findall(gt)]
        if not spans:
            spans = [(float(t), float(t)) for t in RE_T.findall(gt)]
        if not spans:
            continue
        dur = clip_duration(r)
        if dur is None:
            t = frame_times(r)
            dur = float(t[-1]) if len(t) else None
        if not dur or dur <= 0:
            continue
        start_abs = _md(r, "input_video_start_time", "start_time")
        lo = min(s for s, _ in spans)
        hi = max(e for _, e in spans)
        durs.append(dur); gt_max.append(hi)

        v_rel = hi > dur * 1.10 + 1.0            # exceeds the clip
        v_abs = (start_abs is not None and lo < start_abs - 1.0)  # begins too early
        if v_rel and v_abs:
            both += 1
        elif v_rel:
            rel_bad += 1
        elif v_abs:
            abs_bad += 1
        else:
            neither += 1

    tot = max(1, rel_bad + abs_bad + both + neither)
    print("\n=== TIME BASE CALIBRATION ===")
    print(f"samples with parseable spans: {tot}")
    print(f"  violates CLIP-RELATIVE only (span past clip end) : {rel_bad:5d} "
          f"({100*rel_bad/tot:.1f}%)")
    print(f"  violates ABSOLUTE only (span before clip start)  : {abs_bad:5d} "
          f"({100*abs_bad/tot:.1f}%)")
    print(f"  violates both                                    : {both:5d}")
    print(f"  consistent with both                             : {neither:5d}")
    if durs:
        r_ = np.array(gt_max) / np.maximum(1e-6, np.array(durs))
        print(f"  max(GT) / clip_duration: p50={np.median(r_):.3f} "
              f"p90={np.percentile(r_,90):.3f} p99={np.percentile(r_,99):.3f}")
        print("     (near 1.0 => clip-relative; >>1 => absolute)")
    print("\ntime_source coverage:", dict(src))
    verdict = "CLIP_RELATIVE" if rel_bad <= abs_bad else "ABSOLUTE"
    print(f"\nVERDICT: ground-truth times look {verdict}")
    if rel_bad > 0.02 * tot and abs_bad > 0.02 * tot:
        print("  !! both hypotheses violated often — inspect a few rows by hand")
    return verdict


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainval", default=None)
    ap.add_argument("--n", type=int, default=1500)
    a = ap.parse_args()
    from .paths import need
    calibrate(a.trainval or need("trainval_json"), a.n)
