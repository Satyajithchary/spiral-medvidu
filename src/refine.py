"""
C4 - Self-Refining Grounding (SRG)
C5 - Ordinal Expectation Decoding (OED)

Both are INFERENCE-ONLY. No training, no extra VRAM, no risk of a failed run.

--------------------------------------------------------------------------
C4a  ITERATIVE TEMPORAL ZOOM
--------------------------------------------------------------------------
The leaderboard says TAG@0.3 = 0.504 but TAG@0.5 = 0.441. The model FINDS the
event and misses the BOUNDARIES. That is not a semantic failure, it is a
temporal-resolution failure: 64 frames over a 180 s clip is 2.8 s per frame, so
a boundary cannot be localised tighter than ~3 s no matter how good the model is.

Pass A : 64 frames over the whole clip     -> coarse hypothesis (s, e)
Pass B : 48 frames over [s-m, e+m], m=0.6*(e-s)
         -> effective frame rate inside the window goes up ~5-8x
         -> re-ask for the exact boundaries, told what its own guess was

The frames in pass B still carry their true clip-relative timestamps, so no
coordinate remapping is needed and the answer is directly comparable.

--------------------------------------------------------------------------
C4b  SPATIAL ZOOM
--------------------------------------------------------------------------
STG sits at ~0.20 mIoU for EVERY model on the board, 4B and 7B alike. When model
capacity doesn't move a metric, the bottleneck isn't the model. A surgical
instrument tip occupies maybe 40x40 px of a 854x480 frame; downscaled to 448 px
and patchified at 28 px, that is under two patches. The information is gone
before the LLM sees it.

Pass A : full anchor frame       -> coarse box b0
Pass B : crop to 2.2 x b0, feed the crop at full processor resolution
         -> re-predict in crop coordinates -> map back to frame coordinates

This is how we get the "1024x1024 Med-Perceiver" benefit without training
anything: spend inference compute instead of VRAM.

--------------------------------------------------------------------------
C5   ORDINAL EXPECTATION DECODING
--------------------------------------------------------------------------
The MedGRPO paper's own failure analysis: on CVS, their models emit (0,0,0),
GPT-4.1 emits (2,2,2), ground truth is (1,0,1). Nobody ever predicts 1. That is
a calibration pathology of greedy decoding on an ordinal rubric - argmax over a
3-way ordinal head collapses to modal classes.

Fix: sample K generations at T>0, parse each into ordinal scores, take the
EXPECTATION per criterion, then round. E[score] = 0.4*0 + 0.6*1 = 0.6 -> 1.
Intermediate classes become reachable. Categorical tasks (NAP) use plurality
vote instead, which is the same marginalisation with a different aggregator.
"""
from __future__ import annotations
import os, math, collections, torch
from PIL import Image
import numpy as np

from .formats import (parse_spans, merge_spans, parse_tboxes, parse_cvs,
                      parse_osats, fmt_spans, fmt_cvs, fmt_osats, fmt_tboxes,
                      OSATS_DIMS, CVS_DIMS, anchor_index_from_rcinfo)
from .sampling import anchor_index, normalize_task
from .timebase import frame_times
from .dataset import remap_path

# ------------------------------------------------------------------ utilities


@torch.no_grad()
def _gen(model, processor, collate, msgs, row, max_new, do_sample=False,
         temperature=0.8, n=1):
    tok = processor.tokenizer
    batch = collate([{"messages": msgs, "answer": "", "row": row}])
    batch.pop("_rows", None)
    batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
    plen = batch["input_ids"].shape[1]
    out = model.generate(**batch, do_sample=do_sample,
                         temperature=temperature if do_sample else None,
                         top_p=0.95 if do_sample else None,
                         num_return_sequences=n,
                         max_new_tokens=max_new,
                         pad_token_id=tok.pad_token_id)
    return [tok.decode(out[i, plen:], skip_special_tokens=True).strip()
            for i in range(out.shape[0])]


# ------------------------------------------------------- C4a temporal zoom

def temporal_zoom(model, processor, collate, ds, row, coarse_pred,
                  k_frames=48, margin_frac=0.6, max_new=256):
    """Refine a coarse temporal prediction by re-sampling inside the hypothesis."""
    spans = parse_spans(coarse_pred)
    if not spans:
        return coarse_pred, {"zoomed": False, "reason": "no parse"}

    times = frame_times(row)
    T = float(times[-1]) if len(times) else 0.0
    s0, e0 = spans[0][0], spans[-1][1]
    dur = max(1.0, e0 - s0)
    lo = max(0.0, s0 - margin_frac * dur)
    hi = min(T, e0 + margin_frac * dur) if T > 0 else e0 + margin_frac * dur

    # frames whose timestamp falls inside the zoom window
    inside = [i for i, t in enumerate(times) if lo <= t <= hi]
    if len(inside) < 6:
        return coarse_pred, {"zoomed": False, "reason": "window too sparse"}

    # if the window already holds fewer frames than our budget, we gain nothing
    if len(inside) <= k_frames and len(inside) >= 0.9 * len(times):
        return coarse_pred, {"zoomed": False, "reason": "no resolution gain"}

    sel = (inside if len(inside) <= k_frames
           else [inside[int(j)] for j in
                 np.round(np.linspace(0, len(inside) - 1, k_frames))])
    sel = sorted(set(int(i) for i in sel))
    zoom_times = [float(times[i]) for i in sel]
    eff_fps = len(sel) / max(1e-6, (zoom_times[-1] - zoom_times[0]))

    prior = (f"On a first coarse pass over the whole clip you localised this "
             f"action to approximately {s0:.1f}-{e0:.1f} seconds. You are now "
             f"given {len(sel)} frames covering only {lo:.1f}-{hi:.1f} seconds "
             f"at roughly {eff_fps:.1f} frames per second - about "
             f"{eff_fps * max(1e-6, (times[-1]-times[0])) / max(1, len(times)):.0f}x "
             f"the temporal detail of the first pass. Determine the exact start "
             f"and end. Your refined answer may fall outside the coarse estimate "
             f"if the frames show that.")

    msgs, _ = ds.build_messages(row, ctcd=prior, force_idx=sel)
    outs = _gen(model, processor, collate, msgs, row, max_new)
    refined = parse_spans(outs[0])
    if not refined:
        return coarse_pred, {"zoomed": False, "reason": "refine unparseable"}

    # guard: reject a refinement that has drifted absurdly far from the prior
    r_s, r_e = refined[0][0], refined[-1][1]
    if r_e <= r_s or r_s > hi + 2 or r_e < lo - 2:
        return coarse_pred, {"zoomed": False, "reason": "refine out of window"}

    txt = fmt_spans([(round(s, 1), round(e, 1))
                     for s, e in merge_spans(refined)[:12]])
    return txt, {"zoomed": True, "coarse": (s0, e0), "refined": (r_s, r_e),
                 "eff_fps": eff_fps}


# -------------------------------------------------------- C4b spatial zoom

def spatial_zoom(model, processor, collate, ds, row, coarse_pred,
                 expand=2.2, max_new=64, max_boxes=12):
    """Refine EVERY box in the predicted sequence, not just the first.

    STG ground truth is a time-indexed sequence, so a single refined box would
    leave the rest coarse. For each (t, box) we crop the frame nearest t and
    re-predict at full processor resolution, then map back."""
    seq = parse_tboxes(coarse_pred)
    if not seq:
        return coarse_pred, {"zoomed": False, "reason": "no parse"}

    times = frame_times(row)
    out, n_ok = [], 0
    for t, b0 in seq[:max_boxes]:
        idx = (int(np.argmin(np.abs(np.asarray(times) - t)))
               if len(times) else (anchor_index(row) or 0))
        nb, info = _zoom_one_box(model, processor, collate, ds, row, b0, idx,
                                 expand, max_new, t)
        out.append((t, nb))
        n_ok += int(info.get("zoomed", False))
    for t, b in seq[max_boxes:]:
        out.append((t, b))
    if n_ok == 0:
        return coarse_pred, {"zoomed": False, "reason": "no box refined"}
    return fmt_tboxes(sorted(out)), {"zoomed": True, "n_boxes": len(seq),
                                     "n_refined": n_ok}


def _zoom_one_box(model, processor, collate, ds, row, b0, frame_idx,
                  expand, max_new, t):
    """NOTE: reads the ORIGINAL frame, never the cache.

    The cache is downscaled to 448px but ground-truth boxes are in original
    resolution (EgoSurgery boxes reach [1256, 392, 1850, 1079] on a 1920x1080
    frame). Cropping 448px imagery with original-res coordinates put every crop
    outside the image, which is why 173/205 STG refinements were rejected.
    Full resolution is the entire point of spatial zoom."""
    vids = row.get("video") or []
    if not vids:
        return b0, {"zoomed": False, "reason": "no frames"}
    frame_idx = max(0, min(frame_idx, len(vids) - 1))

    orig_root = getattr(ds, "orig_root", None)
    fp = None
    for cand in ([remap_path(vids[frame_idx], ds.src_prefix, orig_root)]
                 if orig_root else []) + [
                 vids[frame_idx],
                 remap_path(vids[frame_idx], ds.src_prefix, ds.cache_root)]:
        if cand and os.path.exists(cand) and os.access(cand, os.R_OK):
            fp = cand
            break
    if fp is None:
        return b0, {"zoomed": False, "reason": "frame missing"}

    im = Image.open(fp).convert("RGB")
    W, H = im.size

    # if we only found the downscaled cache, rescale the prior into its space
    scale = 1.0
    if max(W, H) < max(b0[2], b0[3]) * 0.95:
        scale = max(W, H) / max(b0[2], b0[3])
        b0 = [v * scale for v in b0]
    x1, y1, x2, y2 = b0
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = max(24.0, (x2 - x1) * expand), max(24.0, (y2 - y1) * expand)
    cx0, cy0 = max(0, int(cx - w / 2)), max(0, int(cy - h / 2))
    cx1, cy1 = min(W, int(cx + w / 2)), min(H, int(cy + h / 2))
    if cx1 - cx0 < 16 or cy1 - cy0 < 16:
        return b0, {"zoomed": False, "reason": "crop degenerate"}
    if (cx1 - cx0) >= 0.92 * W and (cy1 - cy0) >= 0.92 * H:
        return b0, {"zoomed": False, "reason": "crop is whole frame"}

    crop = im.crop((cx0, cy0, cx1, cy1))
    cw, ch = crop.size
    q = next((m["value"] for m in row["conversations"] if m["from"] == "human"), "")
    q = q.replace("<video>", "").strip()[:900]
    sysmsg = ("You are an expert surgical video analyst. You are given a "
              f"magnified crop of the frame at {t:.1f} seconds. The crop is "
              f"{cw}x{ch} pixels. Answer ONLY with the bounding box of the "
              "requested object IN CROP COORDINATES, format [x1, y1, x2, y2], "
              "top-left origin. No other text.")
    user = (f"On the full {W}x{H} frame you estimated [{x1:.0f}, {y1:.0f}, "
            f"{x2:.0f}, {y2:.0f}]. This crop is that region magnified. Give the "
            f"tight box in crop coordinates.\n\n{q}")
    msgs = [{"role": "system", "content": [{"type": "text", "text": sysmsg}]},
            {"role": "user", "content": [{"type": "image", "image": crop},
                                         {"type": "text", "text": user}]}]
    outs = _gen(model, processor, collate, msgs, row, max_new)
    bc = parse_tboxes(outs[0])
    if not bc:
        return b0, {"zoomed": False, "reason": "refine unparseable"}
    bc = bc[0][1]
    fx1 = cx0 + max(0.0, min(bc[0], cw)); fy1 = cy0 + max(0.0, min(bc[1], ch))
    fx2 = cx0 + max(0.0, min(bc[2], cw)); fy2 = cy0 + max(0.0, min(bc[3], ch))
    if fx2 <= fx1 or fy2 <= fy1:
        return b0, {"zoomed": False, "reason": "mapped box degenerate"}
    ix = max(0.0, min(fx2, x2) - max(fx1, x1))
    iy = max(0.0, min(fy2, y2) - max(fy1, y1))
    if ix * iy <= 0:
        return b0, {"zoomed": False, "reason": "disjoint from prior"}
    if scale != 1.0:                       # map back out of cache space
        fx1, fy1, fx2, fy2 = [v / scale for v in (fx1, fy1, fx2, fy2)]
    return [fx1, fy1, fx2, fy2], {"zoomed": True}


# --------------------------------------------------- C5 ordinal expectation

CVS_NAMES = CVS_DIMS


def ordinal_expectation(model, processor, collate, ds, row, K=5, temperature=0.8,
                        max_new=160, ctcd=None):
    """Marginalise over K sampled generations instead of taking the argmax."""
    task = normalize_task(row["qa_type"])
    msgs, _ = ds.build_messages(row, ctcd=ctcd)
    outs = _gen(model, processor, collate, msgs, row, max_new,
                do_sample=True, temperature=temperature, n=K)

    if task == "cvs_assessment":
        votes = np.array([parse_cvs(o) for o in outs], dtype=float)  # K x 3
        if votes.size == 0:
            return outs[0], {}
        exp = votes.mean(axis=0)
        final = [int(np.clip(round(v), 0, 2)) for v in exp]
        txt = fmt_cvs(final)
        return txt, {"expectation": exp.round(2).tolist(), "K": K,
                     "n_intermediate": int(sum(v == 1 for v in final))}

    if task == "skill_assessment":
        # six OSATS dimensions rated 1-5 -> marginalise each independently
        per = {d: [] for d in OSATS_DIMS}
        for o in outs:
            g = parse_osats(o)
            for d in OSATS_DIMS:
                if d in g:
                    per[d].append(g[d])
        if not any(per.values()):
            return outs[0], {}
        exp = {d: (float(np.mean(v)) if v else 4.0) for d, v in per.items()}
        final = {d: int(np.clip(round(x), 1, 5)) for d, x in exp.items()}
        return fmt_osats(final), {"expectation": {k: round(v, 2)
                                                  for k, v in exp.items()},
                                  "K": K}

    # categorical -> plurality vote over normalised strings
    norm = [o.strip().lower().split("\n")[0][:120] for o in outs if o.strip()]
    if not norm:
        return outs[0], {}
    top, cnt = collections.Counter(norm).most_common(1)[0]
    return top, {"agreement": cnt / len(norm), "K": K}


# --------------------------------------------------------------- dispatcher

REFINE_TASKS = {"tal", "stg", "cvs_assessment", "skill_assessment", "next_action"}


def refine(model, processor, collate, ds, row, coarse_pred, cfg):
    """Route a row to the right refinement. Returns (prediction, info)."""
    t = normalize_task(row["qa_type"])
    if t == "tal" and cfg.get("temporal_zoom", True):
        return temporal_zoom(model, processor, collate, ds, row, coarse_pred,
                             k_frames=cfg.get("zoom_frames", 48),
                             margin_frac=cfg.get("zoom_margin", 0.6))
    if t == "stg" and cfg.get("spatial_zoom", True):
        return spatial_zoom(model, processor, collate, ds, row, coarse_pred,
                            expand=cfg.get("zoom_expand", 2.2))
    if t in ("cvs_assessment", "skill_assessment", "next_action") \
            and cfg.get("oed", True):
        # skill needs room for six "X: n/5" pairs; 16 tokens truncated it always
        mx = {"skill_assessment": 160, "cvs_assessment": 96,
              "next_action": 48}[t]
        return ordinal_expectation(model, processor, collate, ds, row,
                                   K=cfg.get("oed_k", 5),
                                   temperature=cfg.get("oed_temp", 0.8),
                                   max_new=mx)
    return coarse_pred, {"zoomed": False, "reason": "not applicable"}
