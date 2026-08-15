from __future__ import annotations
import hashlib, re
import numpy as np


def qhash(row_or_rec) -> str:
    """Short hash of the QUESTION text.

    (id, qa_type) is still not unique: one AVOS clip carries a TAL row asking
    "when does cutting happen?" and another asking "when does tying happen?" —
    same id, same qa_type, different answers. 6,245 test rows collapse to 5,644
    (id, qa_type) pairs, so ~601 rows were served another question's answer.
    The question text is the only remaining discriminator."""
    if row_or_rec.get("qhash"):
        return row_or_rec["qhash"]
    q = ""
    for m in (row_or_rec.get("conversations") or []):
        if m.get("from") == "human":
            q = m.get("value", "")
            break
    q = " ".join(q.replace("<video>", "").split()).lower()
    return hashlib.sha1(q.encode()).hexdigest()[:10]


def rowkey(row_or_rec):
    """(id, qa_type, question-hash) — the only combination that is unique."""
    return (row_or_rec["id"], row_or_rec["qa_type"], qhash(row_or_rec))

# ------------------------------------------------------------------- regexes
RE_SPAN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|–|—|to|until)\s*(\d+(?:\.\d+)?)")
RE_TBOX = re.compile(r"(\d+(?:\.\d+)?)\s*seconds?\s*:\s*"
                     r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
                     r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", re.I)
RE_BOX_ANY = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
                        r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")
RE_EVENT_LBL = re.compile(
    r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*seconds?\s*:\s*"
    r"([^:\n]{1,60}?)\s*:\s*([^\n]+)", re.I)
RE_EVENT_NOLBL = re.compile(
    r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*seconds?\s*:\s*([^\n]+)", re.I)
RE_RATING = re.compile(r"([A-Za-z][A-Za-z /\-]{2,40}?)\s*:\s*([1-5])\s*/\s*5")
RE_CVS = re.compile(r"([A-Za-z][A-Za-z ]{2,40}?)\s*:\s*([0-2])\b")

OSATS_DIMS = ["Respect for tissue", "Suture/needle handling", "Time and motion",
              "Flow of operation", "Overall performance",
              "Quality of final product"]
CVS_DIMS = ["Two structures", "Cystic plate", "Hepatocystic triangle"]


# ---------------------------------------------------------------------- TAL
def parse_spans(text: str):
    """-> [(start, end)]. Zero-duration spans preserved; they are real events."""
    if not text:
        return []
    if text.strip().lower().rstrip(".") in ("none", "n/a", "no", "never"):
        return []
    out = []
    for m in RE_SPAN.finditer(text):
        s, e = float(m.group(1)), float(m.group(2))
        if e >= s and (e - s) < 7200:          # >= not >, see module docstring
            out.append((s, e))
    return out


def merge_spans(spans, gap=0.0):
    if not spans:
        return []
    spans = sorted(spans)
    out = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def fmt_spans(spans) -> str:
    if not spans:
        return "none"
    return ", ".join(f"{s:.1f}-{e:.1f}" for s, e in spans) + " seconds."


# ---------------------------------------------------------------------- STG
def parse_tboxes(text: str):
    """-> [(t, [x1,y1,x2,y2])], the time-indexed box sequence."""
    out = [(float(m.group(1)),
            [float(m.group(i)) for i in range(2, 6)])
           for m in RE_TBOX.finditer(text or "")]
    if out:
        return sorted(out)
    # degraded: bare boxes with no timestamps -> synthesise an index
    return [(float(i), [float(m.group(j)) for j in range(1, 5)])
            for i, m in enumerate(RE_BOX_ANY.finditer(text or ""))]


def fmt_tboxes(tb) -> str:
    return " ".join(f"{t:.1f} seconds: [{b[0]:.2f}, {b[1]:.2f}, "
                    f"{b[2]:.2f}, {b[3]:.2f}]" for t, b in tb)


def box_iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    xa, ya, xb, yb = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, xb - xa) * max(0.0, yb - ya)
    u = ((a[2]-a[0]) * (a[3]-a[1])) + ((b[2]-b[0]) * (b[3]-b[1])) - i
    return i / u if u > 0 else 0.0


def stg_miou(pred_text: str, gt_text: str, t_tol=2.0) -> float:
    """Mean IoU over the GT timestamps, each matched to the nearest predicted
    timestamp within t_tol seconds. Unmatched GT timestamps score 0, which is
    what makes a single-box answer correctly cheap instead of accidentally fine."""
    P, G = parse_tboxes(pred_text), parse_tboxes(gt_text)
    if not G:
        return 0.0
    if not P:
        return 0.0
    pt = np.array([t for t, _ in P])
    ious = []
    for t, gb in G:
        j = int(np.argmin(np.abs(pt - t)))
        ious.append(box_iou(P[j][1], gb) if abs(pt[j] - t) <= t_tol else 0.0)
    return float(np.mean(ious))


# ------------------------------------------------------------ dense captions
def parse_events(text: str):
    """-> [{start,end,action,desc}]. Handles labelled and unlabelled forms."""
    if not text:
        return []
    ev = [{"start": float(m.group(1)), "end": float(m.group(2)),
           "action": m.group(3).strip().lower(), "desc": m.group(4).strip()}
          for m in RE_EVENT_LBL.finditer(text)]
    if ev:
        return ev
    return [{"start": float(m.group(1)), "end": float(m.group(2)),
             "action": "", "desc": m.group(3).strip()}
            for m in RE_EVENT_NOLBL.finditer(text)]


def fmt_events(ev) -> str:
    out = []
    for e in ev:
        head = f"{e['start']:.1f}-{e['end']:.1f} seconds"
        out.append(f"{head}: {e['action']}: {e['desc']}" if e.get("action")
                   else f"{head}: {e['desc']}")
    return "\n".join(out)


# --------------------------------------------------------------------- OSATS
def parse_osats(text: str):
    """-> dict{dimension: 1..5}. Falls back to positional if labels differ."""
    got = {}
    for m in RE_RATING.finditer(text or ""):
        key = m.group(1).strip().lower()
        for d in OSATS_DIMS:
            if key.startswith(d.split()[0].lower()) or d.lower() in key:
                got[d] = int(m.group(2))
                break
        else:
            got[m.group(1).strip()] = int(m.group(2))
    if not got:
        nums = [int(x) for x in re.findall(r"\b([1-5])\s*/\s*5", text or "")]
        if not nums:
            nums = [int(x) for x in re.findall(r"\b([1-5])\b", text or "")][:6]
        got = {d: v for d, v in zip(OSATS_DIMS, nums)}
    return got


def fmt_osats(d) -> str:
    return ", ".join(f"{k}: {int(d.get(k, 3))}/5" for k in OSATS_DIMS)


def osats_score(pred: str, gt: str) -> float:
    """Exact-match rate over the six dimensions (what SA_acc measures)."""
    p, g = parse_osats(pred), parse_osats(gt)
    if not g:
        return 0.0
    return float(np.mean([1.0 if p.get(k) == v else 0.0 for k, v in g.items()]))


def osats_mae(pred: str, gt: str) -> float:
    p, g = parse_osats(pred), parse_osats(gt)
    if not g:
        return 5.0
    return float(np.mean([abs(p.get(k, 3) - v) for k, v in g.items()]))


# ----------------------------------------------------------------------- CVS
def parse_cvs(text: str):
    got = {}
    for m in RE_CVS.finditer(text or ""):
        key = m.group(1).strip().lower()
        for d in CVS_DIMS:
            if d.lower() in key or key in d.lower():
                got[d] = int(m.group(2))
                break
    if len(got) < 3:
        nums = [int(x) for x in re.findall(r"\b([0-2])\b", text or "")][:3]
        for d, v in zip(CVS_DIMS, nums):
            got.setdefault(d, v)
    return [int(got.get(d, 0)) for d in CVS_DIMS]


def fmt_cvs(v) -> str:
    return ", ".join(f"{d}: {int(x)}" for d, x in zip(CVS_DIMS, v))


# ------------------------------------------------------------------ RC_info
def anchor_index_from_rcinfo(sample):
    """RC_info['start_frame'] is a FILE PATH, not an integer index."""
    rc = sample.get("RC_info") or {}
    sf = rc.get("start_frame")
    if sf is None:
        return None
    vids = sample.get("video") or []
    if isinstance(sf, str) and not sf.strip().lstrip("-").isdigit():
        if sf in vids:
            return vids.index(sf)
        base = sf.rsplit("/", 1)[-1]
        for i, p in enumerate(vids):
            if p.rsplit("/", 1)[-1] == base:
                return i
        return None
    try:
        n = int(float(sf))
    except (TypeError, ValueError):
        return None
    arr = sample.get("sampled_video_frames") or []
    if arr:
        return int(np.argmin(np.abs(np.asarray(arr, dtype=float) - n)))
    return None


# ---------------------------------------------------------------- struc_info
def struc_facts(sample):
    """The pre-parsed structured ground truth. Far better than regexing text."""
    out = {"procedure": None, "action_list": [], "actions": [], "spans": [],
           "events": []}
    si = sample.get("struc_info")
    if not si:
        return out
    if isinstance(si, dict):
        si = [si]
    for blk in si:
        if not isinstance(blk, dict):
            continue
        if blk.get("procedure"):
            out["procedure"] = blk["procedure"]
        for a in blk.get("action_list") or []:
            if a not in out["action_list"]:
                out["action_list"].append(a)
        act = blk.get("action")
        if act and act not in out["actions"]:
            out["actions"].append(act)
        for sp in blk.get("spans") or []:
            try:
                s, e = float(sp["start"]), float(sp["end"])
            except (KeyError, TypeError, ValueError):
                continue
            out["spans"].append((s, e))
            out["events"].append({"start": s, "end": e,
                                  "action": (act or "").lower(), "desc": ""})
        for k in ("bbox", "boxes", "bboxes"):
            if blk.get(k):
                out.setdefault("boxes", []).append(blk[k])
    out["spans"].sort()
    out["events"].sort(key=lambda e: e["start"])
    return out
