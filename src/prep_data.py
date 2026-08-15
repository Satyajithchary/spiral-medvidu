"""
STEP 2 - Information Harvesting (IH) + dataset construction.

This is contribution #1 and the reason this pipeline should beat a plain SFT on
the same 6,270 samples.

The insight: the released split gives one (question, answer) pair per row.
But each *answer* is dense with structured facts, and multiple rows share a
video. So:

  (a) HARVEST - parse every GT answer into typed facts:
        time spans, instruments, verbs, anatomy, bboxes, action names, scores.
      One caption -> up to seven supervision streams.

  (b) INDEX - build a per-video fact table by fusing facts across ALL tasks that
      touch that video. A TAL row tells you when 'cutting' happens; a DVC row
      tells what it looked like; an RC row tells you which tool was where.

  (c) SYNTHESISE - emit auxiliary QA pairs from the fused index that no row in
      the original data contains ("which instruments are visible between 20 and
      35 seconds?", "list the actions in temporal order", "which quadrant is the
      grasper in at 12s?"). These are free labels, exactly the "use information
      other teams leave on the floor" idea.

  (d) SPLIT - by VIDEO ID, never by row. Same video in train and val = a
      meaningless val score and a bad decision.

Usage:
    python -m src.prep_data --trainval medvidu_eccv2026_trainval.json \
        --out data/ --cache /path/to/medvidu_frames --val_frac 0.12 --aux_ratio 0.35
"""
from __future__ import annotations
import argparse, json, os, re, random, collections
from .ontology import extract, preferred_surface
from .sampling import normalize_task
from .timebase import frame_times, clip_duration
from .prompts import get_answer
from .formats import (parse_spans, parse_events, parse_tboxes, parse_cvs,
                      parse_osats, struc_facts)

# ------------------------------------------------------------------ harvesting

RE_SPAN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|–|to|until)\s*(\d+(?:\.\d+)?)\s*"
                     r"(?:seconds?|secs?|s\b)?", re.I)
RE_SPAN_LABELLED = re.compile(
    r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*seconds?\s*:\s*([^:\n]+?)\s*:\s*(.+)",
    re.I)
RE_BOX = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]")
RE_SCORES = re.compile(r"\b([0-2])\b")
QUADRANTS = ["upper left", "upper right", "lower left", "lower right",
             "top left", "top right", "bottom left", "bottom right",
             "centre", "center", "midline"]


def harvest(sample) -> dict:
    """Parse one sample's GT into typed facts."""
    gt = get_answer(sample) or ""
    task = normalize_task(sample["qa_type"])
    f = {"task": task, "dataset": sample["dataset_name"],
         "video_id": sample["metadata"]["video_id"],
         "sample_id": sample["id"],
         "spans": [], "events": [], "boxes": [],
         "instruments": [], "verbs": [], "anatomy": [],
         "quadrants": [], "raw": gt}

    # (0) struc_info - the pre-parsed structured ground truth. This release
    #     ships it and it is strictly better than regexing the answer string:
    #     exact spans, the action label, AND the closed action vocabulary for
    #     the procedure. Regex parsing is now only the fallback.
    sf = struc_facts(sample)
    f["procedure"] = sf.get("procedure")
    f["action_list"] = sf.get("action_list") or []
    f["named_actions"] = sf.get("actions") or []
    f["events"].extend(sf.get("events") or [])
    f["spans"].extend(sf.get("spans") or [])

    # (1) labelled / unlabelled dense-caption events from the answer text
    for e in parse_events(gt):
        f["events"].append(e)
        f["spans"].append((e["start"], e["end"]))

    # (2) bare time spans (TAL) - zero-duration spans are real, keep them
    if not f["spans"]:
        f["spans"].extend(parse_spans(gt))

    # (3) time-indexed box sequence (STG)
    for t, b in parse_tboxes(gt):
        f["boxes"].append([float(v) for v in b])
    rc = sample.get("RC_info") or {}
    if rc.get("start_frame_bbox"):
        f["boxes"].append([float(v) for v in rc["start_frame_bbox"]])

    # (4) ontology terms
    ont = extract(gt)
    f["instruments"] = sorted(ont["instrument"])
    f["verbs"] = sorted(ont["verb"])
    f["anatomy"] = sorted(ont["anatomy"])

    # (5) spatial language
    low = gt.lower()
    f["quadrants"] = [q for q in QUADRANTS if q in low]

    # (6) task-specific scalars
    if task == "cvs_assessment":
        f["cvs"] = parse_cvs(gt)
    if task == "skill_assessment":
        f["osats"] = parse_osats(gt)       # six dimensions rated 1-5
    if task == "next_action":
        f["next_action"] = gt.strip()
    return f


# --------------------------------------------------------- video-level index

def build_index(samples) -> dict:
    """Fuse facts across every task touching the same video."""
    idx = collections.defaultdict(lambda: {
        "instruments": collections.Counter(), "verbs": collections.Counter(),
        "anatomy": collections.Counter(), "events": [], "spans": [],
        "dataset": None, "tasks": set(), "quadrants": collections.Counter(),
        "procedure": None, "action_list": []})
    for s in samples:
        f = harvest(s)
        v = idx[f["video_id"]]
        v["dataset"] = f["dataset"]
        if f.get("procedure"):
            v["procedure"] = f["procedure"]
        for a in f.get("action_list") or []:
            if a not in v.setdefault("action_list", []):
                v["action_list"].append(a)
        v["tasks"].add(f["task"])
        v["instruments"].update(f["instruments"])
        v["verbs"].update(f["verbs"])
        v["anatomy"].update(f["anatomy"])
        v["quadrants"].update(f["quadrants"])
        v["events"].extend(f["events"])
        v["spans"].extend(f["spans"])
        s["_facts"] = f
    for v in idx.values():
        v["tasks"] = sorted(v["tasks"])
        v["events"].sort(key=lambda e: e["start"])
        v["spans"] = sorted(set(v["spans"]))
    return dict(idx)


# ------------------------------------------------- auxiliary sample synthesis

def synth_aux(sample, index, rng) -> list[dict]:
    """Create auxiliary QA rows from harvested facts. Each returned dict is a
    lightweight record reusing the SAME frames as its parent sample."""
    f = sample["_facts"]
    vid = index.get(f["video_id"], {})
    out = []

    def mk(qtype, q, a):
        r = dict(sample)
        r.pop("_facts", None)
        r = {k: v for k, v in r.items() if not k.startswith("_")}
        r["id"] = f"{sample['id']}##aux_{qtype}"
        r["qa_type"] = sample["qa_type"]          # keeps frame budget/mode
        r["_aux_type"] = qtype
        r["conversations"] = [{"from": "human", "value": "<video>\n" + q},
                              {"from": "gpt", "value": a}]
        out.append(r)

    # --- A. instrument inventory (from any task on this video) --------------
    inst = f["instruments"] or [i for i, _ in vid.get("instruments", {}).most_common(3)]
    if inst:
        mk("inventory",
           "List every surgical instrument visible in this clip, comma separated, "
           "using the standard name for each. No other text.",
           ", ".join(preferred_surface(i) for i in inst))

    # --- B. instrument-action-target triplet -------------------------------
    if f["instruments"] and f["verbs"]:
        tgt = f["anatomy"][0] if f["anatomy"] else "unspecified"
        mk("triplet",
           "State the primary surgical action as a triplet in the format "
           "<instrument, action, target>. No other text.",
           f"<{preferred_surface(f['instruments'][0])}, {f['verbs'][0]}, "
           f"{preferred_surface(tgt)}>")

    # --- C. temporal ordering of events ------------------------------------
    if len(f["events"]) >= 2:
        mk("ordering",
           "List the actions in this clip in temporal order, one per line, "
           "prefixed by their start time in seconds.",
           "\n".join(f"{e['start']:.1f}s: {e['action']}" for e in f["events"]))

    # --- D. temporal localisation of a harvested action --------------------
    if f["events"]:
        e = rng.choice(f["events"])
        mk("when",
           f"During which time interval does '{e['action']}' occur? "
           f"Answer only as <start>-<end> seconds.",
           f"{e['start']:.1f}-{e['end']:.1f} seconds")

    # --- E. spatial quadrant -----------------------------------------------
    if f["quadrants"] and f["instruments"]:
        mk("where",
           f"In which quadrant of the field is the "
           f"{preferred_surface(f['instruments'][0])} located? "
           f"Answer with the quadrant only.",
           f["quadrants"][0])

    # --- F0. closed-vocabulary recall, straight from struc_info ------------
    if f.get("action_list") and f.get("named_actions"):
        mk("vocab_recall",
           "From this procedure's action list, name every action that occurs "
           "in this clip. Comma separated, exact wording from the list.\n"
           "List: " + "; ".join(f["action_list"]),
           ", ".join(f["named_actions"]))

    # --- F. cross-task counterfactual: what is NOT happening ---------------
    #     Uses the video-level index; a fact no single row contains.
    all_v = set(vid.get("verbs", {}).keys())
    if all_v and f["verbs"]:
        absent = sorted(all_v - set(f["verbs"]))
        if absent:
            mk("absent",
               "Name one action that does NOT occur in this particular clip, "
               "chosen from actions that occur elsewhere in this procedure.",
               absent[0])

    # --- G. densification: rewrite a vague description precisely ------------
    if f["task"] in ("region_caption", "video_summary") and f["instruments"]:
        mk("precision",
           "Describe the primary instrument-tissue interaction in one sentence. "
           "Name the instrument, the action, the anatomy, and the quadrant.",
           f["raw"])
    return out


# ------------------------------------- C3: consistency-conditioned training
#
# CTCD injects the model's OWN pass-1 timeline into captioning prompts at test
# time. If we never train that way, it is a train/test mismatch and the model
# will either ignore the prior or, worse, copy it verbatim including its errors.
#
# So during training we inject a NOISED ground-truth timeline that simulates
# what an imperfect pass-1 actually produces: jittered boundaries, dropped
# segments, spurious segments, and sometimes nothing at all. The model learns
# the right behaviour - use the prior as a hint, override it when the frames
# disagree - instead of learning to trust or ignore it unconditionally.
#
# The noise parameters below are deliberately pessimistic relative to a trained
# model's real pass-1 error. Over-noising is safe (the model learns scepticism);
# under-noising teaches blind copying, which is the failure we must avoid.

CTCD_DROP_P = 0.25        # a real segment the pass-1 model missed
CTCD_FALSE_P = 0.30       # a segment pass-1 hallucinated
CTCD_JITTER = 0.15        # boundary noise, as a fraction of segment duration
CTCD_NO_PRIOR_P = 0.15    # sometimes give no prior, so the model still works alone


def noised_prior(sample, index, rng) -> str | None:
    """Build a plausible-but-wrong 'previous pass' timeline for this video."""
    if rng.random() < CTCD_NO_PRIOR_P:
        return None
    vid = index.get(sample["metadata"]["video_id"])
    if not vid:
        return None

    events = list(vid.get("events") or [])
    spans = [(s, e, "action") for s, e in (vid.get("spans") or [])]
    items = [(e["start"], e["end"], e.get("action", "action")) for e in events] \
        or spans
    if not items:
        return None

    horizon = max(e for _, e, _ in items) or 1.0
    out = []
    for s, e, name in items[:8]:
        if rng.random() < CTCD_DROP_P:
            continue
        dur = max(0.5, e - s)
        js = max(0.0, s + rng.gauss(0, CTCD_JITTER * dur))
        je = max(js + 0.5, e + rng.gauss(0, CTCD_JITTER * dur))
        out.append((js, je, name))

    if rng.random() < CTCD_FALSE_P:
        fs = rng.uniform(0, max(1.0, horizon * 0.8))
        out.append((fs, fs + rng.uniform(2.0, 12.0),
                    rng.choice([n for _, _, n in items]) if items else "action"))

    if not out:
        return None
    out.sort()
    line = "; ".join((f"{n} {s:.1f}-{e:.1f}s" if n and n.strip()
                      else f"{s:.1f}-{e:.1f}s") for s, e, n in out[:8])
    return ("Action timeline you previously derived for this video: " + line)


def attach_ctcd_priors(rows, index, rng, tasks=("dense_captioning",
                                                "video_summary",
                                                "region_caption")):
    """Store the noised prior on the row; dataset.py feeds it to build_system."""
    n_with = 0
    for r in rows:
        if normalize_task(r["qa_type"]) not in tasks:
            continue
        p = noised_prior(r, index, rng)
        if p:
            r["_ctcd_prior"] = p
            n_with += 1
    return n_with


# --------------------------------------------------------------------- splits

def video_split(samples, val_frac, seed=0):
    """Split by video_id, stratified so every (dataset, task) appears in val."""
    rng = random.Random(seed)
    by_video = collections.defaultdict(list)
    for s in samples:
        by_video[s["metadata"]["video_id"]].append(s)
    vids = sorted(by_video)
    rng.shuffle(vids)

    # greedy: fill val until it holds val_frac of rows AND covers all task/ds pairs
    need = {(s["dataset_name"], normalize_task(s["qa_type"])) for s in samples}
    val_vids, covered, n_val = set(), set(), 0
    target = val_frac * len(samples)
    for v in vids:
        pairs = {(s["dataset_name"], normalize_task(s["qa_type"]))
                 for s in by_video[v]}
        if n_val < target or (pairs - covered):
            val_vids.add(v)
            covered |= pairs
            n_val += len(by_video[v])
        if n_val >= target and covered >= need:
            break
    train = [s for s in samples if s["metadata"]["video_id"] not in val_vids]
    val = [s for s in samples if s["metadata"]["video_id"] in val_vids]
    return train, val, sorted(val_vids)


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainval", default=None)
    ap.add_argument("--out", default="data")
    ap.add_argument("--cache", default=None, help="SSD frame cache root")
    ap.add_argument("--val_frac", type=float, default=0.12)
    ap.add_argument("--aux_ratio", type=float, default=0.35,
                    help="aux samples as a fraction of real training samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ctcd_train", type=int, default=1,
                    help="C3: attach noised timeline priors to captioning rows")
    args = ap.parse_args()
    from .paths import cfg as _cfg, need
    args.trainval = args.trainval or need("trainval_json")
    args.cache = args.cache or _cfg("ssd_cache")
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)

    with open(args.trainval) as f:
        data = json.load(f)
    print(f"[prep] loaded {len(data)} rows")

    index = build_index(data)
    print(f"[prep] video index: {len(index)} videos")
    cov = collections.Counter(len(v["tasks"]) for v in index.values())
    print(f"[prep] tasks per video: {dict(sorted(cov.items()))}"
          "   <- videos with >1 task are what CTCD exploits at test time")

    train, val, val_vids = video_split(data, args.val_frac, args.seed)
    print(f"[prep] split by VIDEO: train={len(train)} rows / "
          f"{len(set(s['metadata']['video_id'] for s in train))} vids | "
          f"val={len(val)} rows / {len(val_vids)} vids")

    # ---- auxiliary synthesis on the TRAIN half only ----------------------
    aux_pool = []
    for s in train:
        aux_pool.extend(synth_aux(s, index, rng))
    rng.shuffle(aux_pool)
    n_aux = int(args.aux_ratio * len(train))
    aux = aux_pool[:n_aux]
    print(f"[prep] harvested {len(aux_pool)} candidate aux rows, keeping {len(aux)}")
    at = collections.Counter(a["_aux_type"] for a in aux)
    print("[prep] aux breakdown:", dict(at))

    train_final = train + aux
    rng.shuffle(train_final)

    # ---- C3: consistency-conditioned training -----------------------------
    if args.ctcd_train:
        n_ctcd = attach_ctcd_priors(train_final, index, rng)
        print(f"[prep] C3: attached noised timeline priors to {n_ctcd} "
              f"captioning rows "
              f"(drop={CTCD_DROP_P} false={CTCD_FALSE_P} jitter={CTCD_JITTER})")
        ex = next((r for r in train_final if "_ctcd_prior" in r), None)
        if ex:
            print(f"[prep]    example prior: {ex['_ctcd_prior'][:180]}")
    else:
        print("[prep] C3 DISABLED - CTCD at inference will be a train/test "
              "mismatch. Only do this for the ablation row.")

    for name, rows in [("train", train_final), ("val", val)]:
        p = os.path.join(args.out, f"{name}.json")
        with open(p, "w") as f:
            json.dump([{k: v for k, v in r.items() if k != "_facts"} for r in rows],
                      f)
        print(f"[prep] wrote {p}  ({len(rows)} rows)")

    with open(os.path.join(args.out, "video_index.json"), "w") as f:
        json.dump({k: {"dataset": v["dataset"], "tasks": v["tasks"],
                       "procedure": v.get("procedure"),
                       "action_list": v.get("action_list", []),
                       "instruments": dict(v["instruments"]),
                       "verbs": dict(v["verbs"]),
                       "anatomy": dict(v["anatomy"]),
                       "events": v["events"][:200],
                       "spans": [list(s) for s in v["spans"][:200]]}
                   for k, v in index.items()}, f)
    print(f"[prep] wrote {args.out}/video_index.json")
    with open(os.path.join(args.out, "val_videos.json"), "w") as f:
        json.dump(val_vids, f)

    # sanity: total supervision streams
    probe = data[:500]
    n_streams = 0
    for s in probe:
        h = harvest(s)
        n_streams += (len(h["events"]) + len(h["spans"]) + len(h["instruments"])
                      + len(h["verbs"]) + len(h["anatomy"]) + len(h["boxes"])
                      + len(h["quadrants"]))
    print(f"\n[prep] harvested ~{n_streams/max(1,len(probe)):.1f} typed facts "
          f"per row (vs 1 answer string in the raw data)")


if __name__ == "__main__":
    main()
