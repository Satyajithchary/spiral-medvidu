"""
STEP 4b - Local evaluation on the video-disjoint val split.

Reproduces the leaderboard's metric names:
    CVS_acc  NAP_acc  SA_acc  STG_mIoU  TAG_mIoU@0.3  TAG_mIoU@0.5  DVC_F1
plus proxy scores for the three LLM-judge columns (DVC_llm/VS_llm/RC_llm) -
and, optionally, the REAL GPT judge on a subsample if an API key is available.

Also prints PER-DATASET breakdowns. This is the diagnostic MedGRPO's own paper
shows matters most: their failure mode was easy datasets swamping hard ones.

    python -m src.evaluate --preds preds/val/raw_predictions.json --gt data/val.json
    python -m src.evaluate ... --judge --judge_n 150   # needs OPENAI_API_KEY
"""
from __future__ import annotations
import argparse, json, collections, os, numpy as np
from .rewards import best_tiou, dvc_f1, caption_proxy, raw_metric, format_ok
from .formats import (parse_spans, parse_events, parse_cvs, parse_osats,
                      stg_miou, osats_score, osats_mae)
from .sampling import normalize_task
from .prompts import get_answer

JUDGE_PROMPT = """You are grading a surgical video caption against a reference.
Rate how CLOSELY the generated caption matches the reference on five dimensions,
1-5 each (5 = semantically equivalent, 1 = completely different):
R1 medical terminology precision
R2 instrument & anatomy identification
R3 specificity vs vagueness
R4 clinical procedure context
R5 action & state accuracy

REFERENCE: {ref}
GENERATED: {gen}

Reply with exactly five integers separated by spaces, nothing else."""


def llm_judge(pairs, model="gpt-4.1", n_workers=8):
    from openai import OpenAI
    from concurrent.futures import ThreadPoolExecutor
    cl = OpenAI()

    def one(p):
        gen, ref = p
        try:
            r = cl.chat.completions.create(
                model=model, temperature=0,
                messages=[{"role": "user",
                           "content": JUDGE_PROMPT.format(ref=ref[:1500],
                                                          gen=gen[:1500])}])
            nums = [int(x) for x in r.choices[0].message.content.split()[:5]]
            return sum(nums) / len(nums) if nums else None
        except Exception:
            return None
    with ThreadPoolExecutor(n_workers) as ex:
        return [s for s in ex.map(one, pairs) if s is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--embedder", default=None,
                    help="e.g. sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--judge_n", type=int, default=120)
    ap.add_argument("--out", default=None)
    ap.add_argument("--raw", action="store_true",
                    help="skip postprocess repairs (default is to apply them, "
                         "so the val number matches what the leaderboard sees)")
    a = ap.parse_args()

    P = json.load(open(a.preds))
    if isinstance(P, dict):
        P = list(P.values())
    # (id, qa_type) - id alone collides across tasks on the same clip
    from .formats import qhash as _qh
    pred = {(p["id"], p["qa_type"], p.get("qhash") or ""): p["prediction"]
            for p in P}
    # tolerate predictions written before the qhash fix
    pred.update({(p["id"], p["qa_type"]): p["prediction"] for p in P})

    # apply the same repairs the submission gets, otherwise val under-reports
    if not a.raw:
        from .postprocess import (fix_tal, fix_stg, fix_dvc, fix_cvs, fix_osats,
                                  fix_caption, clip_duration)
        n_rep = 0
        for row in json.load(open(a.gt)):
            k = (row["id"], row["qa_type"], _qh(row))
            k = k if k in pred else (row["id"], row["qa_type"])
            if k not in pred:
                continue
            t = normalize_task(row["qa_type"])
            p0 = pred[k]
            dur = clip_duration(row)
            ds_ = row["dataset_name"]
            try:
                if t == "tal":
                    p1 = fix_tal(p0, dur)
                elif t == "stg":
                    p1 = fix_stg(p0, row, dur)
                elif t == "dense_captioning":
                    p1 = fix_dvc(p0, dur, ds_)
                elif t == "cvs_assessment":
                    p1 = fix_cvs(p0)
                elif t == "skill_assessment":
                    p1 = fix_osats(p0)
                elif t in ("video_summary", "region_caption"):
                    p1 = fix_caption(p0, ds_, t)
                else:
                    p1 = p0
            except Exception:
                p1 = p0
            if p1 != p0:
                n_rep += 1
            pred[k] = p1
        print(f"[eval] applied postprocess repairs to {n_rep} predictions "
              f"(use --raw to disable)")
    n_unhashed = sum(1 for p in P if not p.get("qhash"))
    if n_unhashed:
        print(f"[eval] !! {n_unhashed}/{len(P)} predictions carry no qhash. These "
              f"predate the question-hash fix;\n[eval] !! rows sharing "
              f"(id, qa_type) cannot be told apart. Regenerate before reporting.")
    if len({p["id"] for p in P}) < len(P):
        print(f"[eval] note: {len(P)} predictions span only "
              f"{len({p['id'] for p in P})} unique ids - keying by (id, qa_type)")
    gt_rows = json.load(open(a.gt))
    if len(P) > len(gt_rows) * 1.05:
        print(f"[eval] !! {len(P)} predictions for {len(gt_rows)} ground-truth "
              f"rows. Duplicate keys present; numbers below are NOT valid.")
    gt_rows = [r for r in gt_rows if not r["id"].endswith("_precision")
               and "##aux_" not in r["id"]]

    emb = None
    if a.embedder:
        from sentence_transformers import SentenceTransformer
        emb = SentenceTransformer(a.embedder)

    acc = collections.defaultdict(list)          # metric -> values
    per_ds = collections.defaultdict(lambda: collections.defaultdict(list))
    fmt_fail = collections.Counter()
    cap_pairs = collections.defaultdict(list)
    n_eval = 0

    n_missing = 0
    for row in gt_rows:
        rid = (row["id"], row["qa_type"], _qh(row))
        rid = rid if rid in pred else (row["id"], row["qa_type"])
        if rid not in pred:
            n_missing += 1
            continue
        n_eval += 1
        p, g = pred[rid], get_answer(row) or ""
        t = normalize_task(row["qa_type"])
        ds = row["dataset_name"]
        if not format_ok(p, row["qa_type"]):
            fmt_fail[t] += 1

        if t == "tal":
            iou = best_tiou(parse_spans(p), parse_spans(g))
            # "mIoU@t" is ambiguous. Report both readings:
            #   masked  = mean(IoU if IoU>=t else 0)   <- what we used
            #   recall  = fraction of rows with IoU>=t <- likely the leaderboard
            acc["TAG_mIoU@0.3"].append(iou if iou >= 0.3 else 0.0)
            acc["TAG_mIoU@0.5"].append(iou if iou >= 0.5 else 0.0)
            acc["TAG_R@0.3"].append(1.0 if iou >= 0.3 else 0.0)
            acc["TAG_R@0.5"].append(1.0 if iou >= 0.5 else 0.0)
            acc["TAG_rawIoU"].append(iou)
            per_ds[ds]["TAG_rawIoU"].append(iou)
        elif t == "stg":
            v = stg_miou(p, g)             # sequence-aware
            acc["STG_mIoU"].append(v); per_ds[ds]["STG_mIoU"].append(v)
        elif t == "dense_captioning":
            pe, ge = parse_events(p), parse_events(g)
            v = dvc_f1(pe, ge)
            acc["DVC_F1"].append(v); per_ds[ds]["DVC_F1"].append(v)
            cp = caption_proxy(" ".join(e["desc"] for e in pe),
                               " ".join(e["desc"] for e in ge), emb)
            acc["DVC_llm_proxy"].append(cp)
            cap_pairs["DVC"].append((p, g))
        elif t == "video_summary":
            acc["VS_llm_proxy"].append(caption_proxy(p, g, emb))
            per_ds[ds]["VS_llm_proxy"].append(caption_proxy(p, g, emb))
            cap_pairs["VS"].append((p, g))
        elif t == "region_caption":
            acc["RC_llm_proxy"].append(caption_proxy(p, g, emb))
            per_ds[ds]["RC_llm_proxy"].append(caption_proxy(p, g, emb))
            cap_pairs["RC"].append((p, g))
        elif t == "next_action":
            v = raw_metric(p, g, row["qa_type"])
            acc["NAP_acc"].append(v); per_ds[ds]["NAP_acc"].append(v)
        elif t == "cvs_assessment":
            pc, gc = parse_cvs(p), parse_cvs(g)
            acc["CVS_acc"].append(sum(x == y for x, y in zip(pc, gc)) / 3.0)
            acc["CVS_exact"].append(float(pc == gc))
            per_ds[ds]["CVS_acc"].append(sum(x == y for x, y in zip(pc, gc)) / 3.0)
        elif t == "skill_assessment":
            v = osats_score(p, g)          # six OSATS dims, exact match rate
            acc["SA_acc"].append(v); per_ds[ds]["SA_acc"].append(v)
            acc["SA_MAE"].append(osats_mae(p, g))

    print(f"\n=== LOCAL VAL  ({n_eval} scored / {len(gt_rows)} gt rows"
          + (f", {n_missing} MISSING PREDICTIONS" if n_missing else "") + ") ===")
    if n_missing:
        print("  !! incomplete inference - re-run src.infer (it resumes) before "
              "trusting any number below")
    order = ["CVS_acc", "CVS_exact", "NAP_acc", "SA_acc", "SA_MAE", "STG_mIoU",
             "TAG_mIoU@0.3", "TAG_mIoU@0.5", "TAG_R@0.3", "TAG_R@0.5",
             "TAG_rawIoU", "DVC_F1",
             "DVC_llm_proxy", "VS_llm_proxy", "RC_llm_proxy"]
    LB = {"CVS_acc": 0.898, "NAP_acc": 0.576, "SA_acc": 0.354, "STG_mIoU": 0.202,
          "TAG_mIoU@0.3": 0.504, "TAG_mIoU@0.5": 0.441,
          "TAG_R@0.3": 0.504, "TAG_R@0.5": 0.441, "DVC_F1": 0.480}
    res = {}
    for m in order:
        if not acc[m]:
            continue
        v = float(np.mean(acc[m]))
        res[m] = v
        tgt = LB.get(m)
        flag = ""
        if tgt is not None:
            flag = ("  BEATS #1" if v > tgt else f"  (#1 = {tgt:.3f}, "
                    f"gap {v-tgt:+.3f})")
        print(f"  {m:16s} {v:.4f}  n={len(acc[m]):5d}{flag}")

    if fmt_fail:
        print("\n  format failures (each one is a guaranteed zero):")
        for k, v in fmt_fail.most_common():
            print(f"    {k:20s} {v}")

    print("\n=== PER-DATASET (watch for easy/hard collapse) ===")
    for ds in sorted(per_ds):
        bits = [f"{m}={np.mean(v):.3f}(n={len(v)})"
                for m, v in sorted(per_ds[ds].items())]
        print(f"  {ds:16s} " + "  ".join(bits))

    if a.judge:
        if not os.environ.get("OPENAI_API_KEY"):
            print("\n[judge] OPENAI_API_KEY not set, skipping")
        else:
            print("\n=== REAL LLM JUDGE (subsample) ===")
            import random
            for k, pairs in cap_pairs.items():
                sub = random.Random(0).sample(pairs, min(a.judge_n, len(pairs)))
                scores = llm_judge(sub)
                if scores:
                    print(f"  {k}_llm  {np.mean(scores):.3f}  (n={len(scores)}, "
                          f"scale 1-5)")
                    res[f"{k}_llm"] = float(np.mean(scores))

    # per-cell scores in "dataset|task" form, consumed by src/balance.py to
    # upweight the cells the model is worst at in a second SFT pass
    for ds_, mm in per_ds.items():
        for m, v in mm.items():
            task = {"TAG_rawIoU": "tal", "STG_mIoU": "stg", "DVC_F1":
                    "dense_captioning", "VS_llm_proxy": "video_summary",
                    "RC_llm_proxy": "region_caption", "NAP_acc": "next_action",
                    "CVS_acc": "cvs_assessment", "SA_acc": "skill_assessment"}.get(m)
            if task:
                res[f"{ds_}|{task}"] = float(np.mean(v))

    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)
        print(f"\n[eval] -> {a.out}")


if __name__ == "__main__":
    main()
