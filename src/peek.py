"""
Look at what the model ACTUALLY wrote, per task, next to the ground truth.

Aggregate metrics tells whether a number is bad or not. This tells why. 

    python -m src.peek --preds preds/val_full/raw_predictions.json --gt data/val.json
    python -m src.peek --preds ... --gt ... --task stg --n 8
    python -m src.peek --preds ... --gt ... --duplicates      # id collision audit
"""
from __future__ import annotations
import argparse, json, collections
from .sampling import normalize_task
from .prompts import get_answer
from .rewards import raw_metric, format_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--task", default=None, help="filter to one task")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--n", type=int, default=4, help="examples per task")
    ap.add_argument("--worst", action="store_true",
                    help="show the lowest-scoring examples instead of the first")
    ap.add_argument("--duplicates", action="store_true",
                    help="audit duplicate ids and stop")
    a = ap.parse_args()

    gt_rows = [r for r in json.load(open(a.gt)) if "##aux_" not in r["id"]]

    # ---------------- duplicate-id audit --------------------------------
    if a.duplicates:
        from .formats import qhash as _qh2
        by_id = collections.defaultdict(list)
        for r in gt_rows:
            by_id[r["id"]].append(r["qa_type"])
        k2 = collections.Counter((r["id"], r["qa_type"]) for r in gt_rows)
        k3 = collections.Counter((r["id"], r["qa_type"], _qh2(r))
                                 for r in gt_rows)
        print(f"unique (id,qa_type)      : {len(k2)}")
        print(f"unique (id,qa_type,qhash): {len(k3)}   "
              f"<- {'UNIQUE' if len(k3)==len(gt_rows) else 'STILL COLLIDING'}")
        same_q = sum(v - 1 for v in k2.values() if v > 1)
        print(f"rows sharing id+qa_type  : {same_q} "
              f"(these got another question's answer)")
        dup = {k: v for k, v in by_id.items() if len(v) > 1}
        print(f"gt rows           : {len(gt_rows)}")
        print(f"unique ids        : {len(by_id)}")
        print(f"ids used >1 time  : {len(dup)}")
        if dup:
            print("\nThis is why predictions MUST be keyed by (id, qa_type).")
            print("Keying by id alone makes one task's answer overwrite another's.\n")
            combos = collections.Counter(tuple(sorted(set(v)))
                                         for v in dup.values())
            for c, n in combos.most_common(12):
                print(f"  {n:5d}x  {' + '.join(c)}")
        else:
            print("\nids are unique — collision is not the problem here.")
        return

    P = json.load(open(a.preds))
    if isinstance(P, dict):
        P = list(P.values())
    from .formats import qhash as _qh
    pred = {(p["id"], p["qa_type"], p.get("qhash") or ""): p["prediction"]
            for p in P}
    pred.update({(p["id"], p["qa_type"]): p["prediction"] for p in P})

    buckets = collections.defaultdict(list)
    for r in gt_rows:
        t = normalize_task(r["qa_type"])
        if a.task and t != a.task:
            continue
        if a.dataset and r["dataset_name"] != a.dataset:
            continue
        k = (r["id"], r["qa_type"], _qh(r))
        k = k if k in pred else (r["id"], r["qa_type"])
        if k not in pred:
            buckets[t].append((None, r, -1.0))
            continue
        p = pred[k]
        buckets[t].append((p, r, raw_metric(p, get_answer(r) or "", r["qa_type"])))

    for t in sorted(buckets):
        rows = buckets[t]
        miss = sum(1 for p, _, _ in rows if p is None)
        scored = [x for x in rows if x[0] is not None]
        bad_fmt = sum(1 for p, r, _ in scored if not format_ok(p, r["qa_type"]))
        avg = sum(s for _, _, s in scored) / max(1, len(scored))
        print("\n" + "=" * 78)
        print(f"### {t}   n={len(rows)}  missing={miss}  "
              f"format_fail={bad_fmt}  mean_metric={avg:.4f}")
        print("=" * 78)
        show = sorted(scored, key=lambda x: x[2])[:a.n] if a.worst \
            else scored[:a.n]
        for p, r, sc in show:
            g = get_answer(r) or ""
            ok = "OK " if format_ok(p, r["qa_type"]) else "BAD"
            print(f"\n  [{r['dataset_name']}]  score={sc:.3f}  fmt={ok}")
            print(f"  GT   : {g[:300]}")
            print(f"  PRED : {p[:300]}")
        if miss:
            print(f"\n  ({miss} rows have NO prediction — inference incomplete)")


if __name__ == "__main__":
    main()
