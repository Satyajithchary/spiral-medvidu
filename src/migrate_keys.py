"""
Migrate an existing prediction store onto the (id, qa_type, qhash) key, so the
re-run only has to generate the rows that were actually missed.

Your test run produced 5,644 predictions for 6,245 rows. Every prediction is
valid — it just needs to be attached to the right row. This script assigns each
existing prediction to the FIRST test row matching its (id, qa_type), stamps the
qhash, and rewrites the resume files. `src.infer` then sees ~601 rows still to
do instead of 6,245.

    python -m src.migrate_keys --preds preds/test_sft \
        --test "$(python -c 'from src.paths import cfg;print(cfg("test_json"))')"

Then re-run the same src.infer command; it resumes.
"""
from __future__ import annotations
import argparse, json, os, shutil, collections
from .formats import qhash
from .sampling import normalize_task


def load_jsonl(p):
    out = []
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="e.g. preds/test_sft")
    ap.add_argument("--test", required=True)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    rows = json.load(open(a.test))
    print(f"[migrate] test rows              : {len(rows)}")
    k2 = collections.Counter((r["id"], r["qa_type"]) for r in rows)
    k3 = collections.Counter((r["id"], r["qa_type"], qhash(r)) for r in rows)
    print(f"[migrate] unique (id,qa_type)    : {len(k2)}")
    print(f"[migrate] unique (id,qa_type,q#) : {len(k3)}")
    dup3 = {k: v for k, v in k3.items() if v > 1}
    if dup3:
        print(f"[migrate] !! {len(dup3)} keys STILL collide even with the "
              f"question hash — those rows are byte-identical duplicates and "
              f"one answer for them is correct.")

    # first row per (id, qa_type), in file order
    first = {}
    for r in rows:
        first.setdefault((r["id"], r["qa_type"]), r)

    by_task_missing = collections.Counter()
    have3 = set()

    for fname in ("pass1.jsonl", "refined.jsonl", "pass2.jsonl"):
        p = os.path.join(a.preds, fname)
        recs = load_jsonl(p)
        if not recs:
            continue
        out, n_stamped = [], 0
        for rec in recs:
            if rec.get("qhash"):
                out.append(rec)
                have3.add((rec["id"], rec["qa_type"], rec["qhash"]))
                continue
            row = first.get((rec["id"], rec["qa_type"]))
            if row is None:
                continue
            rec["qhash"] = qhash(row)
            n_stamped += 1
            out.append(rec)
            have3.add((rec["id"], rec["qa_type"], rec["qhash"]))
        print(f"[migrate] {fname:15s} {len(recs)} records, stamped {n_stamped}")
        if not a.dry_run:
            shutil.copy(p, p + ".bak")
            with open(p, "w") as f:
                for rec in out:
                    f.write(json.dumps(rec) + "\n")

    for r in rows:
        if (r["id"], r["qa_type"], qhash(r)) not in have3:
            by_task_missing[normalize_task(r["qa_type"])] += 1

    n_missing = sum(by_task_missing.values())
    print(f"\n[migrate] rows still needing inference: {n_missing}")
    for k, v in by_task_missing.most_common():
        print(f"    {k:20s} {v}")
    if a.dry_run:
        print("\n[migrate] DRY RUN — nothing written. Drop --dry_run to apply.")
    else:
        print(f"\n[migrate] resume files rewritten (.bak kept).")
        print("[migrate] now re-run the SAME src.infer command — it will do "
              f"only those {n_missing} rows.")


if __name__ == "__main__":
    main()
