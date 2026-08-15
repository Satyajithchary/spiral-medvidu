"""
Paired significance testing between two prediction sets.

An ablation delta is meaningless without knowing whether it exceeds sampling
noise. Because both variants score the SAME rows with the SAME model and differ
only in the operator under test, the comparison is paired, and a paired test is
far more sensitive than comparing two independent means.

Reports, per task:
  * the mean of each variant and the paired difference
  * a paired bootstrap 95 percent confidence interval on that difference
  * a Wilcoxon signed rank p value over the rows that actually differ
  * the count of rows improved, worsened and unchanged

    python -m src.paired_test --a preds/val_full/raw_predictions.json \
        --b preds/abl_B_no_tzoom/raw_predictions.json --gt data/val.json \
        --label_a "complete" --label_b "no temporal resampling"

Interpretation. When the interval spans zero the two configurations are not
distinguished by the evidence, and the honest wording is that no measurable
effect was detected rather than that the operator helped or harmed.
"""
from __future__ import annotations
import argparse, json, collections
import numpy as np

from .formats import qhash
from .sampling import normalize_task
from .prompts import get_answer
from .rewards import raw_metric

TASKS = ["tal", "stg", "dense_captioning", "video_summary", "region_caption",
         "next_action", "cvs_assessment", "skill_assessment"]


def load(path):
    d = json.load(open(path))
    d = list(d.values()) if isinstance(d, dict) else d
    out = {}
    for p in d:
        out[(p["id"], p["qa_type"], p.get("qhash") or "")] = p["prediction"]
        out.setdefault((p["id"], p["qa_type"]), p["prediction"])
    return out


def paired_bootstrap(diff, n_boot=20000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(diff)
    if n == 0:
        return 0.0, 0.0, 0.0
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), \
        float(np.percentile(means, 97.5))


def wilcoxon(diff):
    """Two sided Wilcoxon signed rank over non-zero differences."""
    d = diff[diff != 0]
    n = len(d)
    if n < 6:
        return None, n
    order = np.argsort(np.abs(d))
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)
    # average ranks within ties of |d|
    a = np.abs(d)[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[j + 1] == a[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = np.arange(i + 1, j + 2).mean()
        i = j + 1
    w_pos = ranks[d > 0].sum()
    w_neg = ranks[d < 0].sum()
    w = min(w_pos, w_neg)
    mu = n * (n + 1) / 4.0
    sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sd == 0:
        return None, n
    z = (w - mu + 0.5) / sd
    from math import erfc, sqrt
    p = erfc(abs(z) / sqrt(2))
    return float(p), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="reference prediction set")
    ap.add_argument("--b", required=True, help="variant prediction set")
    ap.add_argument("--gt", default="data/val.json")
    ap.add_argument("--label_a", default="A")
    ap.add_argument("--label_b", default="B")
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    PA, PB = load(a.a), load(a.b)
    rows = [r for r in json.load(open(a.gt)) if "##aux_" not in r["id"]]
    want = [t.strip() for t in a.tasks.split(",")] if a.tasks else TASKS

    print(f"\n{a.label_a}  vs  {a.label_b}")
    print("=" * 92)
    print(f"{'task':18s}{'n':>5s}{a.label_a[:9]:>10s}{a.label_b[:9]:>10s}"
          f"{'diff':>10s}{'95% CI':>20s}{'p':>9s}{'+/-/=':>12s}")
    print("-" * 92)

    results = {}
    for t in want:
        sa, sb = [], []
        for r in rows:
            if normalize_task(r["qa_type"]) != t:
                continue
            k3 = (r["id"], r["qa_type"], qhash(r))
            k2 = (r["id"], r["qa_type"])
            if k3 not in PA and k2 not in PA:
                continue
            if k3 not in PB and k2 not in PB:
                continue
            g = get_answer(r) or ""
            pa = PA.get(k3, PA.get(k2, ""))
            pb = PB.get(k3, PB.get(k2, ""))
            sa.append(raw_metric(pa, g, r["qa_type"]))
            sb.append(raw_metric(pb, g, r["qa_type"]))
        if not sa:
            continue
        sa, sb = np.asarray(sa), np.asarray(sb)
        diff = sa - sb                      # positive means A above B
        m, lo, hi = paired_bootstrap(diff)
        p, n_ne = wilcoxon(diff)
        up = int((diff > 0).sum())
        dn = int((diff < 0).sum())
        eq = int((diff == 0).sum())
        sig = "" if (p is None or p >= 0.05) else "  *"
        span0 = lo <= 0.0 <= hi
        results[t] = {"n": len(sa), "mean_a": float(sa.mean()),
                      "mean_b": float(sb.mean()), "diff": m,
                      "ci_low": lo, "ci_high": hi, "p": p,
                      "n_better": up, "n_worse": dn, "n_equal": eq,
                      "ci_spans_zero": bool(span0)}
        pstr = "  n/a" if p is None else f"{p:.3f}"
        print(f"{t:18s}{len(sa):5d}{sa.mean():10.4f}{sb.mean():10.4f}"
              f"{m:+10.4f}   [{lo:+.4f},{hi:+.4f}]{pstr:>9s}"
              f"{up:5d}/{dn:d}/{eq:d}{sig}")

    print("-" * 92)
    print("diff is A minus B. A positive diff means the reference scores higher.")
    print("An interval containing zero means the two are NOT distinguished by")
    print("this evidence, and the correct wording is that no measurable effect")
    print("was detected. An asterisk marks p < 0.05.")

    if a.out:
        json.dump(results, open(a.out, "w"), indent=2)
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
