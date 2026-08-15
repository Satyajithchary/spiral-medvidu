"""
C2 — Difficulty-Balanced Multi-Source Sampling.

MedGRPO's central and correct observation is that heterogeneous medical datasets
have wildly different difficulty, and that raw reward magnitudes let the easy
ones dominate the gradient. Their fix is a median-centred logistic normaliser on
the REWARD, applied only during RL.

But the same imbalance exists in supervised fine-tuning and nobody normalises it
there. In the challenge split:

    NurViD        1395        JIGSAWS        150      (9.3x)
    TAL           1700        Region caption 310      (5.5x)

Under uniform sampling the model sees a NurViD row nine times for every JIGSAWS
row, and a TAL row five times for every region caption. Gradient share tracks
sample count, not task importance or difficulty — which is exactly the failure
mode the paper diagnoses one stage later.

We apply temperature-scaled inverse-frequency weighting over the (dataset, task)
product, the SFT analogue of their reward normalisation:

    w(d,t)  =  ( 1 / n(d,t) ) ** tau        tau in [0, 1]

    tau = 0   -> uniform over rows          (the baseline)
    tau = 1   -> uniform over (d,t) cells   (over-corrects, starves big cells)
    tau = 0.5 -> square-root balancing      (default; the usual sweet spot)

Optional difficulty term: after a validation pass you can supply per-cell scores
and upweight the cells the model is worst at, w *= (1 - acc)**gamma. That closes
the loop — the SFT sampler and the RL normaliser then use the same statistic.

This is one config line and roughly free, and it gives the paper a clean
symmetry: normalise the DATA distribution in stage 1, the REWARD distribution in
stage 2.
"""
from __future__ import annotations
import json, collections, math
import numpy as np

from .sampling import normalize_task


def cell_of(row):
    return (row.get("dataset_name", "?"), normalize_task(row.get("qa_type", "?")))


def compute_weights(rows, tau=0.5, difficulty=None, gamma=1.0,
                    aux_weight=0.7, verbose=True):
    """Return a per-row weight array.

    difficulty: optional {"dataset|task": accuracy_in_0_1} from a val pass.
                Cells the model is bad at get upweighted by (1-acc)**gamma.
    aux_weight: harvested auxiliary rows are useful but synthetic; damp them a
                little so they cannot dominate the real supervision.
    """
    counts = collections.Counter(cell_of(r) for r in rows)
    base = {c: (1.0 / n) ** tau for c, n in counts.items()}

    if difficulty:
        for c in list(base):
            key = f"{c[0]}|{c[1]}"
            acc = difficulty.get(key)
            if acc is not None:
                base[c] *= max(0.05, (1.0 - float(acc))) ** gamma

    w = np.array([base[cell_of(r)] *
                  (aux_weight if "##aux_" in r.get("id", "") else 1.0)
                  for r in rows], dtype=np.float64)
    w = w / w.sum()

    if verbose:
        share = collections.defaultdict(float)
        for r, x in zip(rows, w):
            share[cell_of(r)] += x
        print(f"[balance] tau={tau} gamma={gamma} "
              f"cells={len(counts)} rows={len(rows)}")
        print(f"{'dataset|task':40s} {'n':>6s} {'uniform%':>9s} {'balanced%':>10s}")
        for c in sorted(counts, key=lambda k: -counts[k]):
            u = 100 * counts[c] / len(rows)
            b = 100 * share[c]
            print(f"{c[0]+'|'+c[1]:40s} {counts[c]:6d} {u:8.2f}% {b:9.2f}%")
        # effective sample size tells you how much diversity you kept
        ess = 1.0 / float((w ** 2).sum())
        print(f"[balance] effective sample size {ess:.0f} / {len(rows)} "
              f"({100*ess/len(rows):.0f}%)  <- keep above ~55%")
    return w


def make_sampler(rows, tau=0.5, difficulty=None, gamma=1.0, aux_weight=0.7,
                 num_samples=None, verbose=True):
    from torch.utils.data import WeightedRandomSampler   # lazy: keeps this
    w = compute_weights(rows, tau, difficulty, gamma, aux_weight, verbose)  # module torch-free
    return WeightedRandomSampler(weights=w.tolist(),
                                 num_samples=num_samples or len(rows),
                                 replacement=True)


def difficulty_from_eval(eval_json_path):
    """Turn evaluate.py's per-dataset output into a difficulty dict.

    Expects a json of {"dataset|task": score}. Produce it with
    `python -m src.evaluate ... --out logs/val.json --per_cell`.
    """
    try:
        with open(eval_json_path) as f:
            d = json.load(f)
        return {k: v for k, v in d.items() if "|" in k and isinstance(v, (int, float))}
    except Exception:
        return None


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", default="data/train.json")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--difficulty", default=None)
    a = ap.parse_args()
    rows = json.load(open(a.train_json))
    compute_weights(rows, a.tau,
                    difficulty_from_eval(a.difficulty) if a.difficulty else None)
