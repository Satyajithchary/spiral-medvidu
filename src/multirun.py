"""
Multi-run evaluation for the stochastically decoded tasks.

Addresses the chair's second required revision. Ordinal expectation decoding
samples K generations at non-zero temperature, so skill assessment, critical
view of safety and next action prediction vary between otherwise identical
executions. The submitted paper reports next action prediction at 0.5638 and
0.5106 across two runs, which places the headline skill margin of 0.337 against
0.331 inside run-to-run variation.

This script repeats inference on the affected tasks alone with a different seed
each time and reports mean and sample standard deviation. Only those tasks are
regenerated, so the cost is a fraction of a full pass.

    python -m src.multirun --runs 5                      # validation split
    python -m src.multirun --runs 5 --test_split         # hidden-test tasks only

Validation cost: 155 rows per run (SA 24, CVS 37, NAP 94) at roughly 1.5 s per
row, so about 4 minutes per run and 20 minutes for five.

Hidden-test cost: 1,478 rows per run (SA 160, CVS 648, NAP 670), about 37
minutes per run. Running skill assessment alone is 160 rows and roughly 4
minutes, which is the cheapest way to place an interval on the headline number.
Splice each run into the full submission with `postprocess --base` and use the
leaderboard's evaluate-only mode.
"""
from __future__ import annotations
import argparse, json, os, statistics, subprocess, sys

STOCHASTIC = ["skill_assessment", "cvs_assessment", "next_action"]
METRIC_OF = {"skill_assessment": "SA_acc", "cvs_assessment": "CVS_acc",
             "next_action": "NAP_acc"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--tasks", default=",".join(STOCHASTIC))
    ap.add_argument("--gt", default="data/val.json")
    ap.add_argument("--test_split", action="store_true",
                    help="run on the hidden test split instead of validation")
    ap.add_argument("--base", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--adapter", default="runs/sft/final")
    ap.add_argument("--out_dir", default="preds/multirun")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    from .paths import cfg
    cache = cfg("ssd_cache")
    if a.test_split:
        a.gt = cfg("test_json")
        prefix, orig = cfg("test_src_prefix", "/root/data"), cfg("test_frame_root")
    else:
        prefix, orig = cfg("trainval_src_prefix", "/root/data"), cfg("trainval_frame_root")

    tasks = [t.strip() for t in a.tasks.split(",")]
    os.makedirs("logs", exist_ok=True)
    results = {t: [] for t in tasks}

    for r in range(a.runs):
        out = f"{a.out_dir}/run{r}"
        cmd = ["python", "-m", "src.infer", "--test", a.gt, "--base", a.base,
               "--adapter", a.adapter, "--cache", cache, "--src_prefix", prefix,
               "--out", out, "--refine", "--only_task", ",".join(tasks),
               "--seed", str(1000 + r)]
        if orig:
            cmd += ["--orig_root", orig]
        print("\n$ " + " ".join(cmd), flush=True)
        if not a.dry_run and subprocess.run(cmd).returncode != 0:
            print(f"  run {r} failed, skipping")
            continue

        ev = ["python", "-m", "src.evaluate", "--preds",
              f"{out}/raw_predictions.json", "--gt", a.gt,
              "--out", f"logs/multirun_{r}.json"]
        print("$ " + " ".join(ev), flush=True)
        if a.dry_run:
            continue
        subprocess.run(ev, stdout=subprocess.DEVNULL)
        if not os.path.exists(f"logs/multirun_{r}.json"):
            continue
        d = json.load(open(f"logs/multirun_{r}.json"))
        for t in tasks:
            v = d.get(METRIC_OF.get(t, ""))
            if v is not None:
                results[t].append(float(v))

    if a.dry_run:
        return

    print("\n" + "=" * 66)
    print(f"MULTI-RUN SUMMARY over {a.runs} seeds "
          f"({'hidden test' if a.test_split else 'validation'} split)")
    print("=" * 66)
    print(f"{'task':22s}{'mean':>9s}{'std':>9s}{'min':>9s}{'max':>9s}{'runs':>6s}")
    summary = {}
    for t in tasks:
        v = results[t]
        if not v:
            print(f"{t:22s}{'no data':>9s}")
            continue
        m = statistics.mean(v)
        s = statistics.stdev(v) if len(v) > 1 else 0.0
        print(f"{t:22s}{m:9.4f}{s:9.4f}{min(v):9.4f}{max(v):9.4f}{len(v):6d}")
        summary[t] = {"mean": m, "std": s, "min": min(v), "max": max(v),
                      "n_runs": len(v), "values": v}
    json.dump(summary, open("logs/multirun_summary.json", "w"), indent=2)
    print("\n-> logs/multirun_summary.json")
    print("\nReport as mean +/- std in the paper. If the standard deviation on "
          "skill assessment\nexceeds half the 0.006 margin over the competing "
          "entry, state that the two are\nnot separated by this evidence.")


if __name__ == "__main__":
    main()
