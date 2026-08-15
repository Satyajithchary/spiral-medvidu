"""
Ablation table for the report - Python, no shell quoting.

    python -m src.run_ablations                    # B, C, D, G  (~50 min)
    python -m src.run_ablations --variants all     # adds E, F    (~3 h)
    python -m src.run_ablations --dry_run

The cost trick: every variant seeds its resume files from preds/val_full, so
only the stage that actually differs is recomputed.

  B/C/D  disable one refinement -> seed pass1 AND pass2, rerun refine only
         (~10 min each). Captioning columns are held fixed by construction,
         which is what you want: these ablations are about TAG/STG/SA.
  E      no CTCD              -> seed pass1 + refined, rerun pass2 (~55 min)
  F      no refinement        -> seed pass1, rerun pass2 (~55 min)
  G      neither              -> seed pass1, rerun pass2 (~55 min)

"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys

VARIANTS = {
    # name:        (extra infer flags, files to seed from the full run)
    "A_full":      ([], ["pass1.jsonl", "refined.jsonl", "pass2.jsonl"]),
    "B_no_tzoom":  (["--no_temporal_zoom"], ["pass1.jsonl", "pass2.jsonl"]),
    "C_no_szoom":  (["--no_spatial_zoom"], ["pass1.jsonl", "pass2.jsonl"]),
    "D_no_oed":    (["--no_oed"], ["pass1.jsonl", "pass2.jsonl"]),
    "E_no_ctcd":   (["__noctcd"], ["pass1.jsonl", "refined.jsonl"]),
    "F_no_refine": (["__norefine"], ["pass1.jsonl"]),
    "G_plain":     (["__noctcd", "__norefine"], ["pass1.jsonl"]),
}
CHEAP = ["A_full", "B_no_tzoom", "C_no_szoom", "D_no_oed", "G_plain"]

COLS = ["TAG_mIoU@0.3", "TAG_mIoU@0.5", "TAG_R@0.3", "TAG_R@0.5", "STG_mIoU",
        "SA_acc", "CVS_acc", "NAP_acc", "DVC_F1", "VS_llm_proxy", "RC_llm_proxy"]


def load_jsonl(path):
    out = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def run(cmd, dry=False):
    print("\n$ " + " ".join(cmd), flush=True)
    if dry:
        return 0
    return subprocess.run(cmd).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="cheap",
                    help="'cheap', 'all', or a comma list of variant names")
    ap.add_argument("--src", default="preds/val_full",
                    help="the completed full run to seed from")
    ap.add_argument("--gt", default="data/val.json")
    ap.add_argument("--base", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--adapter", default="runs/sft/final")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--keep", action="store_true",
                    help="do NOT wipe existing variant directories. Unsafe: a "
                         "stale refined.jsonl makes every variant a no-op.")
    a = ap.parse_args()

    from .paths import cfg
    cache = cfg("ssd_cache")
    prefix = cfg("trainval_src_prefix", "/root/data")
    orig = cfg("trainval_frame_root")
    if not cache:
        sys.exit("configs/paths.yaml missing ssd_cache - run src.paths --write")

    if a.variants == "cheap":
        names = CHEAP
    elif a.variants == "all":
        names = list(VARIANTS)
    else:
        names = [v.strip() for v in a.variants.split(",")]

    if not os.path.exists(os.path.join(a.src, "pass1.jsonl")):
        sys.exit(f"{a.src}/pass1.jsonl not found - run the full val inference first")

    # ---- GUARD 1: the seed source must carry question hashes ---------------
    # Records written before the (id, qa_type, qhash) fix have no qhash field.
    # Seeding them produces two keys per row, which silently doubles the merged
    # prediction file and makes the evaluation resolve arbitrary duplicates.
    seed_recs = load_jsonl(os.path.join(a.src, "pass1.jsonl"))
    n_hashed = sum(1 for r in seed_recs if r.get("qhash"))
    if seed_recs and n_hashed < len(seed_recs):
        sys.exit(
            f"\n[ablate] {a.src}/pass1.jsonl carries qhash on only "
            f"{n_hashed}/{len(seed_recs)} records.\n"
            f"  These predate the question-hash fix. Seeding them gives every row\n"
            f"  two keys, the merged file doubles, and every ablation becomes a\n"
            f"  no-op with arbitrary numbers.\n\n"
            f"  Regenerate the reference run first:\n"
            f"    rm -rf {a.src}\n"
            f"    python -m src.infer --test {a.gt} --base <BASE> "
            f"--adapter {a.adapter} \\\n"
            f"      --cache <CACHE> --src_prefix <PREFIX> --orig_root <ROOT> "
            f"--out {a.src} --ctcd --refine\n")

    # ---- GUARD 2: never reuse a stale variant directory --------------------
    if not a.keep and not a.dry_run:
        for name in names:
            if name == "A_full":
                continue
            d = f"preds/abl_{name}"
            if os.path.isdir(d):
                shutil.rmtree(d)
                print(f"[ablate] wiped stale {d}")

    os.makedirs("logs", exist_ok=True)
    for name in names:
        if name not in VARIANTS:
            print(f"  skip unknown variant {name}")
            continue
        flags, seed = VARIANTS[name]
        out = f"preds/abl_{name}"

        if name == "A_full":
            out = a.src
        else:
            os.makedirs(out, exist_ok=True)
            for f in seed:
                src_f = os.path.join(a.src, f)
                dst_f = os.path.join(out, f)
                if os.path.exists(src_f):
                    shutil.copy(src_f, dst_f)
                    print(f"  seeded {dst_f} "
                          f"({sum(1 for _ in open(dst_f))} rows)")
            # any stage NOT seeded must be absent, otherwise it will be skipped
            for f in ("pass1.jsonl", "refined.jsonl", "pass2.jsonl"):
                if f not in seed and os.path.exists(os.path.join(out, f)):
                    os.remove(os.path.join(out, f))
                    print(f"  removed {out}/{f} so it is recomputed")

            cmd = ["python", "-m", "src.infer", "--test", a.gt,
                   "--base", a.base, "--adapter", a.adapter,
                   "--cache", cache, "--src_prefix", prefix,
                   "--out", out]
            if orig:
                cmd += ["--orig_root", orig]
            if "__noctcd" not in flags:
                cmd += ["--ctcd"]
            if "__norefine" not in flags:
                cmd += ["--refine"]
            cmd += [f for f in flags if not f.startswith("__")]
            if run(cmd, a.dry_run) != 0:
                print(f"  !! {name} inference failed, skipping")
                continue
            rp = os.path.join(out, "raw_predictions.json")
            if os.path.exists(rp):
                n_pred = len(json.load(open(rp)))
                n_rows = len(json.load(open(a.gt)))
                if n_pred > n_rows * 1.05:
                    print(f"  !! {name}: {n_pred} predictions for {n_rows} rows. "
                          f"Duplicate keys present, results are NOT valid.")
                    continue

        run(["python", "-m", "src.evaluate",
             "--preds", os.path.join(out, "raw_predictions.json"),
             "--gt", a.gt, "--out", f"logs/abl_{name}.json"], a.dry_run)

    if a.dry_run:
        return

    print("\n" + "=" * 118)
    print("ABLATION TABLE  (validation split, video-disjoint)")
    print("=" * 118)
    res = {}
    for name in names:
        p = f"logs/abl_{name}.json"
        if os.path.exists(p):
            res[name] = json.load(open(p))
    if not res:
        print("no results")
        return
    print(f"{'variant':14s}" + "".join(f"{c:>14s}" for c in COLS))
    base = res.get("A_full")
    for name, v in res.items():
        row = f"{name:14s}"
        for c in COLS:
            x = v.get(c)
            row += f"{x:14.4f}" if x is not None else f"{'-':>14s}"
        print(row)
    if base:
        print("-" * 118)
        for name, v in res.items():
            if name == "A_full":
                continue
            row = f"{'d ' + name:14s}"
            for c in COLS:
                x, b = v.get(c), base.get(c)
                row += (f"{x - b:+14.4f}" if x is not None and b is not None
                        else f"{'-':>14s}")
            print(row)
        print("\nNegative deltas mean the removed component was helping.")
    json.dump(res, open("logs/ablation_table.json", "w"), indent=2)
    print("\n-> logs/ablation_table.json")


if __name__ == "__main__":
    main()
