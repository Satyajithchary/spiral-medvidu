"""
Cache coverage diagnostic.

Run this when inference reports PermissionError or FileNotFoundError on paths
under the original prefix. Those errors mean the dataloader fell back to the raw
JSON paths because no cached frame resolved for that row, and on a machine where
the original prefix exists but is unreadable the fallback fails hard.

    python -m src.check_cache --gt data/val.json
    python -m src.check_cache --gt data/val.json --fix

Reports coverage per source dataset and per file extension, and lists the exact
frames that are missing. With --fix it writes the missing frames into the cache,
provided the originals are reachable.
"""
from __future__ import annotations
import argparse, collections, json, os, sys

from .paths import cfg
from .cache_frames import rel_key, remap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/val.json")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--src_prefix", default=None)
    ap.add_argument("--frame_root", default=None)
    ap.add_argument("--sample", type=int, default=0,
                    help="check only the first N rows (0 checks all)")
    ap.add_argument("--fix", action="store_true")
    ap.add_argument("--max_side", type=int, default=448)
    a = ap.parse_args()

    tj = cfg("test_json") or ""
    is_test = bool(tj) and os.path.abspath(a.gt) == os.path.abspath(tj)
    cache = a.cache or cfg("ssd_cache")
    prefix = a.src_prefix or cfg(
        "test_src_prefix" if is_test else "trainval_src_prefix", "/root/data")
    root = a.frame_root or cfg(
        "test_frame_root" if is_test else "trainval_frame_root")
    print(f"[cache-check] gt={a.gt}\n[cache-check] cache={cache}\n"
          f"[cache-check] src_prefix={prefix!r}\n[cache-check] frame_root={root}\n")

    rows = json.load(open(a.gt))
    if a.sample:
        rows = rows[: a.sample]

    per_ds = collections.defaultdict(lambda: [0, 0])       # [hit, total]
    per_ext = collections.defaultdict(lambda: [0, 0])
    missing, dead_rows = [], []

    for r in rows:
        ds = r.get("dataset_name", "?")
        hit_this = 0
        for p in r.get("video") or []:
            cp = os.path.join(cache, rel_key(p, prefix))
            ext = os.path.splitext(p)[1].lower() or "<none>"
            ok = os.path.exists(cp)
            per_ds[ds][1] += 1
            per_ext[ext][1] += 1
            if ok:
                per_ds[ds][0] += 1
                per_ext[ext][0] += 1
                hit_this += 1
            elif len(missing) < 400000:
                missing.append((p, cp))
        if hit_this == 0 and (r.get("video") or []):
            dead_rows.append(r["id"])

    print(f"{'source':16s}{'cached':>10s}{'total':>10s}{'coverage':>10s}")
    bad_ds = []
    for ds in sorted(per_ds):
        h, t = per_ds[ds]
        cov = h / max(1, t)
        flag = "  <-- GAP" if cov < 0.999 else ""
        if cov < 0.999:
            bad_ds.append(ds)
        print(f"{ds:16s}{h:10d}{t:10d}{cov:9.3f}{flag}")

    print(f"\n{'extension':16s}{'cached':>10s}{'total':>10s}{'coverage':>10s}")
    for e in sorted(per_ext):
        h, t = per_ext[e]
        print(f"{e:16s}{h:10d}{t:10d}{h/max(1,t):9.3f}")

    print(f"\nrows with ZERO cached frames: {len(dead_rows)}")
    print("  These fall back to the original paths, which is the source of any")
    print("  PermissionError under the original prefix.")
    for i in dead_rows[:10]:
        print(f"    {i}")
    if len(dead_rows) > 10:
        print(f"    ... and {len(dead_rows)-10} more")

    if missing:
        print(f"\nmissing frames: {len(missing)}")
        for p, cp in missing[:6]:
            src_on_disk = remap(p, prefix, root)
            print(f"    json   : {p}")
            print(f"    cache  : {cp}   exists={os.path.exists(cp)}")
            print(f"    origin : {src_on_disk}   "
                  f"exists={os.path.exists(src_on_disk)}  "
                  f"readable={os.access(src_on_disk, os.R_OK)}")
            print()

    if bad_ds and not a.fix:
        print(f"\nRe-cache the affected sources with:\n"
              f"  python -m src.cache_frames --split "
              f"{'test' if is_test else 'trainval'} --workers 16\n"
              f"or repair only what is missing with:\n"
              f"  python -m src.check_cache --gt {a.gt} --fix")

    if a.fix and missing:
        from PIL import Image
        print(f"\n[fix] writing {len(missing)} frames into the cache")
        ok = fail = 0
        for n, (p, cp) in enumerate(missing):
            src_on_disk = remap(p, prefix, root)
            try:
                im = Image.open(src_on_disk).convert("RGB")
                w, h = im.size
                if max(w, h) > a.max_side:
                    s = a.max_side / max(w, h)
                    im = im.resize((max(1, int(w*s)), max(1, int(h*s))),
                                   Image.BILINEAR)
                os.makedirs(os.path.dirname(cp), exist_ok=True)
                im.save(cp, "JPEG", quality=88)
                ok += 1
            except Exception as e:
                fail += 1
                if fail <= 5:
                    print(f"  fail {src_on_disk}: {type(e).__name__}: {e}")
            if (n+1) % 2000 == 0:
                print(f"  {n+1}/{len(missing)}", flush=True)
        print(f"[fix] wrote {ok}, failed {fail}")
        if fail:
            print("[fix] remaining failures mean the originals are unreadable. "
                  "Check where the frames actually live and re-run src.paths.")
    elif not bad_ds:
        print("\nCoverage is complete. Any PermissionError has another cause.")


if __name__ == "__main__":
    main()
