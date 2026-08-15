"""
STEP 1 — HDD -> SSD frame cache + motion-score precompute.

Why this matters more than it looks: you have ~200k small JPEGs on a spinning
disk. A 7200rpm HDD does ~150 random IOPS. At 64 frames/sample that is ~0.4 s of
pure seek time per sample, per epoch — the GPU will sit idle ~70% of the time.
Copying to SSD and downscaling turns this into a non-issue and shrinks ~18 GB to
~3 GB.

While we're touching every frame anyway, we compute a cheap per-frame
information score (grayscale frame-difference + spatial gradient energy). This
is the input to Adaptive Information Sampling in sampling.py. Doing it here
costs almost nothing; doing it at train time would cost everything.

    python -m src.cache_frames \
        --jsons medvidu_eccv2026_trainval.json cleaned_test_data_11_04.json \
        --src_prefix /root/data --hdd_root /mnt/hdd/valdata \
        --out /mnt/ssd/medvidu_frames --workers 16 --max_side 448
"""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def remap(p: str, src_prefix: str, hdd_root: str) -> str:
    """Map a json path onto disk. src_prefix may be '' (the test split uses
    relative paths like 'testdata/AVOS/...'), so str.replace is not safe here."""
    return os.path.join(hdd_root, rel_key(p, src_prefix))


def rel_key(p: str, src_prefix: str) -> str:
    """Stable relative key used as the cache path, e.g. AVOS/frames_15fps/xyz/1.jpg"""
    if src_prefix and p.startswith(src_prefix):
        return p[len(src_prefix):].lstrip("/")
    return p.lstrip("/")


def _process_video(job):
    """Resize+copy every frame of one video, and compute its motion scores."""
    key, paths_in, paths_out, max_side, jpeg_q = job
    prev = None
    scores = []
    n_ok = 0
    for pi, po in zip(paths_in, paths_out):
        try:
            if os.path.exists(po):
                im = Image.open(po).convert("RGB")
            else:
                im = Image.open(pi).convert("RGB")
                w, h = im.size
                if max(w, h) > max_side:
                    s = max_side / max(w, h)
                    im = im.resize((max(1, int(w * s)), max(1, int(h * s))),
                                   Image.BILINEAR)
                os.makedirs(os.path.dirname(po), exist_ok=True)
                im.save(po, "JPEG", quality=jpeg_q, optimize=False)
            n_ok += 1
        except Exception:
            scores.append(0.0)
            continue

        # --- information score on a tiny thumbnail (fast) ---------------
        g = np.asarray(im.convert("L").resize((64, 64), Image.BILINEAR),
                       dtype=np.float32) / 255.0
        motion = 0.0 if prev is None else float(np.abs(g - prev).mean())
        gx = np.abs(np.diff(g, axis=1)).mean()
        gy = np.abs(np.diff(g, axis=0)).mean()
        detail = float(gx + gy)
        scores.append(motion * 3.0 + detail * 0.5)   # motion dominates
        prev = g
    if scores:
        scores[0] = float(np.median(scores)) if len(scores) > 1 else 1.0
    return key, np.asarray(scores, dtype=np.float32), n_ok, len(paths_in)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=None,
                    choices=["trainval", "test", "both"],
                    help="read json/root/prefix from configs/paths.yaml "
                         "(the two splits have different frame roots)")
    ap.add_argument("--jsons", nargs="+", default=None)
    ap.add_argument("--src_prefix", default="/root/data",
                    help="prefix baked into the json 'video' paths")
    ap.add_argument("--hdd_root", default=None,
                    help="only needed with --jsons; otherwise from paths.yaml")
    ap.add_argument("--out", default=None, help="SSD cache root")
    ap.add_argument("--max_side", type=int, default=448)
    ap.add_argument("--jpeg_q", type=int, default=88)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    # ---- resolve everything from configs/paths.yaml unless overridden -----
    from .paths import cfg as _cfg, need
    if args.split or not args.jsons:
        splits = (["trainval", "test"] if (args.split or "both") == "both"
                  else [args.split])
        args.out = args.out or need("ssd_cache")
        jobs_cfg = []
        for sp in splits:
            jp = _cfg(f"{sp}_json")
            if not jp:
                print(f"[cache] {sp}: not in paths.yaml, skipping")
                continue
            jobs_cfg.append((jp, _cfg(f"{sp}_frame_root"),
                             _cfg(f"{sp}_src_prefix", "") or ""))
        if not jobs_cfg:
            raise SystemExit("[cache] nothing to do — run src.paths --write first")
        for jp, root, pref in jobs_cfg:
            print(f"\n[cache] === split json={jp}\n"
                  f"[cache]     root={root} prefix='{pref}'")
            _run_one([jp], pref, root, args.out, args.max_side, args.jpeg_q,
                     args.workers)
        return

    # ---- collect unique frames, grouped by video so motion scores are ordered
    _run_one(args.jsons, args.src_prefix, args.hdd_root, args.out,
             args.max_side, args.jpeg_q, args.workers)


def _run_one(jsons, src_prefix, hdd_root, out, max_side, jpeg_q, workers):
    videos = {}   # video_key -> ordered unique frame paths
    for jf in jsons:
        with open(jf) as f:
            data = json.load(f)
        for s in data:
            vk = s["metadata"]["video_id"]
            ds = s["dataset_name"]
            key = f"{ds}/{vk}"
            bucket = videos.setdefault(key, {})
            for p in s["video"]:
                bucket[p] = None
    print(f"[cache] {len(videos)} videos, "
          f"{sum(len(v) for v in videos.values())} unique frame refs")

    jobs = []
    for key, bucket in videos.items():
        # sort by numeric frame index in the filename so motion diffs are temporal
        paths = sorted(bucket.keys(),
                       key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
                       if os.path.splitext(os.path.basename(p))[0].isdigit() else 0)
        pin = [remap(p, src_prefix, hdd_root) for p in paths]
        pout = [os.path.join(out, rel_key(p, src_prefix)) for p in paths]
        jobs.append((key, pin, pout, max_side, jpeg_q))

    os.makedirs(out, exist_ok=True)
    score_dir = os.path.join(out, "_scores")
    os.makedirs(score_dir, exist_ok=True)

    t0, done, tot_ok, tot_n = time.time(), 0, 0, 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_process_video, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            key, scores, n_ok, n = fut.result()
            np.save(os.path.join(score_dir, key.replace("/", "__") + ".npy"), scores)
            done += 1
            tot_ok += n_ok
            tot_n += n
            if done % 20 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"  {done}/{len(jobs)} videos  frames {tot_ok}/{tot_n}  "
                      f"{el:.0f}s  eta {el/done*(len(jobs)-done):.0f}s", flush=True)

    missing = tot_n - tot_ok
    print(f"\n[cache] done. cached {tot_ok} frames, {missing} missing/failed.")
    if missing > 0.02 * tot_n:
        print("  !! >2% frames failed. Check --hdd_root and --src_prefix.",
              file=sys.stderr)
    # write a manifest so downstream code knows how to remap
    with open(os.path.join(out, "_manifest.json"), "w") as f:
        json.dump({"src_prefix": src_prefix, "cache_root": out,
                   "max_side": max_side, "n_frames": tot_ok}, f, indent=2)
    print(f"[cache] manifest -> {out}/_manifest.json")


if __name__ == "__main__":
    main()
