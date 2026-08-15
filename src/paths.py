"""
Path resolution — derives everything from the data, assumes nothing.

Why this is not trivial: the two splits use DIFFERENT path conventions.

  trainval json:  "/root/data/AVOS/frames_15fps/xyz/1.jpg"     absolute, prefixed
  test json:      "testdata/AVOS/frames_15fps/xyz/1.jpg"       relative

and the test frames are grouped under container names that are not the dataset
names ("CAMMA_data" holds CholecT50 + CholecTrack20, "JIGSAWS_frames" holds
JIGSAWS). So a single hard-coded src_prefix cannot work.

Algorithm, per split:
  1. sample ~40 real frame paths from the json
  2. enumerate candidate container directories under --root (bounded depth,
     never descending into frame dumps)
  3. for every (container D, split point k) pair, test whether
        D / "/".join(path_components[k:])
     exists for the sampled paths
  4. keep the (D, prefix) pair with the most hits

That recovers prefix="/root/data" + root=<...>/MedVIU_valdata for train, and
prefix="" + root=<...>/MedVIU_valdata for test, with no special-casing.

    python -m src.paths --root /path/to/MedVIU_valdata --write
    python -m src.paths --root /path/to/MedVIU_valdata \
        --test_json /explicit/path.json --write
"""
from __future__ import annotations
import argparse, json, os, glob, random, yaml, sys, collections

TRAIN_JSON_HINTS = ["medvidu_eccv2026_trainval.json", "*trainval*.json",
                    "*train_val*.json", "*_trainval_*.json"]
TEST_JSON_HINTS = ["cleaned_test_data_11_04.json", "cleaned_test_data*.json",
                   "*test_data*.json", "*testdata*.json", "*_test_*.json",
                   "test.json"]

# every name that has been observed as a top-level frame container, across both
# the MedVidU split and the MedGRPO test dump (which regroups some sources)
DATASET_DIRS = [
    "AVOS", "CholecT50", "CholecTrack20", "Cholec80_CVS", "Cholec80-CVS",
    "CoPESD", "EgoSurgery", "JIGSAWS", "NurViD",
    "CAMMA_data", "JIGSAWS_frames", "EgoSurgery_frames", "NurViD_frames",
    "Endoscapes", "cholec80", "camma",
]
SKIP_DIRS = {".git", "__pycache__", "_scores", ".cache", "node_modules"}


# --------------------------------------------------------------- json finding

def find_jsons(root, hints, depth=5):
    hits = []
    for h in hints:
        for d in range(depth + 1):
            for p in glob.glob(os.path.join(root, *(["*"] * d), h)):
                if os.path.isfile(p) and p not in hits:
                    hits.append(p)
    return hits


def all_data_files(root, depth=6, limit=200):
    """Every json / jsonl / parquet / zip that could plausibly be the split."""
    out = []
    for ext in ("*.json", "*.jsonl", "*.parquet", "*.zip"):
        for d in range(depth + 1):
            for p in glob.glob(os.path.join(root, *(["*"] * d), ext)):
                if os.path.isfile(p) and p not in out:
                    out.append(p)
                if len(out) >= limit:
                    return out
    return out


def looks_like_split(path, deep=True):
    """Is this a list of rows with video + conversations?

    The cheap head-of-file check is not enough: a single row's `video` field is
    50-180 frame paths at ~60 chars each, so the first record alone can exceed
    10 kB and the markers we want sit past it. Reading 4 kB rejected a valid
    65 MB trainval file. Read a real chunk, then fall back to parsing.
    """
    try:
        with open(path, errors="ignore") as f:
            head = f.read(2_000_000)          # ~2 MB covers several records
        if '"video"' in head and '"conversations"' in head:
            return True
        # still ambiguous -> parse it properly (bounded by file size)
        if deep and os.path.getsize(path) < 800e6 and path.endswith(".json"):
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list) and data and isinstance(data[0], dict):
                k = set(data[0])
                return "video" in k and ("conversations" in k or "qa_type" in k)
        return False
    except Exception:
        return False


def describe_tree(root, depth=2):
    """Show what is actually on disk, so a miss is diagnosable at a glance."""
    print(f"\n[paths] directory tree under {root} (depth {depth}):")

    def walk(d, lvl, pad):
        if lvl > depth:
            return
        try:
            entries = sorted(os.scandir(d), key=lambda e: e.name)
        except OSError:
            return
        dirs = [e for e in entries if e.is_dir() and e.name not in SKIP_DIRS]
        files = [e for e in entries if e.is_file()]
        ext = collections.Counter(os.path.splitext(f.name)[1] or "<none>"
                                  for f in files)
        if files:
            summary = ", ".join(f"{n}x{e}" for e, n in ext.most_common(5))
            print(f"{pad}  [{len(files)} files: {summary}]")
        for e in dirs[:25]:
            print(f"{pad}  {e.name}/")
            walk(e.path, lvl + 1, pad + "    ")
        if len(dirs) > 25:
            print(f"{pad}  ... and {len(dirs)-25} more directories")

    walk(root, 1, "")


# ----------------------------------------------------------- container search

def candidate_containers(root, depth=3, cap=800):
    """Directories that plausibly sit directly above the dataset folders."""
    cands, seen = [root], {os.path.abspath(root)}

    def walk(d, lvl):
        if lvl > depth or len(cands) >= cap:
            return
        try:
            entries = list(os.scandir(d))
        except OSError:
            return
        # never descend into a frame dump (hundreds of numeric subdirs)
        if len(entries) > 400:
            return
        for e in entries:
            if not e.is_dir() or e.name in SKIP_DIRS or e.name.startswith("."):
                continue
            ap = os.path.abspath(e.path)
            if ap not in seen:
                seen.add(ap)
                cands.append(e.path)
            walk(e.path, lvl + 1)

    walk(root, 1)
    return cands


def resolve_split(root, json_path, n_probe=40, verbose=True):
    """Return (frame_root, src_prefix, hits, tried)."""
    with open(json_path) as f:
        data = json.load(f)
    rng = random.Random(0)
    probes = []
    for s in rng.sample(data, min(80, len(data))):
        v = s.get("video") or []
        if v:
            probes.append(v[0])
            if len(v) > 2:
                probes.append(v[len(v) // 2])
    probes = probes[:n_probe]
    if not probes:
        return None, None, 0, 0

    comps = [p.replace("\\", "/").strip("/").split("/") for p in probes]
    containers = candidate_containers(root)
    if verbose:
        print(f"    probing {len(probes)} frames against {len(containers)} "
              f"candidate roots")

    # Frame filenames repeat across videos ("0.jpg" exists under every video
    # dir), so a short suffix matches almost anywhere and yields a degenerate
    # answer. Two guards:
    #   MIN_SUFFIX  the matched tail must keep >=3 components (dataset/.../file),
    #               so we are anchoring on directory structure, not a filename
    #   minimal k   iterate strip-depth from 0 upward and take the FIRST depth
    #               that clears the threshold, i.e. strip as little as possible
    # Ties inside a depth go to the container closest to root.
    MIN_SUFFIX = 3
    THRESH = 0.9
    max_k = max(len(c) - MIN_SUFFIX for c in comps)
    containers = sorted(containers, key=lambda d: (d.count(os.sep), len(d)))

    best = (None, None, 0)
    for k in range(0, max(0, min(4, max_k)) + 1):
        usable = [c for c in comps if len(c) - k >= MIN_SUFFIX]
        if not usable:
            continue
        for D in containers:
            hits = sum(os.path.exists(os.path.join(D, *c[k:])) for c in usable)
            if hits > best[2]:
                prefix = "/".join(comps[0][:k])
                if probes[0].startswith("/") and k > 0:
                    prefix = "/" + prefix
                best = (D, prefix, hits)
            if hits >= THRESH * len(usable):
                prefix = "/".join(comps[0][:k])
                if probes[0].startswith("/") and k > 0:
                    prefix = "/" + prefix
                return D, prefix, hits, len(usable)
    return best[0], best[1], best[2], len(comps)


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/path/to/MedVIU_valdata")
    ap.add_argument("--trainval_json", default=None, help="skip auto-detection")
    ap.add_argument("--test_json", default=None, help="skip auto-detection")
    ap.add_argument("--ssd_cache", default=None)
    ap.add_argument("--out", default="configs/paths.yaml")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--allow_partial", action="store_true",
                    help="write the yaml even if the test split is unresolved")
    a = ap.parse_args()

    root = os.path.abspath(a.root)
    print(f"[paths] root = {root}\n")

    # ---- locate the two json files -------------------------------------
    def pick(explicit, hints, label):
        if explicit:
            if not os.path.exists(explicit):
                print(f"[paths] {label}: given path does not exist: {explicit}",
                      file=sys.stderr)
                return None
            return explicit
        cands = find_jsons(root, hints)
        for c in cands:
            if looks_like_split(c):
                return c
        if cands:
            print(f"[paths] {label}: filename matched but content check failed "
                  f"for: {cands[:3]}", file=sys.stderr)
        return None

    tv = pick(a.trainval_json, TRAIN_JSON_HINTS, "trainval")
    te = pick(a.test_json, TEST_JSON_HINTS, "test")

    print(f"[paths] trainval json : {tv or 'NOT FOUND'}")
    print(f"[paths] test json     : {te or 'NOT FOUND'}")

    if not te or not tv:
        print("\n[paths] Candidate data files found under root:")
        found = all_data_files(root)
        if not found:
            print("    (none at all)")
        for p in found:
            mb = os.path.getsize(p) / 1e6
            tag = ""
            if p.endswith(".json"):
                tag = "  <- IS a split" if looks_like_split(p) else "  (not a split)"
            elif p.endswith(".parquet"):
                tag = "  <- parquet; convert with src/parquet_to_json.py"
            elif p.endswith(".zip"):
                tag = "  <- still zipped?"
            print(f"    {mb:9.1f} MB  {p}{tag}")
        describe_tree(root, depth=2)
        if not te:
            print("\n" + "=" * 66)
            print("THE TEST SPLIT JSON IS NOT ON DISK.")
            print("=" * 66)
            print("You have the testdata FRAMES but not the metadata json.")
            print("Get it (frames already present, so grab the json only):")
            print()
            print("  cd /path/to/MedVIU_valdata/testdata")
            print("  huggingface-cli download UII-AI/MedVidBench \\")
            print("      cleaned_test_data_11_04.json \\")
            print("      --repo-type dataset --local-dir .")
            print()
            print("  # or the whole repo if the filename has changed:")
            print("  huggingface-cli download UII-AI/MedVidBench \\")
            print("      --repo-type dataset --local-dir .")
            print()
            print("You do NOT need it for the next ~14 hours. Caching, data")
            print("prep and SFT all run on trainval alone. Start those now and")
            print("download in parallel:")
            print()
            print("  python -m src.paths --root %s --allow_partial --write"
                  % root)
            print("=" * 66)

    cfg = {}
    ok = True
    for key, jp in [("trainval", tv), ("test", te)]:
        if not jp:
            ok = False
            print(f"\n[paths] {key}: unresolved, skipping frame search")
            continue
        print(f"\n[paths] resolving {key} frames …")
        fr, prefix, hits, tot = resolve_split(root, jp)
        pct = 100 * hits / max(1, tot)
        print(f"[paths] {key}: frame_root = {fr}")
        print(f"[paths] {key}: src_prefix = '{prefix}'   ({hits}/{tot} probes "
              f"resolve, {pct:.0f}%)")
        if hits < tot * 0.85:
            ok = False
            with open(jp) as f:
                ex = json.load(f)[0]["video"][0]
            print(f"  !! low resolution rate. example path in json: {ex}",
                  file=sys.stderr)
            print(f"     find the directory D such that D + <that path minus "
                  f"its prefix> exists, and pass it manually.", file=sys.stderr)
        cfg[f"{key}_json"] = jp
        cfg[f"{key}_frame_root"] = fr
        cfg[f"{key}_src_prefix"] = prefix

    # keep the cache on the same volume as the data. Using dirname(root) put it
    # on /media when --root was /path/to, which is the wrong filesystem.
    cfg["ssd_cache"] = a.ssd_cache or os.path.join(root, "medvidu_cache")
    cfg["work_dir"] = os.path.join(os.getcwd(), "data")

    print("\n" + "-" * 60)
    print(yaml.safe_dump(cfg, sort_keys=False))
    if a.write and (ok or a.allow_partial):
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"[paths] wrote {a.out}"
              + ("" if ok else "   (PARTIAL — test split unresolved)"))
    elif a.write:
        print("[paths] NOT writing — resolution incomplete. Fix the errors "
              "above, or re-run with --allow_partial to proceed on train only.",
              file=sys.stderr)
        sys.exit(1)


def load(path="configs/paths.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def cfg(key, default=None, path="configs/paths.yaml"):
    """Read one key from configs/paths.yaml, tolerating a missing file.

    Every entry point uses this for its argparse defaults, so no command needs
    a shell variable. Shell variables do not survive between Jupyter `!` cells,
    which silently passed empty strings to half the pipeline."""
    try:
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        v = d.get(key)
        return v if v not in (None, "") else default
    except Exception:
        return default


def need(key, path="configs/paths.yaml"):
    import sys as _s
    v = cfg(key, None, path)
    if not v:
        _s.exit(f"\n[paths] '{key}' is not set in {path}.\n"
                f"  Run:  python -m src.paths --root /path/to/MedVIU_valdata "
                f"--write\n")
    return v


if __name__ == "__main__":
    main()
