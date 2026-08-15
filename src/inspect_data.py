"""
STEP 0.

Everything downstream assumes things about the JSON that inferred from the
dataset card. This script verifies them against the actual files and prints the
ground-truth answer formats per task, which is what the parsers key off.

    python -m src.inspect_data --train medvidu_eccv2026_trainval.json \
                               --test  cleaned_test_data_11_04.json

"""
import argparse, json, re, collections, os, random


def load(path):
    with open(path) as f:
        return json.load(f)


def infer_times(sample):
    """Two independent ways to get per-frame timestamps. They must agree."""
    frames = sample["sampled_video_frames"]
    fps = float(sample["metadata"]["fps"])
    # Method A: index-based. Frame k in the list is at k / fps seconds.
    t_a = [k / fps for k in range(len(frames))]
    # Method B: native-frame-index based. Needs native fps of the frame dump.
    diffs = [b - a for a, b in zip(frames, frames[1:]) if b > a]
    stride = sorted(diffs)[len(diffs) // 2] if diffs else 1
    native = stride * fps
    start = frames[0]
    t_b = [(k - start) / native for k in frames]
    return t_a, t_b, native


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=None)
    ap.add_argument("--test", default=None)
    ap.add_argument("--n_examples", type=int, default=2)
    args = ap.parse_args()
    from .paths import cfg as _cfg
    args.train = args.train or _cfg("trainval_json")
    args.test = args.test or _cfg("test_json")

    for name, path in [("TRAIN/VAL", args.train), ("TEST", args.test)]:
        if not path or not os.path.exists(path):
            print(f"\n### {name}: not provided / not found ({path})")
            continue
        data = load(path)
        print("\n" + "=" * 78)
        print(f"### {name}  ({path})  n={len(data)}")
        print("=" * 78)

        # --- keys -------------------------------------------------------
        keys = collections.Counter()
        for s in data:
            keys.update(s.keys())
        print("\n[keys]", dict(keys))
        md_keys = collections.Counter()
        for s in data:
            md_keys.update(s.get("metadata", {}).keys())
        print("[metadata keys]", dict(md_keys))

        # --- task / dataset distribution -------------------------------
        qa = collections.Counter(s["qa_type"] for s in data)
        ds = collections.Counter(s["dataset_name"] for s in data)
        print("\n[qa_type]")
        for k, v in qa.most_common():
            print(f"    {k:24s} {v}")
        print("[dataset_name]")
        for k, v in ds.most_common():
            print(f"    {k:24s} {v}")

        # --- has ground truth? ------------------------------------------
        n_gt = sum(1 for s in data
                   if any(m["from"] == "gpt" for m in s["conversations"]))
        print(f"\n[ground truth present] {n_gt}/{len(data)}")

        # --- frame counts / durations -----------------------------------
        nf = [len(s["video"]) for s in data]
        nf.sort()
        print(f"[frames per sample] min={nf[0]} p50={nf[len(nf)//2]} max={nf[-1]}")
        fpss = collections.Counter(str(s["metadata"].get("fps")) for s in data)
        print("[fps field]", dict(fpss))

        # --- timestamp consistency check --------------------------------
        print("\n[timestamp consistency: method A (index/fps) vs B (native frames)]")
        bad = 0
        for s in random.Random(0).sample(data, min(300, len(data))):
            ta, tb, native = infer_times(s)
            if abs(ta[-1] - tb[-1]) > 1.5:  # >1.5s disagreement on clip length
                bad += 1
        print(f"    disagreements: {bad}/300"
              + ("   <-- MISMATCH, inspect manually" if bad > 15 else "   OK"))
        s = data[0]
        ta, tb, native = infer_times(s)
        print(f"    example id={s['id']}  native_fps~{native}  "
              f"clipA={ta[-1]:.1f}s clipB={tb[-1]:.1f}s")

        # --- video path prefix ------------------------------------------
        prefixes = collections.Counter("/".join(s["video"][0].split("/")[:3])
                                       for s in data)
        print("\n[video path prefixes]", dict(prefixes))

        # --- region caption info ----------------------------------------
        rc = [s for s in data if s.get("is_RC")]
        print(f"\n[is_RC] {len(rc)} samples")
        if rc:
            print("    RC_info example:", json.dumps(rc[0]["RC_info"]))

        # --- ANSWER FORMATS - the important part ------------------------
        print("\n" + "-" * 78)
        print("ANSWER FORMATS PER TASK  (parsers in src/rewards.py key off these)")
        print("-" * 78)
        by_task = collections.defaultdict(list)
        for smp in data:
            gt = next((m["value"] for m in smp["conversations"]
                       if m["from"] == "gpt"), None)
            if gt:
                by_task[smp["qa_type"]].append((smp["dataset_name"], gt))
        for task in sorted(by_task):
            ex = by_task[task]
            print(f"\n### {task}   (n={len(ex)})")
            lens = sorted(len(g.split()) for _, g in ex)
            print(f"  word count: p10={lens[len(lens)//10]} "
                  f"p50={lens[len(lens)//2]} p90={lens[9*len(lens)//10]}")
            # how many contain a time span / a bbox?
            n_time = sum(1 for _, g in ex if re.search(r"\d+\.?\d*\s*[-–]\s*\d+\.?\d*", g))
            n_box = sum(1 for _, g in ex if re.search(r"\[\s*\d+", g))
            print(f"  contains time-span pattern: {n_time}/{len(ex)}   "
                  f"contains '[num': {n_box}/{len(ex)}")
            for dsname, g in ex[: args.n_examples]:
                print(f"  --[{dsname}]-- {g[:400]}")

        # --- one full sample --------------------------------------------
        print("\n" + "-" * 78)
        print("ONE FULL SAMPLE (truncated frame list)")
        print("-" * 78)
        s = dict(data[0])
        s["video"] = s["video"][:3] + ["...(%d total)" % len(data[0]["video"])]
        s["sampled_video_frames"] = s["sampled_video_frames"][:3] + ["..."]
        print(json.dumps(s, indent=2)[:2500])

    print("\n\nNEXT: if timestamps agree and answer formats match what "
          "src/rewards.py expects, run scripts/01_cache.sh\n")


if __name__ == "__main__":
    main()
