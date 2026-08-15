"""
Write configs/paths.yaml into the training configs, so there is ONE source of
truth for where the data lives.

The previous cascade included a `KeyError: 'src_prefix'` because a shell helper
and a yaml file disagreed about a key name. That class of bug disappears if the
resolver's output is propagated mechanically instead of by hand.

    python -m src.sync_config
"""
import argparse, yaml, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--configs", nargs="+",
                    default=["configs/sft.yaml", "configs/grpo.yaml"])
    a = ap.parse_args()

    p = yaml.safe_load(open(a.paths))
    work = p.get("work_dir") or os.path.join(os.getcwd(), "data")
    updates = {
        "cache_root": p["ssd_cache"],
        "src_prefix": p.get("trainval_src_prefix", "/root/data"),
        "train_json": os.path.join(work, "train.json"),
        "val_json": os.path.join(work, "val.json"),
    }
    for cf in a.configs:
        if not os.path.exists(cf):
            print(f"[sync] skip {cf} (missing)")
            continue
        d = yaml.safe_load(open(cf)) or {}
        changed = {}
        for k, v in updates.items():
            if k in d and d[k] != v:
                changed[k] = (d[k], v)
            if k in d or k in ("cache_root", "src_prefix"):
                d[k] = v
        if "val_json" not in d:
            d.pop("val_json", None)
        with open(cf, "w") as f:
            yaml.safe_dump(d, f, sort_keys=False)
        print(f"[sync] {cf}")
        for k, (old, new) in changed.items():
            print(f"    {k}: {old}  ->  {new}")
        if not changed:
            print("    (already in sync)")


if __name__ == "__main__":
    main()
