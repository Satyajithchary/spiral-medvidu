"""
Fallback: rebuild the split json from HuggingFace parquet.

This reconstructs the json the rest of the pipeline expects.

    python -m src.parquet_to_json --parquet '/path/to/*.parquet' \
        --out /path/to/MedVIU_valdata/testdata/cleaned_test_data_11_04.json
"""
import argparse, glob, json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="file or glob")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pandas as pd
    files = sorted(glob.glob(a.parquet)) if any(c in a.parquet for c in "*?[") \
        else [a.parquet]
    if not files:
        raise SystemExit(f"no parquet matched {a.parquet}")
    print(f"[parquet] {len(files)} shard(s)")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"[parquet] {len(df)} rows, columns: {list(df.columns)}")

    rows = []
    for r in df.to_dict("records"):
        for k in ("video", "sampled_video_frames", "conversations"):
            v = r.get(k)
            if hasattr(v, "tolist"):
                r[k] = v.tolist()
        conv = r.get("conversations") or []
        r["conversations"] = [dict(c) if not isinstance(c, dict) else c
                              for c in conv]
        md = r.get("metadata")
        if isinstance(md, str):
            try:
                r["metadata"] = json.loads(md)
            except Exception:
                pass
        elif md is not None and not isinstance(md, dict):
            r["metadata"] = dict(md)
        rows.append(r)

    with open(a.out, "w") as f:
        json.dump(rows, f)
    print(f"[parquet] wrote {a.out}  ({len(rows)} rows)")
    print(f"[parquet] example video path: {rows[0]['video'][0]}")
    print("Now re-run:  python -m src.paths --root <root> --write")


if __name__ == "__main__":
    main()
