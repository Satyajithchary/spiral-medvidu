#!/usr/bin/env bash
# OPTIONAL: the task-routing experiment (3 grouped adapters vs 1 shared).
#
# Run this ONLY after the shared model is trained, validated, and a submission
# is banked. Total compute is roughly one extra shared run, because each group
# trains on ~1/3 of the data.
#
#   bash scripts/route_experiment.sh
set -euo pipefail
Y() { python -c "import yaml;d=yaml.safe_load(open('configs/paths.yaml'));print(d.get('$1',''))"; }
CACHE=$(Y ssd_cache); PREFIX=$(Y trainval_src_prefix)
BASE=${BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}

python -m src.groups --train_json data/train.json | tee logs/route_sizes.txt

for G in g c a; do
  echo "=== training routed adapter: $G ==="
  python -m src.train_sft --config configs/sft.yaml \
    --override group=$G output_dir=runs/sft_$G 2>&1 | tee logs/route_sft_$G.txt
done

echo "=== evaluating routed vs shared on the SAME val split ==="
python -m src.infer --test data/val.json --base "$BASE" \
  --adapter_map g=runs/sft_g/final,c=runs/sft_c/final,a=runs/sft_a/final \
  --cache "$CACHE" --src_prefix "$PREFIX" --out preds/val_routed --ctcd --refine
python -m src.evaluate --preds preds/val_routed/raw_predictions.json \
  --gt data/val.json --out logs/val_routed.json | tee logs/val_routed.txt

python - <<'PY'
import json, os
shared = json.load(open("logs/04_val_full.json")) if os.path.exists("logs/04_val_full.json") else {}
routed = json.load(open("logs/val_routed.json"))
cols = ["TAG_mIoU@0.3","TAG_mIoU@0.5","STG_mIoU","DVC_F1","CVS_acc","SA_acc",
        "NAP_acc","VS_llm_proxy","RC_llm_proxy"]
print(f"\n{'metric':16s}{'shared':>10s}{'routed':>10s}{'delta':>10s}")
wins = 0
for c in cols:
    s, r = shared.get(c), routed.get(c)
    if s is None or r is None: continue
    wins += (r > s)
    print(f"{c:16s}{s:10.4f}{r:10.4f}{r-s:+10.4f}")
print(f"\nrouted wins {wins}/{len(cols)} metrics.")
print("If routed loses, that IS the result — report it. Deterministic task")
print("routing is proposed constantly and benchmarked against a shared")
print("baseline almost never.")
PY
