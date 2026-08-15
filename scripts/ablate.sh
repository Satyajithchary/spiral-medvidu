#!/usr/bin/env bash

set -euo pipefail
Y() { python -c "import yaml;d=yaml.safe_load(open('configs/paths.yaml'));print(d.get('$1',''))"; }
CACHE=$(Y ssd_cache); PREFIX=$(Y trainval_src_prefix)
BASE=${BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}
ADP=${ADAPTER:-runs/sft/final}
mkdir -p preds logs

run () {  # $1 = name, rest = extra infer flags
  local name=$1; shift
  mkdir -p "preds/abl_$name"
  # seed the cache from the full run so pass 1 is never repeated
  [ -f preds/val_full/pass1.jsonl ] && cp -n preds/val_full/pass1.jsonl \
      "preds/abl_$name/pass1.jsonl" 2>/dev/null || true
  python -m src.infer --test data/val.json --base "$BASE" --adapter "$ADP" \
    --cache "$CACHE" --src_prefix "$PREFIX" --out "preds/abl_$name" "$@"
  python -m src.evaluate --preds "preds/abl_$name/raw_predictions.json" \
    --gt data/val.json --out "logs/abl_$name.json" | tee "logs/abl_$name.txt"
}

echo "### A  full SPIRAL"                    ; run full        --ctcd --refine
echo "### B  - C4a temporal zoom"            ; run no_tzoom    --ctcd --refine --no_temporal_zoom
echo "### C  - C4b spatial zoom"             ; run no_szoom    --ctcd --refine --no_spatial_zoom
echo "### D  - C5 ordinal expectation"       ; run no_oed      --ctcd --refine --no_oed
echo "### E  - CTCD (captioning unconditioned)" ; run no_ctcd  --refine
echo "### F  - all refinement (SFT + CTCD only)"; run sft_ctcd  --ctcd
echo "### G  plain SFT, no SPIRAL inference" ; run plain

echo
echo "Two more rows need a RETRAIN, so only do them if you have spare GPU time:"
echo "  H  - C1 harvesting :  prep_data --aux_ratio 0   -> retrain -> run full"
echo "  I  - C2 balancing  :  sft.yaml balance_tau: 0   -> retrain -> run full"
echo "  J  - C3 conditioning: prep_data --ctcd_train 0  -> retrain -> run full"
echo
python - <<'PY'
import json, glob, os
rows = {}
for f in sorted(glob.glob("logs/abl_*.json")):
    rows[os.path.basename(f)[4:-5]] = json.load(open(f))
if not rows: raise SystemExit
cols = ["TAG_mIoU@0.3","TAG_mIoU@0.5","STG_mIoU","DVC_F1","CVS_acc","SA_acc",
        "NAP_acc","VS_llm_proxy","RC_llm_proxy"]
print(f"{'variant':14s}" + "".join(f"{c:>15s}" for c in cols))
for k, v in rows.items():
    print(f"{k:14s}" + "".join(f"{v.get(c, float('nan')):15.4f}" for c in cols))
PY
