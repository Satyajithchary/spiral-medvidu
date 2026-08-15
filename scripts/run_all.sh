#!/usr/bin/env bash
# SPIRAL — end-to-end. Every stage is resumable. Read the README first.
#   bash scripts/run_all.sh
# Paths come from configs/paths.yaml:  python -m src.paths --root /media/data2 --write
set -euo pipefail
mkdir -p logs data preds runs

# tolerate a PARTIAL paths.yaml (test split not yet extracted) instead of
# dying with KeyError halfway through stage 0
Y() { python -c "import yaml;d=yaml.safe_load(open('configs/paths.yaml'));print(d.get('$1',''))"; }

echo "=== 0a. resolve paths ==="
[ -f configs/paths.yaml ] || python -m src.paths --root /media/data2/MedVIU_valdata --write
TRAINVAL=$(Y trainval_json);  TEST=$(Y test_json)
TR_ROOT=$(Y trainval_frame_root); TE_ROOT=$(Y test_frame_root)
# the two splits use DIFFERENT path conventions: trainval is absolute under
# /root/data, test is relative under testdata/. Hence two prefixes.
TR_PREFIX=$(Y trainval_src_prefix); TE_PREFIX=$(Y test_src_prefix)
CACHE=$(Y ssd_cache)
BASE=${BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}
echo "trainval=$TRAINVAL"; echo "test=$TEST"; echo "cache=$CACHE"; echo "base=$BASE"

echo "=== 0a2. sync configs from paths.yaml ==="
python -m src.sync_config

echo "=== 0a3. PREFLIGHT (stops here if the environment is wrong) ==="
python -m src.doctor --stage data

echo "=== 0b. inspect — READ THE OUTPUT ==="
python -m src.inspect_data --train "$TRAINVAL" --test "$TEST" | tee logs/00_inspect.txt

if [ -z "$TEST" ]; then
  echo
  echo "*** test split not resolved — running TRAIN-ONLY stages (0-4). ***"
  echo "*** run scripts/fetch_test.sh in another terminal, then re-run.  ***"
  echo
fi

echo "=== 1. frame cache (~40-70 min) ==="
python -m src.cache_frames --jsons "$TRAINVAL" --src_prefix "$TR_PREFIX" \
  --hdd_root "$TR_ROOT" --out "$CACHE" --workers 16 --max_side 448
if [ -n "$TEST" ]; then
python -m src.cache_frames --jsons "$TEST" --src_prefix "$TE_PREFIX" \
  --hdd_root "$TE_ROOT" --out "$CACHE" --workers 16 --max_side 448
fi

echo "=== 2. C1 harvesting + C3 priors + video-level split ==="
python -m src.ontology --audit "$TRAINVAL" | tee logs/02_ontology_audit.txt
python -m src.timebase --trainval "$TRAINVAL" | tee logs/02_timebase.txt
python -m src.prep_data --trainval "$TRAINVAL" --out data --cache "$CACHE" \
  --val_frac 0.12 --aux_ratio 0.35 --ctcd_train 1 | tee logs/02_prep.txt
python -m src.balance --train_json data/train.json --tau 0.5 | tee logs/02_balance.txt

python -m src.doctor --stage sft

echo "=== 3. SFT with C2 balanced sampling (~10-13h 8B / ~6-7h 4B) ==="
python -m src.train_sft --config configs/sft.yaml 2>&1 | tee logs/03_sft.txt

echo "=== 4. validate on held-out videos (full SPIRAL) ==="
python -m src.infer --test data/val.json --base "$BASE" --adapter runs/sft/final \
  --cache "$CACHE" --src_prefix "$TR_PREFIX" --out preds/val_full --ctcd --refine
python -m src.evaluate --preds preds/val_full/raw_predictions.json \
  --gt data/val.json --out logs/04_val_full.json | tee logs/04_val_full.txt

echo "=== 4b. TIME BASE CALIBRATION ==="
python -m src.timebase --trainval "$TRAINVAL" | tee logs/04b_timebase.txt

echo "=== 5. BANK A SUBMISSION NOW ==="
if [ -z "$TEST" ]; then
  echo "SKIPPED — no test json yet. Run scripts/fetch_test.sh, then:"
  echo "  python -m src.paths --root /media/data2/MedVIU_valdata --write"
  echo "  bash scripts/run_all.sh"
  exit 0
fi
python -m src.infer --test "$TEST" --base "$BASE" --adapter runs/sft/final \
  --cache "$CACHE" --src_prefix "$TE_PREFIX" --out preds/test_sft --ctcd --refine
python -m src.postprocess --preds preds/test_sft/raw_predictions.json \
  --test "$TEST" --out submission_sft.json
echo ">>> UPLOAD submission_sft.json, run LLM-Judge step 2, THEN continue <<<"

echo "=== 6. ablations for the paper (cheap: reuses pass1.jsonl) ==="
bash scripts/ablate.sh 2>&1 | tee logs/06_ablations.txt

echo "=== 7. GRPO — only if >8h remain ==="
python -m src.train_grpo --config configs/grpo.yaml 2>&1 | tee logs/07_grpo.txt
python -m src.infer --test data/val.json --base "$BASE" --adapter runs/grpo/final \
  --cache "$CACHE" --src_prefix "$TR_PREFIX" --out preds/val_rl --ctcd --refine
python -m src.evaluate --preds preds/val_rl/raw_predictions.json \
  --gt data/val.json --out logs/07_val_rl.json | tee logs/07_val_rl.txt
echo ">>> compare 04_val_full.json vs 07_val_rl.json and submit the WINNER <<<"
