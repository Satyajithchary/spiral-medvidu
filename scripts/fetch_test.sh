#!/usr/bin/env bash
# The test metadata json is almost certainly already on your disk, inside
# testdata.zip (19.4 GB, sitting unextracted at the root). Extracting one ~10 MB
# json out of a zip beats re-downloading anything.
set -euo pipefail
ROOT=${1:-/media/data2/MedVIU_valdata}
ZIP="$ROOT/testdata.zip"

if [ -f "$ZIP" ]; then
  echo "=== json entries inside $ZIP ==="
  unzip -l "$ZIP" | grep -Ei '\.(json|jsonl)$' || echo "  (none found)"
  echo
  echo "=== extracting json only (frames already on disk) ==="
  cd "$ROOT"
  unzip -o -j "$ZIP" '*.json' -d "$ROOT/testdata/" 2>/dev/null || \
  unzip -o "$ZIP" '*/*.json' '*.json' -d "$ROOT/" 2>/dev/null || \
    echo "  no json inside the archive"
  echo
  find "$ROOT/testdata" -maxdepth 2 -name '*.json' -exec ls -la {} \;
fi

if ! find "$ROOT" -maxdepth 3 -name 'cleaned_test_data*.json' | grep -q .; then
  echo
  echo "=== not in the zip; downloading (note: huggingface-cli is now 'hf') ==="
  pip install -q -U huggingface_hub datasets
  cd "$ROOT/testdata"
  hf download UII-AI/MedVidBench cleaned_test_data_11_04.json \
      --repo-type dataset --local-dir . || \
  hf download UII-AI/MedVidBench --repo-type dataset --local-dir . \
      --exclude "testdata.zip" || {
    echo "=== parquet fallback ==="
    python - <<'PY'
from datasets import load_dataset
load_dataset("UII-AI/MedVidBench", split="test").to_parquet("medvidbench_test.parquet")
print("wrote medvidbench_test.parquet -> convert with src/parquet_to_json.py")
PY
  }
fi
echo
echo "Then:  python -m src.paths --root $ROOT --write"
