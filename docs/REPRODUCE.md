# Reproducing the reported numbers

Each table in the paper maps to one command and one file under `results/`. The
JSON files carry the measured values, so any table can be checked without a GPU.

## Without a GPU

```bash
python - <<'PY'
import json
for f in ["leaderboard_test","budget_sweep","indexing","ablation","validation"]:
    d = json.load(open(f"results/{f}.json"))
    print(f"\n=== {f} ===")
    print(json.dumps(d, indent=2)[:900])
PY
```

## With a GPU

Hardware used: one NVIDIA RTX PRO 6000 Blackwell, 96 GB, PyTorch 2.10 with
CUDA 12.8, transformers 5.14. `attn_implementation` is `sdpa`, since flash
attention wheels for sm_120 are unreliable and the speed difference is modest.

| Paper element | Command | Output | Time |
|---|---|---|---|
| Table 1, composition | dataset card | | |
| Table 2, budget sweep | `python -m src.sweep_tokens --task stg --budgets 32 64 128 256 512` | `logs/sweep_stg.json` | 35 min |
| | `python -m src.sweep_tokens --task tal --budgets 32 64 128 256` | | 25 min |
| | `python -m src.sweep_tokens --task skill_assessment --budgets 16 32 64 128` | | 5 min |
| Table 3, indexing | `python -m src.peek --preds <preds> --gt <test> --duplicates` | stdout | seconds |
| Table 4, leaderboard | public MedVidBench leaderboard | | |
| Table 5, validation | `python -m src.infer ... --ctcd --refine` then `python -m src.evaluate ...` | `logs/abl_A_full.json` | 1 h |
| Table 6, per source | same run as Table 5 | | |
| Table 7a, ablation | `python -m src.run_ablations --variants A_full,B_no_tzoom,C_no_szoom,D_no_oed` | `logs/ablation_table.json` | 50 min |
| Table 7a, intervals | `python -m src.paired_test --a <ref> --b <variant> --gt data/val.json` | `logs/paired_*.json` | seconds |
| Table 7b, decoding | `python -m src.multirun --runs 5` | `logs/multirun_summary.json` | 25 min |
| Figure 2 | plotted from `results/budget_sweep.json` | | |

Full training run: three epochs over 7,253 rows, 2,721 optimiser steps,
4 h 08 m, training loss 0.518, validation loss 1.015.

## Verifying a run before trusting it

Three checks catch the failures that cost the most time here.

**Prediction count.** `src.infer` must print `[infer] complete.` with a count at
or below the row count. A larger count means duplicate keys from records written
before the question-hash fix.

**Refinement block.** `[refine] outcomes:` must be non-empty with `changed`
entries. An empty block means nothing was recomputed, which happens when a stale
`refined.jsonl` is present. `src.run_ablations` now wipes variant directories to
prevent this.

**Cache coverage.** `python -m src.check_cache --gt data/val.json` should report
coverage of 1.000 for every source. Anything lower sends frame reads to the
original prefix.

## Known sources of variation

Skill assessment, critical view of safety and next action prediction are decoded
by sampling at temperature 0.8, so single runs vary. Measured standard
deviations over five seeds on the validation split are 0.0151, 0.0040 and 0.0336
respectively. Report means over several seeds for these tasks.

Temporal and spatiotemporal grounding are decoded greedily and reproduce exactly.
