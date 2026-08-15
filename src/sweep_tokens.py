"""
THE HEADLINE EXPERIMENT — output-budget sweep.

Your strongest finding is that a large part of reported VideoLLM failure on
structured medical video tasks is output TRUNCATION, not perception. You have
two points (STG 0.014 at 32 tokens, 0.094 at 512). A curve is a result; two
points are an anecdote.

This sweeps max_new_tokens for one task on the val split and prints a table you
can plot directly. Runs on the val split (GT available) so no leaderboard round
trip is needed.

    python -m src.sweep_tokens --task stg  --budgets 32 64 128 256 512
    python -m src.sweep_tokens --task tal  --budgets 32 64 128 256
    python -m src.sweep_tokens --task skill_assessment --budgets 16 32 64 128

Cost: (#val rows for that task) x (#budgets) generations. STG is 205 val rows,
5 budgets, ~2 s each => ~35 min. TAL 214 rows x 4 => ~25 min.

Why it matters for the paper: ranks 13-16 on the public leaderboard sit at
STG 0.003 and ranks 25-29 at TAG 0.074. Those are the same failure band. If the
curve is steep in the region where everyone's defaults live, the claim
generalises beyond your own run.
"""
from __future__ import annotations
import argparse, json, os, time, torch
from transformers import AutoProcessor
from peft import PeftModel

from .dataset import MedVidDataset, make_collator
from .sampling import normalize_task
from .prompts import get_answer
from .rewards import raw_metric, format_ok
from .train_sft import load_model
from .formats import qhash, parse_tboxes, parse_spans, parse_osats


def n_units(pred, task):
    """How many output units the model actually emitted (boxes / spans / dims)."""
    if task == "stg":
        return len(parse_tboxes(pred))
    if task == "tal":
        return len(parse_spans(pred))
    if task == "skill_assessment":
        return len(parse_osats(pred))
    return len((pred or "").split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--budgets", type=int, nargs="+",
                    default=[32, 64, 128, 256, 512])
    ap.add_argument("--gt", default="data/val.json")
    ap.add_argument("--base", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--adapter", default="runs/sft/final")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--src_prefix", default="/root/data")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from .paths import cfg
    a.cache = a.cache or cfg("ssd_cache")

    processor = AutoProcessor.from_pretrained(
        a.base, min_pixels=64 * 28 * 28, max_pixels=160 * 28 * 28)
    tok = processor.tokenizer
    model = load_model({"model": a.base, "attn": "sdpa"})
    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    model.eval().cuda()
    model.config.use_cache = True

    ds = MedVidDataset(a.gt, processor, src_prefix=a.src_prefix,
                       cache_root=a.cache, train=False)
    ds.orig_root = cfg("trainval_frame_root")
    rows = [r for r in ds.rows if normalize_task(r["qa_type"]) == a.task]
    if a.limit:
        rows = rows[: a.limit]
    print(f"[sweep] task={a.task}  rows={len(rows)}  budgets={a.budgets}")
    collate = make_collator(processor, 8192, train=False)

    results = []
    for B in a.budgets:
        t0, scores, units, fmt_ok, gt_units = time.time(), [], [], 0, []
        for i, row in enumerate(rows):
            gt = get_answer(row) or ""
            try:
                msgs, _ = ds.build_messages(row)
                batch = collate([{"messages": msgs, "answer": "", "row": row}])
                batch.pop("_rows", None)
                batch = {k: (v.cuda() if torch.is_tensor(v) else v)
                         for k, v in batch.items()}
                plen = batch["input_ids"].shape[1]
                with torch.no_grad():
                    g = model.generate(**batch, do_sample=False, max_new_tokens=B,
                                       pad_token_id=tok.pad_token_id)
                pred = tok.decode(g[0, plen:], skip_special_tokens=True).strip()
            except Exception as e:
                print(f"  fail {i}: {type(e).__name__}")
                pred = ""
            scores.append(raw_metric(pred, gt, row["qa_type"]))
            units.append(n_units(pred, a.task))
            gt_units.append(n_units(gt, a.task))
            fmt_ok += int(format_ok(pred, row["qa_type"]))
            if (i + 1) % 50 == 0:
                print(f"  B={B}  {i+1}/{len(rows)}  "
                      f"{(time.time()-t0)/(i+1):.2f}s/row", flush=True)
        m = sum(scores) / max(1, len(scores))
        u = sum(units) / max(1, len(units))
        gu = sum(gt_units) / max(1, len(gt_units))
        cov = sum(min(1.0, a_ / max(1e-9, b_)) for a_, b_ in zip(units, gt_units))
        cov /= max(1, len(units))
        results.append({"budget": B, "metric": m, "units_emitted": u,
                        "units_in_gt": gu, "coverage": cov,
                        "format_ok": fmt_ok / max(1, len(rows)),
                        "secs": time.time() - t0})
        print(f"[sweep] B={B:4d}  metric={m:.4f}  units {u:.2f}/{gu:.2f}  "
              f"coverage={cov:.3f}  fmt_ok={fmt_ok}/{len(rows)}  "
              f"{time.time()-t0:.0f}s")

    print("\n" + "=" * 74)
    print(f"OUTPUT BUDGET SWEEP — {a.task}  (n={len(rows)}, val split)")
    print("=" * 74)
    print(f"{'max_new':>8s} {'metric':>9s} {'units out':>10s} {'units gt':>9s} "
          f"{'coverage':>9s} {'fmt ok':>8s}")
    for r in results:
        print(f"{r['budget']:8d} {r['metric']:9.4f} {r['units_emitted']:10.2f} "
              f"{r['units_in_gt']:9.2f} {r['coverage']:9.3f} {r['format_ok']:8.2f}")
    print("\nNote: 'fmt ok' stays high while the metric collapses — a truncated")
    print("answer parses cleanly, so format validation cannot detect this.")

    out = a.out or f"logs/sweep_{a.task}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump({"task": a.task, "n_rows": len(rows), "results": results},
              open(out, "w"), indent=2)
    print(f"\n[sweep] -> {out}")


if __name__ == "__main__":
    main()
