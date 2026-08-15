"""
STEP 4/6 - Inference with Cross-Task Consistency Decoding (CTCD).

Contribution #2.

The test set asks MULTIPLE questions about the SAME video. Every published
baseline answers each row in isolation. But if the model has already localised
"cutting: 31.0-44.0s" for a video in a TAL row, that is a strong prior when the
same video's dense-captioning row asks for segments - and it costs nothing but a
second forward pass.

Pass 1: answer all grounding + short-form rows (tal, stg, next_action, cvs, sa).
        Build a per-video timeline from the model's OWN predictions.
Pass 2: answer the captioning rows (dvc, vs, rc) with that self-derived timeline
        injected into the system prompt.

No ground truth is used anywhere. 

Resumable: writes JSONL incrementally, skips ids already present.

    python -m src.infer --test cleaned_test_data_11_04.json \
        --base Qwen/Qwen3-VL-8B-Instruct --adapter runs/grpo/final \
        --cache /mnt/ssd/medvidu_frames --out preds/test --ctcd
"""
from __future__ import annotations
import argparse, json, os, sys, time, collections, torch
from transformers import AutoProcessor
from peft import PeftModel

from .dataset import MedVidDataset, make_collator
from .sampling import normalize_task
from .rewards import parse_spans, parse_events, parse_box
from .train_sft import load_model
from .groups import group_of, parse_adapter_map

PASS1 = {"tal", "stg", "next_action", "cvs_assessment", "skill_assessment"}
PASS2 = {"dense_captioning", "video_summary", "region_caption"}

# Token budgets, derived from the ACTUAL ground-truth length distribution
# reported by inspect_data (word-count p90 per task), not from the paper's
# figures. Getting these wrong is silent and catastrophic: a truncated answer
# parses fine, so format_ok() passes and the metric just reads low.
#
#   task              GT p90 words   ~tokens   old   new
#   stg               48 (numeric)     ~300     32   512   <- 6-8 boxes @ ~25 tok
#   dense_captioning  352              ~550    384   900
#   video_summary     88               ~150    220   384
#   tal               5, up to 10 spans ~120     64   256
#   skill_assessment  23 (six "4/5")   ~70      16   128   <- always truncated
#   cvs_assessment    9                ~25      48    96
#   region_caption    28               ~55      96   160
#   next_action       5                ~12      32    48
MAX_NEW = {"tal": 256, "stg": 512, "next_action": 48, "cvs_assessment": 96,
           "skill_assessment": 128, "dense_captioning": 512,
           "video_summary": 320, "region_caption": 160}


from .formats import qhash, rowkey


def rkey(row_or_rec):
    return rowkey(row_or_rec)


def load_done(path):
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[rkey(r)] = r
                except Exception:
                    pass
    return done


def build_ctcd(records, test_rows) -> dict[str, str]:
    """video_id -> context string, from the model's own pass-1 outputs."""
    by_key = {rkey(r): r for r in test_rows}
    per_video = collections.defaultdict(lambda: {"spans": [], "acts": [],
                                                 "boxes": [], "next": None,
                                                 "skill": None})
    for k, rec in records.items():
        row = by_key.get(k)
        if row is None:
            continue
        vid = row["metadata"]["video_id"]
        t = normalize_task(row["qa_type"])
        pred = rec["prediction"]
        q = next((m["value"] for m in row["conversations"]
                  if m["from"] == "human"), "")
        if t == "tal":
            # recover which action the question was about
            act = None
            for cand in ("cutting", "tying", "suturing", "dissect", "clip",
                         "disinfect", "handwashing", "puncture", "observe",
                         "prepare", "release"):
                if cand in q.lower():
                    act = cand
                    break
            for s, e in parse_spans(pred)[:4]:
                per_video[vid]["spans"].append((s, e, act or "action"))
        elif t == "stg":
            b = parse_box(pred)
            if b:
                per_video[vid]["boxes"].append(b)
        elif t == "next_action":
            per_video[vid]["next"] = pred.strip()[:80]
        elif t == "skill_assessment":
            per_video[vid]["skill"] = pred.strip()[:40]

    ctx = {}
    for vid, d in per_video.items():
        bits = []
        if d["spans"]:
            sp = sorted(set(d["spans"]))[:8]
            bits.append("Action timeline you previously derived for this video: "
                        + "; ".join(f"{a} {s:.1f}-{e:.1f}s" for s, e, a in sp))
        if d["boxes"]:
            b = d["boxes"][0]
            bits.append(f"An instrument was localised at box "
                        f"[{int(b[0])},{int(b[1])},{int(b[2])},{int(b[3])}].")
        if d["next"]:
            bits.append(f"Predicted next step: {d['next']}")
        if d["skill"]:
            bits.append(f"Assessed operator skill: {d['skill']}")
        if bits:
            ctx[vid] = "\n".join(bits)
    return ctx


@torch.no_grad()
def refine_pass(rows, coarse, ds, model, processor, collate, out_path, cfg):
    """C4 + C5. Runs AFTER pass 1 and BEFORE CTCD, so the timeline that gets
    injected into the captioning prompts is the refined one - the refinements
    compound rather than sitting in separate silos."""
    from .refine import refine, REFINE_TASKS
    done = load_done(out_path)
    todo = [r for r in rows
            if normalize_task(r["qa_type"]) in REFINE_TASKS
            and rkey(r) not in done and rkey(r) in coarse]
    print(f"[refine] {len(todo)} rows to refine ({len(done)} already done)")
    f = open(out_path, "a")
    stats, t0 = collections.Counter(), time.time()
    for n, row in enumerate(todo):
        base = coarse[rkey(row)]["prediction"]
        try:
            new, info = refine(model, processor, collate, ds, row, base, cfg)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); new, info = base, {"reason": "oom"}
        except Exception as e:
            new, info = base, {"reason": f"{type(e).__name__}"}
        t = normalize_task(row["qa_type"])
        stats[f"{t}:{'changed' if new != base else info.get('reason','same')}"] += 1
        rec = {"id": row["id"], "qa_type": row["qa_type"],
               "qhash": qhash(row), "prediction": new,
               "coarse": base, "info": info}
        f.write(json.dumps(rec) + "\n"); f.flush()
        done[rkey(row)] = rec
        if (n + 1) % 50 == 0:
            el = time.time() - t0
            print(f"[refine] {n+1}/{len(todo)} {el/(n+1):.2f}s/row "
                  f"eta {(len(todo)-n-1)*el/(n+1)/60:.0f}min", flush=True)
    f.close()
    print("[refine] outcomes:")
    for k, v in stats.most_common():
        print(f"    {k:44s} {v}")
    merged = dict(coarse)
    for k, v in done.items():
        merged[k] = {"id": v["id"], "qa_type": v["qa_type"],
                     "qhash": v.get("qhash", k[2]),
                     "prediction": v["prediction"]}
    return merged


@torch.no_grad()
def run_pass(rows, ds, model, processor, collate, out_path, ctcd_ctx=None,
             batch_size=1, tag=""):
    done = load_done(out_path)
    todo = [r for r in rows if rkey(r) not in done]
    print(f"[infer{tag}] {len(todo)} to do ({len(done)} already done)")
    tok = processor.tokenizer
    f = open(out_path, "a")
    t0 = time.time()
    amap = globals().get("_ADAPTER_MAP") or {}
    for n, row in enumerate(todo):
        t = normalize_task(row["qa_type"])
        if amap:
            g = group_of(row["qa_type"])
            if g in amap:
                model.set_adapter(g)
        ctx = (ctcd_ctx or {}).get(row["metadata"]["video_id"])
        try:
            msgs, _ = ds.build_messages(row, ctcd=ctx)
            batch = collate([{"messages": msgs, "answer": "", "row": row}])
            batch.pop("_rows", None)
            batch = {k: (v.cuda() if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            plen = batch["input_ids"].shape[1]
            gen = model.generate(**batch, do_sample=False, num_beams=1,
                                 max_new_tokens=MAX_NEW.get(t, 128),
                                 pad_token_id=tok.pad_token_id)
            pred = tok.decode(gen[0, plen:], skip_special_tokens=True).strip()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  OOM on {row['id']} -> retry at half frames")
            try:
                ds.budget_scale *= 0.5
                msgs, _ = ds.build_messages(row, ctcd=ctx)
                batch = collate([{"messages": msgs, "answer": "", "row": row}])
                batch.pop("_rows", None)
                batch = {k: (v.cuda() if torch.is_tensor(v) else v)
                         for k, v in batch.items()}
                plen = batch["input_ids"].shape[1]
                gen = model.generate(**batch, do_sample=False,
                                     max_new_tokens=MAX_NEW.get(t, 128),
                                     pad_token_id=tok.pad_token_id)
                pred = tok.decode(gen[0, plen:], skip_special_tokens=True).strip()
            finally:
                ds.budget_scale *= 2.0
        except Exception as e:
            n_fail += 1
            if n_fail <= 8:
                print(f"  FAIL {row['id']}: {type(e).__name__}: {e}")
            elif n_fail == 9:
                print("  ... further failures suppressed, count reported at end")
            pred = ""
        rec = {"id": row["id"], "qa_type": row["qa_type"],
               "qhash": qhash(row), "prediction": pred}
        f.write(json.dumps(rec) + "\n")
        f.flush()
        done[rkey(row)] = rec
        if (n + 1) % 25 == 0:
            el = time.time() - t0
            print(f"[infer{tag}] {n+1}/{len(todo)}  {el/(n+1):.2f}s/sample  "
                  f"eta {(len(todo)-n-1)*el/(n+1)/60:.0f}min", flush=True)
    f.close()
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--adapter_map", default=None,
                    help="routed adapters, e.g. "
                         "g=runs/sft_g/final,c=runs/sft_c/final,a=runs/sft_a/final")
    ap.add_argument("--cache", default=None,
                    help="defaults to ssd_cache in configs/paths.yaml")
    ap.add_argument("--src_prefix", default=None,
                    help="defaults to the matching *_src_prefix in paths.yaml")
    ap.add_argument("--orig_root", default=None,
                    help="original (non-downscaled) frame root; spatial zoom "
                         "needs full resolution. Defaults from paths.yaml.")
    ap.add_argument("--out", default="preds/test")
    ap.add_argument("--ctcd", action="store_true",
                    help="C3/CTCD: condition captioning on the self-derived timeline")
    ap.add_argument("--refine", action="store_true",
                    help="C4+C5: temporal zoom, spatial zoom, ordinal expectation")
    ap.add_argument("--no_temporal_zoom", action="store_true")
    ap.add_argument("--no_spatial_zoom", action="store_true")
    ap.add_argument("--no_oed", action="store_true")
    ap.add_argument("--oed_k", type=int, default=5)
    ap.add_argument("--zoom_frames", type=int, default=48)
    ap.add_argument("--zoom_margin", type=float, default=0.6)
    ap.add_argument("--zoom_expand", type=float, default=2.2)
    ap.add_argument("--budget_scale", type=float, default=1.0)
    ap.add_argument("--max_frames", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--max_pixels", type=int, default=160 * 28 * 28)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N rows")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed for the stochastic decoding of C5; varying it "
                         "across runs yields the multi-run intervals required "
                         "for tasks with non-deterministic decoding")
    ap.add_argument("--only_task", default=None,
                    help="comma list of normalised tasks (tal,stg,cvs_assessment,"
                         "skill_assessment,next_action,dense_captioning,"
                         "video_summary,region_caption). Lets you re-run ONE task "
                         "for an ablation and splice it into an existing "
                         "submission via postprocess --base.")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.seed is not None:
        import random as _r, numpy as _np
        torch.manual_seed(a.seed); _r.seed(a.seed); _np.random.seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)
        print(f"[infer] seed = {a.seed}")
    if a.adapter and not os.path.isdir(a.adapter):
        sys.exit(f"\n[infer] --adapter '{a.adapter}' is not a directory.\n"
                 f"  HuggingFace would read it as a repo id and fail confusingly.\n"
                 f"  Train first, or drop --adapter to run the base model.\n")
    for g, pth in parse_adapter_map(a.adapter_map).items():
        if not os.path.isdir(pth):
            sys.exit(f"\n[infer] routed adapter '{g}' -> '{pth}' is not a directory.\n")
    if not os.path.exists(a.test):
        sys.exit(f"\n[infer] --test '{a.test}' does not exist.\n")

    # Resolve everything else from configs/paths.yaml so no command line needs a
    # placeholder. Literal <CACHE> tokens are read by the shell as redirects.
    from .paths import cfg as _cfg
    _tj = _cfg("test_json") or ""
    _is_test = bool(_tj) and os.path.abspath(a.test) == os.path.abspath(_tj)
    if not a.cache:
        a.cache = _cfg("ssd_cache")
    if not a.src_prefix:
        a.src_prefix = _cfg("test_src_prefix" if _is_test
                            else "trainval_src_prefix", "/root/data")
    for _k, _v in (("--cache", a.cache), ("--src_prefix", a.src_prefix)):
        if _v is None or str(_v).startswith("<"):
            sys.exit(f"\n[infer] {_k} is unset or a literal placeholder "
                     f"({_v!r}).\n  Run: python -m src.paths --root "
                     f"/media/data2/MedVIU_valdata --write\n")
    print(f"[infer] split={'test' if _is_test else 'trainval'}  "
          f"cache={a.cache}  src_prefix='{a.src_prefix}'")

    src = a.adapter or a.base
    processor = AutoProcessor.from_pretrained(
        src if os.path.exists(os.path.join(str(src), "preprocessor_config.json"))
        else a.base,
        min_pixels=64 * 28 * 28, max_pixels=a.max_pixels)
    model = load_model({"model": a.base, "attn": "sdpa"})
    amap = parse_adapter_map(a.adapter_map)
    if amap:
        # keep all three resident and flip the active one per row; cheaper and
        # far less error-prone than merging or reloading between tasks
        first = None
        for g, path in amap.items():
            if first is None:
                model = PeftModel.from_pretrained(model, path, adapter_name=g)
                first = g
            else:
                model.load_adapter(path, adapter_name=g)
        model.set_adapter(first)
        print(f"[infer] ROUTED adapters loaded: {list(amap)}")
    elif a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter)
        model = model.merge_and_unload()
        print("[infer] merged adapter:", a.adapter)
    model.eval().cuda()
    globals()["_ADAPTER_MAP"] = amap
    globals()["_MODEL"] = model
    model.config.use_cache = True

    ds = MedVidDataset(a.test, processor, src_prefix=a.src_prefix,
                       cache_root=a.cache, budget_scale=a.budget_scale,
                       max_frames=a.max_frames, max_len=a.max_len, train=False)
    if a.orig_root:
        ds.orig_root = a.orig_root
    else:
        ds.orig_root = _cfg("test_frame_root") if _is_test else _cfg("trainval_frame_root")
    print(f"[infer] orig_root (spatial zoom, full-res) = {ds.orig_root}")
    rows = ds.rows[: a.limit] if a.limit else ds.rows
    if a.only_task:
        keep = {t.strip() for t in a.only_task.split(",")}
        rows = [r for r in rows if normalize_task(r["qa_type"]) in keep]
        print(f"[infer] --only_task {sorted(keep)} -> {len(rows)} rows")
    collate = make_collator(processor, a.max_len, train=False)

    p1 = [r for r in rows if normalize_task(r["qa_type"]) in PASS1]
    p2 = [r for r in rows if normalize_task(r["qa_type"]) in PASS2]
    print(f"[infer] pass1={len(p1)}  pass2={len(p2)}  total={len(rows)}")

    recs = run_pass(p1, ds, model, processor, collate,
                    os.path.join(a.out, "pass1.jsonl"), None, tag=" p1")

    if a.refine:
        rcfg = {"temporal_zoom": not a.no_temporal_zoom,
                "spatial_zoom": not a.no_spatial_zoom,
                "oed": not a.no_oed, "oed_k": a.oed_k,
                "zoom_frames": a.zoom_frames, "zoom_margin": a.zoom_margin,
                "zoom_expand": a.zoom_expand}
        recs = refine_pass(p1, recs, ds, model, processor, collate,
                           os.path.join(a.out, "refined.jsonl"), rcfg)

    ctx = build_ctcd(recs, rows) if a.ctcd else {}
    if a.ctcd:
        print(f"[infer] CTCD context built for {len(ctx)} videos "
              f"({100*len(ctx)/max(1,len(set(r['metadata']['video_id'] for r in rows))):.0f}% "
              f"of test videos)")
        json.dump(ctx, open(os.path.join(a.out, "ctcd_context.json"), "w"), indent=2)

    recs2 = run_pass(p2, ds, model, processor, collate,
                     os.path.join(a.out, "pass2.jsonl"), ctx, tag=" p2")

    allrec = {**recs, **recs2}
    with open(os.path.join(a.out, "raw_predictions.json"), "w") as f:
        json.dump(list(allrec.values()), f, indent=2)
    n_need = len({rkey(r) for r in rows})
    if len(allrec) > len(rows) * 1.05:
        print(f"\n  !! {len(allrec)} predictions for {len(rows)} rows. Duplicate "
              f"keys are present, which means some jsonl records predate the\n"
              f"  !! question-hash fix. Delete {a.out} and re-run, or migrate "
              f"with src.migrate_keys. Results from this file are NOT valid.\n")
    print(f"[infer] {len(allrec)}/{n_need} unique (id, qa_type) predictions -> "
          f"{a.out}/raw_predictions.json")
    if len(allrec) < n_need:
        miss = collections.Counter(
            normalize_task(r["qa_type"]) for r in rows if rkey(r) not in allrec)
        print(f"  !! {n_need - len(allrec)} MISSING: {dict(miss)}")
        print("  !! re-run this exact command - it resumes from the jsonl files.")
    else:
        print("[infer] complete.")


if __name__ == "__main__":
    main()
