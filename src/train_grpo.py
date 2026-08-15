"""
STEP 5 (optional but high-value) - Stage-2 GRPO on verifiable tasks.

Self-contained: no verl, no EasyR1, no TRL.

Scope (deliberately narrow - this is where the headroom is and where rewards are
cheap and exact):
    tal, stg, next_action, cvs_assessment, skill_assessment
Captioning is NOT in the RL loop. Its reward needs a judge, and Stage-1 SFT plus
CTCD at inference gets most of that value for none of the cost.

Key details:
  - LoRA-only updates, ViT frozen, no reference model, no KL term (DAPO-style).
  - Asymmetric clipping eps_low=0.2, eps_high=0.28.
  - Group G=8, one prompt per step, grad accumulation over prompts.
  - Groups with zero reward variance are skipped (no gradient signal, wasted compute).
  - Frame budget halved vs SFT to fit 8 rollouts of KV cache.

    python -m src.train_grpo --config configs/grpo.yaml
"""
from __future__ import annotations
import argparse, json, os, random, time, yaml, torch, numpy as np
import torch.nn.functional as F
from transformers import AutoProcessor, set_seed
from peft import PeftModel

from .dataset import MedVidDataset, make_collator
from .prompts import get_answer
from .rewards import grpo_reward, RewardNormalizer
from .sampling import normalize_task
from .train_sft import load_model, freeze_vision

def _require_dir(path, what, howto):
    import os, sys
    if not os.path.isdir(path):
        sys.exit(f"\n[{what}] '{path}' is not a directory.\n"
                 f"  HuggingFace will otherwise try to read it as a repo id and\n"
                 f"  fail with a confusing 'Repo id must be in the form' error.\n"
                 f"  {howto}\n")


RL_TASKS = {"tal", "stg", "next_action", "cvs_assessment", "skill_assessment"}
MAX_NEW = {"tal": 256, "stg": 512, "next_action": 48,
           "cvs_assessment": 96, "skill_assessment": 160}


def token_logps(model, input_ids, attention_mask, extra, completion_len):
    """Log-probs of the last `completion_len` tokens."""
    out = model(input_ids=input_ids, attention_mask=attention_mask, **extra)
    logits = out.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    lp = torch.log_softmax(logits.float(), dim=-1)
    tok_lp = lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return tok_lp[:, -completion_len:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    print(json.dumps(cfg, indent=2))
    set_seed(cfg.get("seed", 0))
    dev = "cuda"
    _require_dir(cfg["sft_ckpt"], "grpo",
                 "Train stage 1 first:  python -m src.train_sft --config configs/sft.yaml")

    processor = AutoProcessor.from_pretrained(
        cfg["sft_ckpt"], min_pixels=cfg.get("min_pixels", 64 * 28 * 28),
        max_pixels=cfg.get("max_pixels", 128 * 28 * 28))
    tok = processor.tokenizer

    base = load_model({"model": cfg["base_model"], "attn": cfg.get("attn", "sdpa")})
    model = PeftModel.from_pretrained(base, cfg["sft_ckpt"], is_trainable=True)
    model.to(dev)
    freeze_vision(model)
    model.config.use_cache = True
    if cfg.get("grad_ckpt", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    ds = MedVidDataset(cfg["train_json"], processor,
                       src_prefix=cfg.get("src_prefix", "/root/data"),
                       cache_root=cfg.get("cache_root"),
                       budget_scale=cfg.get("budget_scale", 0.5),
                       max_frames=cfg.get("max_frames", 32),
                       max_len=cfg.get("max_len", 6144), train=False)
    pool = [i for i, r in enumerate(ds.rows)
            if normalize_task(r["qa_type"]) in RL_TASKS and get_answer(r)]
    print(f"[grpo] RL pool: {len(pool)} / {len(ds.rows)} rows")

    embedder = None
    if cfg.get("embedder"):
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(cfg["embedder"], device=dev)

    normalizer = RewardNormalizer(k=cfg.get("logistic_k", 3.0))
    if cfg.get("norm_init") and os.path.exists(cfg["norm_init"]):
        normalizer.load(json.load(open(cfg["norm_init"])))
        print("[grpo] loaded warm normaliser stats")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.get("lr", 1e-6), weight_decay=0.0)
    G = cfg.get("group_size", 8)
    eps_lo, eps_hi = cfg.get("eps_low", 0.2), cfg.get("eps_high", 0.28)
    accum = cfg.get("grad_accum", 4)
    steps = cfg.get("max_steps", 600)
    rng = random.Random(cfg.get("seed", 0))
    collate = make_collator(processor, cfg.get("max_len", 6144), train=False)

    log, t0 = [], time.time()
    opt.zero_grad(set_to_none=True)
    for step in range(steps):
        row = ds.rows[rng.choice(pool)]
        task = normalize_task(row["qa_type"])
        item = ds[ds.rows.index(row)] if False else None  # avoid O(n) lookup
        # build directly
        msgs, _ = ds.build_messages(row)
        batch = collate([{"messages": msgs, "answer": get_answer(row), "row": row}])
        batch.pop("_rows", None)
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}

        prompt_len = batch["input_ids"].shape[1]
        max_new = MAX_NEW.get(task, 48)

        # ---------------- rollout ----------------
        model.eval()
        with torch.no_grad():
            gen = model.generate(**batch, do_sample=True,
                                 temperature=cfg.get("temperature", 1.0),
                                 top_p=cfg.get("top_p", 0.95),
                                 max_new_tokens=max_new,
                                 num_return_sequences=G,
                                 pad_token_id=tok.pad_token_id)
        comp_ids = gen[:, prompt_len:]
        texts = tok.batch_decode(comp_ids, skip_special_tokens=True)
        gt = get_answer(row)
        rr = [grpo_reward(t, gt, row, normalizer, embedder) for t in texts]
        rewards = torch.tensor([x[0] for x in rr], device=dev, dtype=torch.float32)
        raws = [x[1] for x in rr]

        if rewards.std() < 1e-4:
            if step % 20 == 0:
                print(f"[grpo] step {step} skipped (zero variance, "
                      f"r={rewards.mean():.3f}, task={task})")
            continue
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

        # ---------------- policy gradient ----------------
        model.train()
        extra = {k: v for k, v in batch.items()
                 if k not in ("input_ids", "attention_mask")}
        # expand visual tensors to G
        exp = {}
        for k, v in extra.items():
            if torch.is_tensor(v):
                exp[k] = v.repeat(G, *([1] * (v.dim() - 1))) if v.dim() > 0 and \
                    v.shape[0] == 1 else v
            else:
                exp[k] = v
        attn = torch.ones_like(gen)
        attn[gen == tok.pad_token_id] = 0
        attn[:, :prompt_len] = batch["attention_mask"].repeat(G, 1)

        micro = cfg.get("micro_bs", 2)
        loss_acc = 0.0
        for s in range(0, G, micro):
            sl = slice(s, min(s + micro, G))
            sub_extra = {k: (v[sl] if torch.is_tensor(v) and v.dim() > 0
                             and v.shape[0] == G else v) for k, v in exp.items()}
            lp = token_logps(model, gen[sl], attn[sl], sub_extra, comp_ids.shape[1])
            mask = (comp_ids[sl] != tok.pad_token_id).float()
            # single inner epoch => ratio == 1; clipping is a no-op but kept so the
            # objective is literally DAPO's if you raise inner_epochs.
            ratio = torch.ones_like(lp)
            a_ = adv[sl].unsqueeze(1)
            unclipped = ratio * a_
            clipped = torch.clamp(ratio, 1 - eps_lo, 1 + eps_hi) * a_
            pg = -torch.min(unclipped, clipped) * lp        # REINFORCE w/ baseline
            loss = (pg * mask).sum() / mask.sum().clamp(min=1) / accum
            loss.backward()
            loss_acc += loss.item()

        if (step + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(params, cfg.get("clip_grad", 0.5))
            opt.step()
            opt.zero_grad(set_to_none=True)

        log.append({"step": step, "task": task, "ds": row["dataset_name"],
                    "reward": float(rewards.mean()), "raw": float(np.mean(raws)),
                    "best": float(max(raws)), "loss": loss_acc})
        if step % cfg.get("log_steps", 10) == 0:
            r = np.mean([l["reward"] for l in log[-50:]])
            x = np.mean([l["raw"] for l in log[-50:]])
            print(f"[grpo] {step}/{steps} r={r:.3f} raw={x:.3f} "
                  f"task={task} ds={row['dataset_name']} "
                  f"{(time.time()-t0)/max(1,step+1):.1f}s/step", flush=True)
        if step and step % cfg.get("save_steps", 150) == 0:
            out = os.path.join(cfg["output_dir"], f"step{step}")
            model.save_pretrained(out); processor.save_pretrained(out)
            json.dump(normalizer.state(), open(os.path.join(out, "norm.json"), "w"))
            json.dump(log, open(os.path.join(cfg["output_dir"], "log.json"), "w"))

    out = os.path.join(cfg["output_dir"], "final")
    model.save_pretrained(out); processor.save_pretrained(out)
    json.dump(normalizer.state(), open(os.path.join(out, "norm.json"), "w"))
    json.dump(log, open(os.path.join(cfg["output_dir"], "log.json"), "w"))
    print("[grpo] saved ->", out)


if __name__ == "__main__":
    main()
