"""
STEP 3 - Stage-1 LoRA SFT.

6,270 real rows (+ ~2,200 harvested aux rows). At that scale full fine-tuning a
4B on 3 epochs will memorise and generalise worse than LoRA. 
Rank-64 is used for LoRA on the LLM + projector, ViT frozen.

    accelerate launch -m src.train_sft --config configs/sft.yaml
"""
from __future__ import annotations
import argparse, os, json, math, yaml, torch
from transformers import (AutoProcessor, AutoConfig, Trainer, TrainingArguments,
                          set_seed)
from peft import LoraConfig, get_peft_model

from .dataset import MedVidDataset, make_collator
from .balance import make_sampler, difficulty_from_eval
from .groups import filter_rows, summarise, GROUP_NAMES


MODEL_CLASS_CANDIDATES = [
    # transformers v5 renamed the generic vision-language auto class; the old
    # AutoModelForVision2Seq no longer exists there. Try newest-first and fall
    # back, so one requirements.txt works across 4.5x and 5.x.
    ("transformers", "Qwen3VLForConditionalGeneration"),
    ("transformers", "Qwen3VLMoeForConditionalGeneration"),
    ("transformers", "AutoModelForImageTextToText"),   # v5 generic
    ("transformers", "Qwen2_5_VLForConditionalGeneration"),
    ("transformers", "AutoModelForVision2Seq"),        # <=4.5x generic
    ("transformers", "AutoModel"),
]


def load_model(cfg):
    """Load the VLM under whichever transformers version is installed."""
    import importlib
    kw = dict(dtype=torch.bfloat16,
              attn_implementation=cfg.get("attn", "sdpa"),
              device_map=None)
    name = cfg["model"]
    tried = []
    for mod, cls_name in MODEL_CLASS_CANDIDATES:
        try:
            cls = getattr(importlib.import_module(mod), cls_name)
        except (ImportError, AttributeError):
            tried.append(f"{cls_name}: not in this transformers version")
            continue
        for kwargs in (kw, {**{k: v for k, v in kw.items() if k != "dtype"},
                            "torch_dtype": torch.bfloat16}):
            try:
                m = cls.from_pretrained(name, **kwargs)
                print(f"[model] loaded {name} via {cls_name}")
                return m
            except TypeError as e:
                if "dtype" in str(e) or "torch_dtype" in str(e):
                    continue          # old/new dtype kwarg mismatch, retry
                tried.append(f"{cls_name}: {type(e).__name__}: {e}")
                break
            except Exception as e:
                tried.append(f"{cls_name}: {type(e).__name__}: {str(e)[:160]}")
                break
    import transformers
    raise RuntimeError(
        "Could not load the model with any known class.\n"
        f"  model  : {name}\n"
        f"  transformers {transformers.__version__}\n  tried:\n    "
        + "\n    ".join(tried)
        + "\n\nIf Qwen3-VL is missing, upgrade:  "
          "pip install -U 'transformers>=4.57'")


def freeze_vision(model):
    n_frozen = 0
    for name, p in model.named_parameters():
        if any(k in name for k in ("visual.blocks", "vision_tower.", "visual.patch_embed")):
            p.requires_grad_(False)
            n_frozen += p.numel()
    print(f"[sft] froze {n_frozen/1e6:.1f}M vision params")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    for kv in a.override:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    print(json.dumps(cfg, indent=2))
    set_seed(cfg.get("seed", 0))

    processor = AutoProcessor.from_pretrained(
        cfg["model"],
        min_pixels=cfg.get("min_pixels", 64 * 28 * 28),
        max_pixels=cfg.get("max_pixels", 200 * 28 * 28),
    )
    # cap video token budget explicitly - this is the main VRAM knob
    if hasattr(processor, "image_processor"):
        ip = processor.image_processor
        for k, v in [("min_pixels", cfg.get("min_pixels")),
                     ("max_pixels", cfg.get("max_pixels"))]:
            if v and hasattr(ip, k):
                setattr(ip, k, v)

    model = load_model(cfg)
    model.config.use_cache = False
    freeze_vision(model)

    lora = LoraConfig(
        r=cfg.get("lora_r", 64), lora_alpha=cfg.get("lora_alpha", 128),
        lora_dropout=cfg.get("lora_dropout", 0.05), bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.get("lora_targets",
                               ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"]),
        modules_to_save=cfg.get("modules_to_save", None),
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    if cfg.get("grad_ckpt", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    common = dict(processor=processor, src_prefix=cfg.get("src_prefix", "/root/data"),
                  cache_root=cfg.get("cache_root"), max_len=cfg.get("max_len", 8192),
                  budget_scale=cfg.get("budget_scale", 1.0),
                  max_frames=cfg.get("max_frames", 64))
    train_ds = MedVidDataset(cfg["train_json"], train=True, **common)
    grp = cfg.get("group")            # None/"all" = shared multi-task (default)
    if grp and grp != "all":
        n0 = len(train_ds.rows)
        train_ds.rows = filter_rows(train_ds.rows, grp)
        print(f"[sft] ROUTED ADAPTER group='{grp}' "
              f"({GROUP_NAMES.get(grp, grp)}): {len(train_ds.rows)}/{n0} rows")
        summarise(train_ds.rows)
        if len(train_ds.rows) < 800:
            print("[sft] !! under 800 rows for this group - expect memorisation.")
    eval_ds = (MedVidDataset(cfg["val_json"], train=True, **common)
               if cfg.get("val_json") and cfg.get("do_eval", True) else None)
    if eval_ds is not None and grp and grp != "all":
        eval_ds.rows = filter_rows(eval_ds.rows, grp)
    print(f"[sft] train={len(train_ds)} val={len(eval_ds) if eval_ds else 0}")

    args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg.get("epochs", 3),
        per_device_train_batch_size=cfg.get("bs", 1),
        gradient_accumulation_steps=cfg.get("grad_accum", 8),
        per_device_eval_batch_size=1,
        learning_rate=cfg.get("lr", 1e-4),
        warmup_ratio=cfg.get("warmup", 0.05),
        lr_scheduler_type="cosine",
        weight_decay=0.01, max_grad_norm=1.0,
        bf16=True, tf32=True,
        logging_steps=cfg.get("log_steps", 10),
        save_strategy="epoch", save_total_limit=3,
        eval_strategy="epoch" if eval_ds else "no",
        dataloader_num_workers=cfg.get("workers", 8),
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=cfg.get("report_to", "none"),
        gradient_checkpointing=False,   # handled manually above
        seed=cfg.get("seed", 0),
    )

    class BalancedTrainer(Trainer):
        """C2: temperature-scaled inverse-frequency sampling over (dataset,task)."""
        def _get_train_sampler(self, *a, **k):
            tau = cfg.get("balance_tau", 0.5)
            if not tau:
                return super()._get_train_sampler(*a, **k)
            return make_sampler(
                self.train_dataset.rows, tau=tau,
                difficulty=difficulty_from_eval(cfg["difficulty_json"])
                if cfg.get("difficulty_json") else None,
                gamma=cfg.get("balance_gamma", 1.0),
                aux_weight=cfg.get("aux_weight", 0.7))

    trainer = BalancedTrainer(model=model, args=args, train_dataset=train_ds,
                              eval_dataset=eval_ds,
                              data_collator=make_collator(
                                  processor, cfg.get("max_len", 8192), True))
    trainer.train(resume_from_checkpoint=cfg.get("resume", None))
    trainer.save_model(os.path.join(cfg["output_dir"], "final"))
    processor.save_pretrained(os.path.join(cfg["output_dir"], "final"))
    print("[sft] saved ->", os.path.join(cfg["output_dir"], "final"))


if __name__ == "__main__":
    main()
