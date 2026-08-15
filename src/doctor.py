"""
PREFLIGHT — run this before anything else, and again whenever something breaks.

Ten seconds here would have saved the last cascade: a transformers-v5 rename, a
stale config key, and a missing checkpoint all surfaced as unrelated-looking
tracebacks three stages apart.

    python -m src.doctor
    python -m src.doctor --stage sft      # also check SFT prerequisites
    python -m src.doctor --stage infer

Exit code 0 = safe to proceed, 1 = something will fail.
"""
from __future__ import annotations
import argparse, importlib, os, shutil, sys, json

OK, WARN, BAD = "  ok  ", " warn ", " FAIL "
_fails, _warns = [], []


def check(label, fn, hint="", fatal=True):
    try:
        good, detail = fn()
    except Exception as e:
        good, detail = False, f"{type(e).__name__}: {e}"
    tag = OK if good else (BAD if fatal else WARN)
    print(f"[{tag}] {label:38s} {detail}")
    if not good:
        (_fails if fatal else _warns).append((label, detail, hint))
    return good


# ------------------------------------------------------------------ env

def c_python():
    v = sys.version_info
    return (v >= (3, 9)), f"{v.major}.{v.minor}.{v.micro}"


def c_torch():
    import torch
    cuda = torch.cuda.is_available()
    d = f"torch {torch.__version__}"
    if cuda:
        cap = torch.cuda.get_device_capability(0)
        gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        d += f" | {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]} {gb:.0f}GB"
    return cuda, d + ("" if cuda else " | NO CUDA")


def c_transformers():
    import transformers as t
    ver = t.__version__
    major = int(ver.split(".")[0])
    have = [c for c in ("Qwen3VLForConditionalGeneration",
                        "AutoModelForImageTextToText",
                        "Qwen2_5_VLForConditionalGeneration",
                        "AutoModelForVision2Seq") if hasattr(t, c)]
    note = f"v{ver} | classes: {', '.join(have) or 'NONE'}"
    if major >= 5:
        note += " | v5: AutoModelForVision2Seq removed (handled)"
    return bool(have), note


def c_qwen3vl():
    import transformers as t
    ok = hasattr(t, "Qwen3VLForConditionalGeneration")
    return ok, ("Qwen3VLForConditionalGeneration present" if ok else
                "MISSING — Qwen3-VL unsupported by this transformers build")


def c_pkg(name, attr=None):
    def f():
        m = importlib.import_module(name)
        return True, getattr(m, "__version__", "installed")
    return f


def c_flashattn():
    try:
        importlib.import_module("flash_attn")
        return True, "present (configs use sdpa anyway)"
    except ImportError:
        return True, "absent — fine, configs use sdpa"


# ------------------------------------------------------------------ paths

def c_paths_yaml():
    import yaml
    p = "configs/paths.yaml"
    if not os.path.exists(p):
        return False, "missing — run: python -m src.paths --root <root> --write"
    d = yaml.safe_load(open(p)) or {}
    need = ["trainval_json", "trainval_frame_root", "trainval_src_prefix",
            "test_json", "test_frame_root", "test_src_prefix", "ssd_cache"]
    miss = [k for k in need if k not in d]
    if miss:
        return False, f"missing keys: {miss}"
    return True, f"{len(d)} keys, both splits resolved"


def c_files():
    import yaml
    d = yaml.safe_load(open("configs/paths.yaml"))
    bad = [k for k in ("trainval_json", "test_json")
           if not os.path.exists(d.get(k, ""))]
    return (not bad), (f"missing: {bad}" if bad else "both json present")


def c_frames():
    import yaml, random
    d = yaml.safe_load(open("configs/paths.yaml"))
    out = []
    for split in ("trainval", "test"):
        jp, root = d.get(f"{split}_json"), d.get(f"{split}_frame_root")
        pref = d.get(f"{split}_src_prefix") or ""
        if not jp or not os.path.exists(jp):
            continue
        rows = json.load(open(jp))
        hit = tot = 0
        for r in random.Random(0).sample(rows, min(25, len(rows))):
            p = r["video"][0]
            rel = p[len(pref):].lstrip("/") if pref and p.startswith(pref) \
                else p.lstrip("/")
            tot += 1
            hit += os.path.exists(os.path.join(root, rel))
        out.append(f"{split} {hit}/{tot}")
    good = all(x.split()[-1].split("/")[0] == x.split()[-1].split("/")[1]
               for x in out) if out else False
    return good, " | ".join(out)


def c_cache():
    import yaml
    d = yaml.safe_load(open("configs/paths.yaml"))
    c = d.get("ssd_cache", "")
    if not os.path.exists(c):
        return False, f"{c} does not exist yet — run src.cache_frames"
    n = sum(len(f) for _, _, f in os.walk(c))
    free = shutil.disk_usage(c).free / 1e9
    return n > 1000, f"{n} files cached, {free:.0f}GB free on that volume"


def c_disk():
    free = shutil.disk_usage(".").free / 1e9
    return free > 60, f"{free:.0f}GB free in cwd (need ~60GB for weights+cache)"


# --------------------------------------------------------------- stages

def c_prep():
    ok = os.path.exists("data/train.json") and os.path.exists("data/val.json")
    if not ok:
        return False, "run: python -m src.prep_data --trainval ... --out data"
    n = len(json.load(open("data/train.json")))
    m = len(json.load(open("data/val.json")))
    return True, f"train {n} / val {m} rows"


def c_sft_ckpt():
    p = "runs/sft/final"
    if not os.path.isdir(p):
        return False, f"{p} does not exist — SFT has not produced a checkpoint"
    have = os.listdir(p)
    ok = any(f.startswith("adapter_model") for f in have)
    return ok, (f"{len(have)} files" if ok else f"no adapter_model in {have[:6]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="env",
                    choices=["env", "data", "sft", "grpo", "infer", "all"])
    a = ap.parse_args()
    st = a.stage

    print("=" * 74)
    print("PREFLIGHT")
    print("=" * 74)
    check("python", c_python)
    check("torch + cuda", c_torch)
    check("transformers", c_transformers,
          "pip install -U 'transformers>=4.57'")
    check("Qwen3-VL support", c_qwen3vl,
          "pip install -U 'transformers>=4.57'  (or use a Qwen2.5-VL base)")
    check("peft", c_pkg("peft"), "pip install -U peft")
    check("accelerate", c_pkg("accelerate"), "pip install -U accelerate")
    check("qwen_vl_utils", c_pkg("qwen_vl_utils"),
          "pip install -U 'qwen-vl-utils[decord]'")
    check("PIL", c_pkg("PIL"), "pip install -U pillow")
    check("flash-attn", c_flashattn, fatal=False)
    check("disk space", c_disk, fatal=False)

    if st in ("data", "sft", "grpo", "infer", "all"):
        print("-" * 74)
        check("configs/paths.yaml", c_paths_yaml,
              "python -m src.paths --root /media/data2/MedVIU_valdata --write")
        if os.path.exists("configs/paths.yaml"):
            check("json files exist", c_files)
            check("frames resolve on disk", c_frames,
                  "re-run src.paths; check *_frame_root and *_src_prefix")
            check("frame cache", c_cache,
                  "python -m src.cache_frames ...", fatal=False)

    if st in ("sft", "grpo", "infer", "all"):
        print("-" * 74)
        check("data/train.json + val.json", c_prep,
              "python -m src.prep_data --trainval <json> --out data")

    if st in ("grpo", "infer", "all"):
        print("-" * 74)
        check("runs/sft/final adapter", c_sft_ckpt,
              "train SFT first, or point --adapter at a real directory",
              fatal=(st != "infer"))

    print("=" * 74)
    if _warns:
        print(f"{len(_warns)} warning(s):")
        for l, d, h in _warns:
            print(f"  - {l}: {d}" + (f"\n      -> {h}" if h else ""))
    if _fails:
        print(f"\n{len(_fails)} BLOCKING problem(s):")
        for l, d, h in _fails:
            print(f"  - {l}: {d}")
            if h:
                print(f"      -> {h}")
        sys.exit(1)
    print("All checks passed. Safe to proceed.")


if __name__ == "__main__":
    main()
