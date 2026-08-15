"""
Background job launcher that survives a closed notebook.

"""
from __future__ import annotations
import argparse, os, signal, subprocess, sys, time, json

def _test_cmd(extra_flags):
    """Build the test-set inference command from configs/paths.yaml."""
    from .paths import cfg
    return [sys.executable, "-m", "src.infer",
            "--test", cfg("test_json"),
            "--base", os.environ.get("BASE_MODEL",
                                     "Qwen/Qwen3-VL-4B-Instruct"),
            "--adapter", "runs/sft/final",
            "--cache", cfg("ssd_cache"),
            "--src_prefix", cfg("test_src_prefix", "/root/data"),
            "--orig_root", cfg("test_frame_root"),
            "--out", "preds/test_sft"] + extra_flags


JOBS = {
    "sft":  [sys.executable, "-m", "src.train_sft", "--config", "configs/sft.yaml"],
    "grpo": [sys.executable, "-m", "src.train_grpo", "--config", "configs/grpo.yaml"],
    # the long one. ~13h. Fully resumable — re-launch and it picks up.
    "test": None,       # built lazily, see start()
    "testfast": None,   # no CTCD, fewer frames on pass 2
}
RUN_DIR = ".jobs"


def _paths(name):
    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    return (os.path.join(RUN_DIR, f"{name}.pid"),
            os.path.join("logs", f"{name}.log"))


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def start(name, extra=None, force=False):
    if name not in JOBS:
        sys.exit(f"unknown job '{name}'. choose from {list(JOBS)}")
    if name == "test":
        JOBS["test"] = _test_cmd(["--ctcd", "--refine"])
    elif name == "testfast":
        JOBS["testfast"] = _test_cmd(["--refine", "--budget_scale", "0.7"])
    pidf, logf = _paths(name)
    if os.path.exists(pidf) and not force:
        pid = int(open(pidf).read().strip() or 0)
        if pid and _alive(pid):
            print(f"[launch] {name} already running as pid {pid}")
            print(f"[launch] logs: {logf}   (use --force to start another)")
            return pid
    cmd = JOBS[name] + (extra or [])
    print(f"[launch] {' '.join(cmd)}")
    print(f"[launch] log -> {logf}")
    with open(logf, "ab", buffering=0) as out:
        out.write(f"\n===== started {time.ctime()} =====\n".encode())
        p = subprocess.Popen(
            cmd, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True,          # detach from the kernel's group
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    open(pidf, "w").write(str(p.pid))
    time.sleep(3)
    if not _alive(p.pid):
        print(f"[launch] !! process exited immediately. Last log lines:")
        _tail(logf, 25)
        sys.exit(1)
    print(f"[launch] running as pid {p.pid} — safe to close the notebook.")
    print(f"[launch] check:  python -m src.launch status")
    return p.pid


def _tail(path, n=40):
    if not os.path.exists(path):
        print(f"(no log at {path})")
        return
    with open(path, errors="ignore") as f:
        lines = f.readlines()
    sys.stdout.write("".join(lines[-n:]))


def status(name=None):
    names = [name] if name else list(JOBS)
    for nm in names:
        pidf, logf = _paths(nm)
        if not os.path.exists(pidf):
            print(f"  {nm:6s} not started")
            continue
        pid = int(open(pidf).read().strip() or 0)
        alive = _alive(pid)
        size = os.path.getsize(logf) / 1e6 if os.path.exists(logf) else 0
        age = ((time.time() - os.path.getmtime(logf)) if os.path.exists(logf)
               else 1e9)
        state = "RUNNING" if alive else "stopped"
        stale = "  (log idle >10min — check it)" if alive and age > 600 else ""
        print(f"  {nm:6s} {state:8s} pid={pid}  log={size:.1f}MB{stale}")


def stop(name):
    pidf, _ = _paths(name)
    if not os.path.exists(pidf):
        sys.exit(f"{name} not started")
    pid = int(open(pidf).read().strip() or 0)
    if not _alive(pid):
        print(f"{name} already stopped")
        return
    os.killpg(os.getpgid(pid), signal.SIGTERM)
    print(f"sent SIGTERM to process group of pid {pid}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["sft", "grpo", "status", "tail", "stop"])
    ap.add_argument("--job", default="sft")
    ap.add_argument("--lines", type=int, default=40)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("extra", nargs="*")
    a = ap.parse_args()

    if a.action in JOBS:
        start(a.action, a.extra, a.force)
    elif a.action == "status":
        status()
    elif a.action == "tail":
        _tail(_paths(a.job)[1], a.lines)
    elif a.action == "stop":
        stop(a.job)


if __name__ == "__main__":
    main()
