#!/usr/bin/env python3
"""
Smoke test target for scripts/gpu_keepalive.py.

Wraps a simulated training run with the standard keepalive try/except.
Use --mode to control the simulated outcome:

  --mode success
      Allocates a tensor, does a short matmul, exits 0. The except block
      should NEVER fire. Verifies the wrapper doesn't accidentally enter
      keepalive on a clean job.

  --mode fail
      Raises a RuntimeError to simulate a training crash. The except block
      catches it and calls keepalive(MODEL_DIR), which blocks forever until
      ${MODEL_DIR}/.keepalive.release is touched.

  --mode fail-release
      Same as fail, but ALSO writes ${MODEL_DIR}/.keepalive.release after
      --self_release_after seconds (default 90). Lets the smoke test orchestrator
      verify both "enter keepalive" and "exit on release" without manual
      intervention.

Markers written to MODEL_DIR for the orchestrator to read:
  .test.entered    -- training entered (always)
  .test.success    -- successful exit (mode=success)
  .test.crashed    -- exception caught (mode=fail*)
  .test.released   -- keepalive returned (mode=fail-release)
  .error.txt       -- traceback of the simulated failure
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import traceback
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["success", "fail", "fail-release"], required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--self_release_after", type=float, default=90.0,
                    help="seconds after entering keepalive to touch release (fail-release only)")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / ".test.entered").write_text(f"pid={os.getpid()} mode={args.mode}\n")

    print(f"[TEST] mode={args.mode} model_dir={model_dir}", flush=True)

    try:
        # ---- simulated training ----
        if args.mode == "success":
            import torch
            n = torch.cuda.device_count()
            print(f"[TEST] success path, n_gpus={n}", flush=True)
            if n > 0:
                a = torch.randn(2048, 2048, device="cuda:0")
                for _ in range(20):
                    _ = a @ a
                    torch.cuda.synchronize()
            (model_dir / ".test.success").write_text("ok\n")
            print("[TEST] success path complete; exiting 0", flush=True)
            return 0
        else:
            raise RuntimeError(f"Simulated training failure (mode={args.mode})")

    except BaseException as e:
        # ---- standard wrapper: persist error, free GPU, enter keepalive ----
        try:
            (model_dir / ".error.txt").write_text(traceback.format_exc())
        except Exception:
            pass
        (model_dir / ".test.crashed").write_text(f"{type(e).__name__}: {e}\n")
        print(f"[TEST] caught {type(e).__name__}: {e}", flush=True)

        # Free GPU memory before keepalive
        try:
            import gc, torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

        # Self-release timer for fail-release mode
        if args.mode == "fail-release":
            def _release_after_delay():
                try:
                    time.sleep(args.self_release_after)
                    (model_dir / ".keepalive.release").write_text("self-release\n")
                    print(f"[TEST] self-release after {args.self_release_after}s", flush=True)
                except Exception as exc:
                    print(f"[TEST] self-release thread exception: {exc!r}", flush=True)
            t = threading.Thread(target=_release_after_delay, daemon=True)
            t.start()

        # Import here so the success path doesn't pay the cost
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from gpu_keepalive import keepalive
        keepalive(model_dir)

        (model_dir / ".test.released").write_text("ok\n")
        print("[TEST] keepalive returned cleanly", flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
