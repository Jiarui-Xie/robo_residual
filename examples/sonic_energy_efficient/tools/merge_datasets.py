"""Merge multiple reference dataset npz files into one."""
from __future__ import annotations
import argparse
import numpy as np
from pathlib import Path

KEYS = ["jpos_abs", "jpos_delta", "jvel", "ang_vel_b", "gravity_b",
        "actions", "root_state_w", "tokens", "cmd_vel"]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    parts = [dict(np.load(p)) for p in args.inputs]
    merged = {}
    for k in KEYS:
        merged[k] = np.concatenate([p[k] for p in parts], axis=0)
    # copy scalar metadata from first file
    for k in ["default_angles", "dt"]:
        if k in parts[0]:
            merged[k] = parts[0][k]
    merged["cmd_buckets"] = np.array(sorted(set(
        float(p["cmd_buckets"][0]) for p in parts
    )), dtype=np.float32)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **merged)

    T = len(merged["tokens"])
    import collections
    cmds = collections.Counter(merged["cmd_vel"][:, 0].round(2))
    print(f"saved {T} frames → {out}")
    for v, n in sorted(cmds.items()):
        print(f"  cmd={v:.1f}: {n} frames ({n*0.02:.0f}s)")

if __name__ == "__main__":
    main()
