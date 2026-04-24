"""Download GEAR-SONIC model checkpoints from Hugging Face Hub.

Usage:
    python download_models.py
    python download_models.py --output-dir ./models
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ID = "nvidia/GEAR-SONIC"

FILES = [
    ("model_encoder.onnx", "model_encoder.onnx"),
    ("model_decoder.onnx", "model_decoder.onnx"),
    ("observation_config.yaml", "observation_config.yaml"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download SONIC ONNX models")
    p.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "models")
    p.add_argument("--token", default=None, help="HuggingFace token")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading from {REPO_ID} -> {args.output_dir}\n")
    for hf_name, local_name in FILES:
        print(f"  {hf_name} ...", end=" ", flush=True)
        cached = hf_hub_download(repo_id=REPO_ID, filename=hf_name, token=args.token)
        dest = args.output_dir / local_name
        shutil.copy2(cached, dest)
        print(f"-> {dest}")

    print(f"\nDone. Models saved to {args.output_dir}")


if __name__ == "__main__":
    main()
