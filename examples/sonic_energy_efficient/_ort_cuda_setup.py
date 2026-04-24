"""Preload nvidia shared libraries so onnxruntime-gpu's dlopen can find them.

conda env `robo` has nvidia-* pip packages under site-packages/nvidia/<pkg>/lib/
but those dirs aren't on LD_LIBRARY_PATH by default, so `libonnxruntime_providers_cuda.so`
fails to load (missing libcublasLt.so.12 etc).

Import this module BEFORE importing onnxruntime (or anything that triggers it).
"""
from __future__ import annotations
import ctypes
import os
from pathlib import Path

_NVIDIA = Path(__file__).parent.parent.parent.parent / "miniconda3/envs/robo/lib/python3.11/site-packages/nvidia"
if not _NVIDIA.is_dir():
    # Fallback for other conda locations
    import site
    for sp in site.getsitepackages():
        cand = Path(sp) / "nvidia"
        if cand.is_dir():
            _NVIDIA = cand
            break


def preload_cuda_libs() -> None:
    """ctypes.CDLL(RTLD_GLOBAL) every .so under nvidia/*/lib so ORT can resolve symbols."""
    if not _NVIDIA.is_dir():
        print(f"[ort_cuda_setup] nvidia package dir not found; skipping ({_NVIDIA})")
        return
    pkgs = ["cublas", "cudnn", "cuda_runtime", "cufft", "curand",
            "cusolver", "cusparse", "nvjitlink", "cuda_nvrtc"]
    loaded = 0
    for pkg in pkgs:
        libdir = _NVIDIA / pkg / "lib"
        if not libdir.is_dir():
            continue
        for so in sorted(libdir.glob("*.so*")):
            try:
                ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
                loaded += 1
            except OSError:
                pass
    if loaded:
        print(f"[ort_cuda_setup] preloaded {loaded} nvidia shared libraries")


preload_cuda_libs()
