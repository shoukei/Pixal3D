"""Translate `device='cuda'` strings to the active accelerator on MPS/CPU builds.

The macOS torch build has no CUDA support and ignores PYTORCH_ENABLE_MPS_FALLBACK,
so any `device='cuda'` (in tensor constructors, `Tensor.to`, `torch.linspace`,
third-party Metal code, ...) raises "Torch not compiled with CUDA enabled".

macos_compat.configure() already patches `.cuda()` methods; this module covers
the remaining path by translating `torch.device('cuda')` -> the configured
device (MPS or CPU). Import it after macos_compat.configure() on non-CUDA
machines. It is a no-op when CUDA is available.
"""

import os
import torch

_target = os.environ.get("PIXAL3D_DEVICE", "auto")
if not torch.cuda.is_available() and _target not in {"cuda"}:
    _is_mps = torch.backends.mps.is_available()
    _fallback_type = "mps" if _is_mps else "cpu"

    _orig_device = torch.device

    def _device(*args, **kwargs):
        dev = _orig_device(*args, **kwargs)
        if dev.type == "cuda":
            dev = _orig_device(_fallback_type)
        return dev

    torch.device = _device
    # torch.tensor / torch.zeros etc. accept both positional and keyword device;
    # both route through torch.device(), so the class patch covers them.
    print(f"[cuda_device_compat] torch.device('cuda') -> '{_fallback_type}' translation enabled")
