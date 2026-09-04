"""
Torch compatibility shims.

`TTS==0.22.0` pins torch to 2.2.2, but the installed `speechbrain` (1.1.0)
expects the newer autocast API `torch.amp.custom_fwd(fwd, device_type=...,
cast_inputs=...)`, which only exists in torch >= 2.4. On 2.2.2 that lives under
`torch.cuda.amp` with a different (CUDA-only, no `device_type`) signature, so
SpeechBrain's ECAPA forward pass raises:

    module 'torch.amp' has no attribute 'custom_fwd'

For CPU/MPS inference (no autocast, float32), these decorators are effectively
no-ops, so we provide signature-compatible shims. This is a bridge, not the
real fix -- pin `speechbrain<1.0` (torch-2.2 compatible) in requirements to
remove the need for it.
"""

import torch


def apply_torch_amp_shim() -> None:
    if not hasattr(torch.amp, "custom_fwd"):
        def custom_fwd(fwd=None, *, device_type=None, cast_inputs=None):
            if fwd is None:
                return lambda f: f
            return fwd
        torch.amp.custom_fwd = custom_fwd

    if not hasattr(torch.amp, "custom_bwd"):
        def custom_bwd(bwd=None, *, device_type=None):
            if bwd is None:
                return lambda f: f
            return bwd
        torch.amp.custom_bwd = custom_bwd
