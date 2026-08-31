"""
Model complexity + resource profiling for the architecture comparison:
parameter count, FLOPs, on-disk size, peak activation memory, and train/infer
throughput. Kept separate from ``metrics.py`` (detection quality) so the
bake-off can report a full "quality vs. cost" picture.
"""

import io
import time
import torch
from torch.utils.flop_counter import FlopCounterMode


def parameter_count(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": total, "params_trainable": trainable, "params_millions": total / 1e6}


def model_size_mb(model: torch.nn.Module) -> float:
    """Serialized state_dict size in MB (what you would upload / ship)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


def flops(model: torch.nn.Module, device: str, img_size: int = 640) -> dict:
    """Forward-pass FLOPs for a single image, via torch's built-in counter."""
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        model(x)
    total = counter.get_total_flops()
    return {"gflops": total / 1e9, "flops": int(total)}


def peak_activation_memory_mb(model: torch.nn.Module, device: str,
                              img_size: int = 640, batch_size: int = 16) -> float:
    """Peak CUDA memory for one training step (fwd + bwd). 0.0 on CPU."""
    if torch.device(device).type != "cuda":
        return 0.0
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device, requires_grad=True)
    out = model(x)
    loss = sum(o.float().pow(2).mean() for o in out) if isinstance(out, (list, tuple)) else out.float().pow(2).mean()
    loss.backward()
    peak = torch.cuda.max_memory_allocated(device) / 1e6
    model.zero_grad(set_to_none=True)
    return peak


def train_throughput(model: torch.nn.Module, device: str, img_size: int = 640,
                     batch_size: int = 16, steps: int = 30, warmup: int = 5) -> float:
    """Images/second for a synthetic fwd+bwd loop (optimizer-free)."""
    model.train()
    is_cuda = torch.device(device).type == "cuda"
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)

    for i in range(warmup + steps):
        if i == warmup and is_cuda:
            torch.cuda.synchronize()
        if i == warmup:
            t0 = time.perf_counter()
        out = model(x)
        loss = sum(o.float().pow(2).mean() for o in out) if isinstance(out, (list, tuple)) else out.float().pow(2).mean()
        loss.backward()
        model.zero_grad(set_to_none=True)
    if is_cuda:
        torch.cuda.synchronize()
    return batch_size * steps / (time.perf_counter() - t0)


def profile_model(model: torch.nn.Module, device: str, img_size: int = 640,
                  batch_size: int = 16) -> dict:
    """Everything except detection quality, as one flat dict."""
    model.to(device)
    rep = {}
    rep.update(parameter_count(model))
    rep["model_size_mb"] = model_size_mb(model)
    rep.update(flops(model, device, img_size))
    rep["peak_train_mem_mb"] = peak_activation_memory_mb(model, device, img_size, batch_size)
    rep["train_imgs_per_s"] = train_throughput(model, device, img_size, batch_size)
    return rep
