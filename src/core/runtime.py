"""Process-wide runtime setup. Call ``configure()`` once at the top of every
entrypoint (train / finetune / eval / infer)."""

import os
import torch


def configure(cpu_threads: int = 8, cudnn_benchmark: bool = True, seed: int | None = None):
    # RunPod gives 16 vCPUs on a 64-core host; without a cap, tiny CPU tensor ops
    # (e.g. during evaluation) fan out OpenMP threads across every core and thrash.
    n = max(1, min(cpu_threads, os.cpu_count() or cpu_threads))
    torch.set_num_threads(n)
    os.environ.setdefault("OMP_NUM_THREADS", str(n))

    # Fixed input size -> let cuDNN pick the fastest convolution algorithms.
    torch.backends.cudnn.benchmark = cudnn_benchmark

    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
