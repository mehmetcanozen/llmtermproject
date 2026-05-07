from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeedReport:
    seed: int
    python_hash_seed: str
    numpy_seeded: bool
    torch_seeded: bool
    cuda_seeded: bool
    deterministic_torch: bool


def set_global_seed(seed: int, deterministic_torch: bool = True) -> SeedReport:
    if seed < 0:
        raise ValueError("Seed must be non-negative.")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch_seeded = False
    cuda_seeded = False
    try:
        import torch

        torch.manual_seed(seed)
        torch_seeded = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            cuda_seeded = True
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
    except ImportError:
        pass

    return SeedReport(
        seed=seed,
        python_hash_seed=os.environ["PYTHONHASHSEED"],
        numpy_seeded=True,
        torch_seeded=torch_seeded,
        cuda_seeded=cuda_seeded,
        deterministic_torch=deterministic_torch,
    )
