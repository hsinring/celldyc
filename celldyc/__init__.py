import celldyc.tools.core as tl
import celldyc.plotting.core as pl
import celldyc.plotting.settings as settings
import celldyc.reproducibility.core as rp
import celldyc.datasets.core as datasets


__all__ = ["tl", "pl", "settings", "rp", "datasets"]


def set_seed(seed=42):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
