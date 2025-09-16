from .DSR_pretrain import *
from .DSR_finetune import *

try:
    # pylint: disable=wrong-import-position
    import torch
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "No module named 'torch', and DSR depends on PyTorch (aka 'torch')."
        "Visit https://pytorch.org/ for installation instructions.")