import os
import urllib.request

import numpy as np
import torch

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_shakespeare(path: str = "data/shakespeare.txt") -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    return path


def split(ids: torch.Tensor, val_frac: float = 0.1):
    n = int(len(ids) * (1 - val_frac))
    return ids[:n], ids[n:]


def load_bin(path: str) -> np.memmap:
    """Token file written by hedy.prepare: flat uint16 array, memory-mapped (never fully in RAM)."""
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data, batch_size: int, block_size: int, device="cpu"):
    """Random windows. `data` is a LongTensor (small corpora) or a uint16 memmap (large ones).
    y is x shifted by one token: that shift is the training signal."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,)).tolist()
    if isinstance(data, np.ndarray):
        x = torch.from_numpy(np.stack([data[i : i + block_size] for i in ix]).astype(np.int64))
        y = torch.from_numpy(np.stack([data[i + 1 : i + block_size + 1] for i in ix]).astype(np.int64))
    else:
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
