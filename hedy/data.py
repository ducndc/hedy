import os
import urllib.request

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


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device="cpu"):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
