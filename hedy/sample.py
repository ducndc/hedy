"""Sample text. Example: python -m hedy.sample --prompt "ROMEO:" """
import argparse
import os

import torch

from .model import GPT, GPTConfig
from .tokenizer import CharTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    a = p.parse_args()
    tok = CharTokenizer.load(os.path.join(a.ckpt, "tokenizer.json"))
    ck = torch.load(os.path.join(a.ckpt, "model.pt"), map_location="cpu")
    model = GPT(GPTConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    idx = torch.tensor([tok.encode(a.prompt)])
    print(tok.decode(model.generate(idx, a.tokens, a.temperature, a.top_k)[0].tolist()))


if __name__ == "__main__":
    main()
