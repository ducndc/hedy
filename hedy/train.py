"""Train a small GPT. Example: python -m hedy.train --steps 2000"""
import argparse
import math
import os

import torch

from .data import download_shakespeare, get_batch, split
from .model import GPT, GPTConfig
from .tokenizer import CharTokenizer


def lr_at(step, steps, lr, warmup=100, min_ratio=0.1):
    if step < warmup:
        return lr * (step + 1) / warmup
    t = (step - warmup) / max(1, steps - warmup)
    return lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t)))


@torch.no_grad()
def estimate_loss(model, data, args, device, iters=20):
    model.eval()
    ls = [model(*get_batch(data, args.batch_size, args.block_size, device))[1].item() for _ in range(iters)]
    model.train()
    return sum(ls) / len(ls)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=None)
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = args.data or download_shakespeare()
    text = open(path, encoding="utf-8").read()
    tok = CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    train, val = split(ids)

    cfg = GPTConfig(tok.vocab_size, args.block_size, args.n_layer, args.n_head, args.n_embd, args.dropout)
    model = GPT(cfg).to(device)
    print(f"device={device} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    os.makedirs(args.out, exist_ok=True)
    tok.save(os.path.join(args.out, "tokenizer.json"))
    for step in range(args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr)
        if step % args.eval_every == 0:
            print(f"step {step}: train {estimate_loss(model, train, args, device):.3f} "
                  f"val {estimate_loss(model, val, args, device):.3f}")
        _, loss = model(*get_batch(train, args.batch_size, args.block_size, device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    torch.save({"config": cfg.__dict__, "model": model.state_dict()}, os.path.join(args.out, "model.pt"))
    print("saved", args.out)


if __name__ == "__main__":
    main()
