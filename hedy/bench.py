"""Generation benchmark: KV-cache speed-up and KV-cache memory under MHA/GQA/MQA.
Random weights are fine here: we measure speed and memory, not quality.
Example: python -m hedy.bench"""
import argparse
import time

import torch

from .model import GPT, GPTConfig


def time_generate(model, prompt, n, use_cache):
    t = time.perf_counter()
    model.generate(prompt, n, top_k=1, use_cache=use_cache)
    return time.perf_counter() - t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_embd", type=int, default=256)
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)

    cfg = GPTConfig(vocab_size=512, block_size=a.block_size, n_layer=a.n_layer, n_head=a.n_head,
                    n_embd=a.n_embd, dropout=0.0, pos="rope", norm="rmsnorm", mlp="swiglu")
    model = GPT(cfg).eval()
    prompt = torch.randint(0, 512, (1, 16))

    print(f"== speed: {a.n_layer}L {a.n_embd}d {a.n_head}h, prompt 16 tokens, greedy, CPU {a.threads} threads")
    print(f"{'new tokens':>10} {'no cache (s)':>13} {'cache (s)':>10} {'speed-up':>9} {'tok/s cached':>13}")
    for n in (64, 128, 256, 480):
        slow, fast = time_generate(model, prompt, n, False), time_generate(model, prompt, n, True)
        print(f"{n:>10} {slow:>13.2f} {fast:>10.2f} {slow / fast:>8.1f}x {n / fast:>13.0f}")

    print(f"\n== KV-cache memory per sequence (fp32 here; halve for bf16), {a.n_layer} layers, head_dim {a.n_embd // a.n_head}")
    print(f"{'n_kv_head':>9} {'name':>5} {'@512 tokens':>13} {'@4096 tokens':>13} {'params':>9}")
    hd = a.n_embd // a.n_head
    for kv in (a.n_head, a.n_head // 2, a.n_head // 4, 1):
        m = GPT(GPTConfig(**{**cfg.__dict__, "n_kv_head": kv}))
        name = "MHA" if kv == a.n_head else "MQA" if kv == 1 else "GQA"
        per_tok = 2 * a.n_layer * kv * hd * 4          # k and v, all layers, 4 bytes
        print(f"{kv:>9} {name:>5} {per_tok * 512 / 1e6:>10.1f} MB {per_tok * 4096 / 1e6:>10.1f} MB "
              f"{sum(p.numel() for p in m.parameters()) / 1e6:>7.2f}M")


if __name__ == "__main__":
    main()
