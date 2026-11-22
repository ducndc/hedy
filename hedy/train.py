"""Train a small GPT.

Quick start (Shakespeare, chars):  python -m hedy.train --steps 2000
Bigger corpus:  python -m hedy.prepare --input data/tinystories-valid.txt --out_dir data/tinystories
                python -m hedy.train --data_dir data/tinystories --grad_accum 4 --dtype bfloat16 --compile
"""
import argparse
import contextlib
import json
import math
import os
import time

import torch

from .data import download_shakespeare, get_batch, load_bin, split
from .model import GPT, GPTConfig
from .tokenizer import BPETokenizer, CharTokenizer, load_tokenizer


def lr_at(step, steps, lr, warmup=100, min_ratio=0.1):
    if step < warmup:
        return lr * (step + 1) / warmup
    t = (step - warmup) / max(1, steps - warmup)
    return lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t)))


def make_optimizer(model, lr, weight_decay, device):
    """AdamW with weight decay only on matrices (>=2-D). Norm gains and biases are not decayed:
    shrinking a scale parameter toward 0 is not regularization, it just fights the normalizer."""
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), fused=(device == "cuda"))


def accumulate(model, xs, ys, ctx, scaler=None):
    """Forward+backward over micro-batches, adding gradients. Each micro-loss is divided by the
    number of micro-batches, so the summed gradient equals the gradient of the mean loss over the
    whole (large) batch. Returns the mean loss (a float). Does not step the optimizer."""
    total = 0.0
    for x, y in zip(xs, ys):
        with ctx:
            loss = model(x, y)[1] / len(xs)
        (scaler.scale(loss) if scaler else loss).backward()
        total += loss.item()
    return total


@torch.no_grad()
def estimate_loss(model, data, args, device, ctx):
    model.eval()
    ls = []
    for _ in range(args.eval_iters):
        x, y = get_batch(data, args.batch_size, args.block_size, device)
        with ctx:
            ls.append(model(x, y)[1].item())
    model.train()
    return sum(ls) / len(ls)


def build_parser():
    p = argparse.ArgumentParser()
    d = p.add_argument
    d("--data", default=None, help="plain text file (tokenized in memory); default Tiny Shakespeare")
    d("--data_dir", default=None, help="output of hedy.prepare (tokenizer + train.bin + val.bin)")
    d("--out", default="checkpoints")
    d("--tokenizer", choices=["char", "bpe"], default="char", help="only for --data")
    d("--vocab_size", type=int, default=512, help="BPE only, only for --data")
    d("--steps", type=int, default=2000, help="optimizer steps")
    d("--batch_size", type=int, default=32, help="micro-batch (what fits in memory at once)")
    d("--grad_accum", type=int, default=1, help="micro-batches per optimizer step")
    d("--block_size", type=int, default=128)
    d("--n_layer", type=int, default=4)
    d("--n_head", type=int, default=4)
    d("--n_embd", type=int, default=128)
    d("--pos", choices=["learned", "rope"], default="learned")
    d("--norm", choices=["layernorm", "rmsnorm"], default="layernorm")
    d("--mlp", choices=["gelu", "swiglu"], default="gelu")
    d("--n_kv_head", type=int, default=0, help="0=MHA, 1=MQA, else GQA (must divide n_head)")
    d("--dtype", choices=["float32", "bfloat16", "float16"], default="float32", help="autocast dtype")
    d("--compile", action="store_true", help="torch.compile the model")
    d("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    d("--dropout", type=float, default=0.1)
    d("--lr", type=float, default=3e-3)
    d("--weight_decay", type=float, default=0.1)
    d("--warmup", type=int, default=100)
    d("--grad_clip", type=float, default=1.0)
    d("--eval_every", type=int, default=250)
    d("--eval_iters", type=int, default=20)
    d("--log_every", type=int, default=50)
    d("--ckpt_every", type=int, default=500, help="save resumable checkpoint every N steps")
    d("--resume", action="store_true", help="continue from <out>/ckpt.pt")
    d("--stop_after", type=int, default=0, help="debug: stop (and checkpoint) after this step")
    d("--seed", type=int, default=1337)
    return p


def load_data(args):
    """Returns tok, train, val, bpb_scale (bits per byte = val_loss * bpb_scale)."""
    if args.data_dir:
        meta = json.load(open(os.path.join(args.data_dir, "meta.json")))
        tok = load_tokenizer(os.path.join(args.data_dir, "tokenizer.json"))
        train, val = load_bin(os.path.join(args.data_dir, "train.bin")), load_bin(os.path.join(args.data_dir, "val.bin"))
        print(f"data_dir={args.data_dir} vocab={tok.vocab_size} train_tokens={len(train):,} val_tokens={len(val):,}")
        return tok, train, val, meta["val_tokens"] / meta["val_bytes"] / math.log(2)
    text = open(args.data or download_shakespeare(), encoding="utf-8").read()
    tok = BPETokenizer.train(text, args.vocab_size) if args.tokenizer == "bpe" else CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    train, val = split(ids)
    # Loss is per token; a BPE token covers more text than a char token, so convert to bits per byte.
    val_bytes = len(tok.decode(val.tolist()).encode("utf-8"))
    print(f"tokenizer={args.tokenizer} vocab={tok.vocab_size} tokens={len(ids):,} "
          f"bytes/token={len(text.encode()) / len(ids):.2f}")
    return tok, train, val, len(val) / val_bytes / math.log(2)


def run(args):
    torch.manual_seed(args.seed)       # reproducible weight init (per-step seeds below cover data/dropout)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, train, val, bpb_scale = load_data(args)

    # --- mixed precision: run matmuls in low precision, keep master weights/optimizer in fp32 ---
    use_amp = args.dtype != "float32"
    assert not (use_amp and device == "cpu" and args.dtype == "float16"), "use bfloat16 on CPU"
    ctx = torch.autocast(device, dtype=getattr(torch, args.dtype)) if use_amp else contextlib.nullcontext()
    # fp16 has a tiny exponent range, so gradients underflow: scale the loss up. bf16 doesn't need it.
    scaler = torch.amp.GradScaler(device) if (args.dtype == "float16" and device == "cuda") else None

    cfg = GPTConfig(tok.vocab_size, args.block_size, args.n_layer, args.n_head, args.n_embd, args.dropout,
                    args.pos, args.norm, args.mlp, args.n_kv_head)
    raw = GPT(cfg).to(device)          # `raw` owns the weights/checkpoints; `model` may be compiled
    opt = make_optimizer(raw, args.lr, args.weight_decay, device)
    start = 0
    ckpt_path = os.path.join(args.out, "ckpt.pt")
    if args.resume:
        ck = torch.load(ckpt_path, map_location=device)
        raw.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start = ck["step"]
        print(f"resumed from step {start}")
    model = torch.compile(raw) if args.compile else raw
    tokens_per_step = args.batch_size * args.grad_accum * args.block_size
    print(f"device={device} params={sum(p.numel() for p in raw.parameters())/1e6:.2f}M dtype={args.dtype} "
          f"compile={args.compile} batch={args.batch_size}x{args.grad_accum} tokens/step={tokens_per_step:,}")

    os.makedirs(args.out, exist_ok=True)
    tok.save(os.path.join(args.out, "tokenizer.json"))

    def save(step):
        ck = {"config": cfg.__dict__, "model": raw.state_dict(), "opt": opt.state_dict(), "step": step}
        torch.save(ck, ckpt_path + ".tmp"); os.replace(ckpt_path + ".tmp", ckpt_path)  # atomic

    model.train()
    t0, last_loss = time.perf_counter(), float("nan")
    for step in range(start, args.steps):
        # Seed from the step so a resumed run draws exactly the batches an uninterrupted run would.
        torch.manual_seed(args.seed * 1_000_003 + step)
        lr = lr_at(step, args.steps, args.lr, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        if step % args.eval_every == 0:
            vl = estimate_loss(model, val, args, device, ctx)
            tl = estimate_loss(model, train, args, device, ctx)
            print(f"step {step}: train {tl:.3f} val {vl:.3f} val_bpb {vl * bpb_scale:.3f}", flush=True)
        batches = [get_batch(train, args.batch_size, args.block_size, device) for _ in range(args.grad_accum)]
        last_loss = accumulate(model, [b[0] for b in batches], [b[1] for b in batches], ctx, scaler)
        if scaler:
            scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(raw.parameters(), args.grad_clip)
        if scaler:
            scaler.step(opt); scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % args.log_every == 0:
            dt = time.perf_counter() - t0
            print(f"  step {step + 1}: loss {last_loss:.3f} lr {lr:.2e} gnorm {gnorm:.2f} "
                  f"{args.log_every * tokens_per_step / dt:,.0f} tok/s", flush=True)
            t0 = time.perf_counter()
        if (step + 1) % args.ckpt_every == 0 or step + 1 == args.stop_after:
            save(step + 1)
        if step + 1 == args.stop_after:
            return raw

    vl = estimate_loss(model, val, args, device, ctx)
    print(f"final: val {vl:.3f} val_bpb {vl * bpb_scale:.3f}")
    torch.save({"config": cfg.__dict__, "model": raw.state_dict()}, os.path.join(args.out, "model.pt"))
    save(args.steps)
    print("saved", args.out)
    return raw


if __name__ == "__main__":
    run(build_parser().parse_args())
