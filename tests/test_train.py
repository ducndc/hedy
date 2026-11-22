import contextlib
import copy

import numpy as np
import torch

from hedy.data import get_batch, load_bin
from hedy.model import GPT, GPTConfig
from hedy.train import accumulate, build_parser, lr_at, make_optimizer, run


def tiny(**kw):
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=30, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0, **kw))


def test_grad_accumulation_equals_big_batch():
    m1 = tiny(); m2 = copy.deepcopy(m1)
    x, y = torch.randint(0, 30, (8, 16)), torch.randint(0, 30, (8, 16))
    l1 = accumulate(m1, [x], [y], contextlib.nullcontext())
    l2 = accumulate(m2, list(x.chunk(4)), list(y.chunk(4)), contextlib.nullcontext())  # 4 micro-batches of 2
    assert abs(l1 - l2) < 1e-5
    for a, b in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(a.grad, b.grad, atol=1e-6)


def test_weight_decay_only_on_matrices():
    opt = make_optimizer(tiny(norm="rmsnorm"), 1e-3, 0.1, "cpu")
    dec, nodec = opt.param_groups
    assert dec["weight_decay"] == 0.1 and nodec["weight_decay"] == 0.0
    assert all(p.ndim >= 2 for p in dec["params"]) and all(p.ndim < 2 for p in nodec["params"])
    assert len(nodec["params"]) > 0   # the RMSNorm gains


def test_lr_schedule():
    assert lr_at(0, 1000, 1.0, warmup=100) < 0.02            # starts near 0
    assert abs(lr_at(99, 1000, 1.0, warmup=100) - 1.0) < 1e-9  # peak at end of warmup
    assert abs(lr_at(1000, 1000, 1.0, warmup=100) - 0.1) < 1e-9  # decays to 10%


def test_bf16_autocast_close_to_fp32():
    m = tiny().eval()
    x = torch.randint(0, 30, (2, 16))
    ref = m(x)[0]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = m(x)[0]
    assert out.dtype == torch.bfloat16 or out.dtype == torch.float32
    assert torch.allclose(ref, out.float(), atol=0.1)


def test_memmap_batches(tmp_path):
    ids = (np.arange(1000) % 50).astype(np.uint16)
    ids.tofile(tmp_path / "t.bin")
    data = load_bin(str(tmp_path / "t.bin"))
    x, y = get_batch(data, 4, 8)
    assert x.shape == y.shape == (4, 8) and x.dtype == torch.int64
    assert torch.equal(y[:, :-1], x[:, 1:])                    # y is x shifted by one


def _args(tmp_path, out, *extra):
    text = tmp_path / "t.txt"
    if not text.exists():
        text.write_text("the quick brown fox jumps over the lazy dog. " * 200)
    return build_parser().parse_args([
        "--data", str(text), "--out", str(tmp_path / out), "--steps", "6", "--batch_size", "4",
        "--grad_accum", "2", "--block_size", "16", "--n_layer", "2", "--n_head", "2", "--n_embd", "32",
        "--dropout", "0.1", "--eval_every", "1000", "--eval_iters", "2", "--log_every", "100",
        "--ckpt_every", "100", "--warmup", "2", *extra])


def test_resume_reproduces_uninterrupted_run(tmp_path):
    full = run(_args(tmp_path, "full"))
    run(_args(tmp_path, "part", "--stop_after", "3"))         # stops at step 3, writes ckpt.pt
    resumed = run(_args(tmp_path, "part", "--resume"))        # continues 3 -> 6
    for (n, a), (_, b) in zip(full.state_dict().items(), resumed.state_dict().items()):
        assert torch.allclose(a, b, atol=1e-6), n


def test_training_reduces_loss(tmp_path, capsys):
    run(_args(tmp_path, "learn", "--steps", "60", "--eval_every", "30", "--lr", "3e-3"))
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("step") and "val" in l]
    first, last = float(lines[0].split("val ")[1].split()[0]), float(lines[-1].split("val ")[1].split()[0])
    assert last < first * 0.7
