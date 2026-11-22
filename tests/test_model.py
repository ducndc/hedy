import torch

from hedy.model import GPT, GPTConfig
from hedy.tokenizer import CharTokenizer


def tiny():
    return GPT(GPTConfig(vocab_size=20, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0))


def test_forward_shapes_and_loss():
    m = tiny()
    x = torch.randint(0, 20, (3, 16))
    logits, loss = m(x, x)
    assert logits.shape == (3, 16, 20) and loss.item() > 0


def test_causality():
    m = tiny().eval()
    x = torch.randint(0, 20, (1, 16))
    y = x.clone(); y[0, -1] = (y[0, -1] + 1) % 20
    a, _ = m(x); b, _ = m(y)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)


def test_overfits_one_batch():
    torch.manual_seed(0)
    m = tiny(); opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    x = torch.randint(0, 20, (4, 16))
    first = m(x, x)[1].item()
    for _ in range(100):
        loss = m(x, x)[1]; opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < first * 0.5


def test_generate_and_tokenizer():
    out = tiny().generate(torch.zeros((1, 1), dtype=torch.long), 10)
    assert out.shape == (1, 11)
    t = CharTokenizer.from_text("hello")
    assert t.decode(t.encode("hello")) == "hello"


def test_step01_checkpoint_config_still_loads_into_the_current_model():
    """A step-01 checkpoint's config has only six fields; later options (pos, norm, mlp, n_kv_head) must default
    to the original architecture so the old weights load and give identical outputs."""
    torch.manual_seed(0)
    old_cfg = {"vocab_size": 20, "block_size": 16, "n_layer": 2, "n_head": 2, "n_embd": 32, "dropout": 0.0}
    a = GPT(GPTConfig(**old_cfg)).eval()
    b = GPT(GPTConfig(**old_cfg)); b.load_state_dict(a.state_dict()); b.eval()
    x = torch.randint(0, 20, (2, 16))
    assert torch.equal(a(x)[0], b(x)[0]) and a.cfg.pos == "learned" and a.cfg.mlp == "gelu" and a.cfg.n_kv_head == 0
