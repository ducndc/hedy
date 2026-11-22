# hedy

A small GPT built from scratch in PyTorch, for studying and researching LLMs.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch   # or default index for GPU
pip install -e ".[dev]"
pytest
```

## Run
```bash
python -m hedy.train --steps 2000          # downloads Tiny Shakespeare, trains, saves checkpoints/
python -m hedy.train --tokenizer bpe --vocab_size 512   # byte-level BPE instead of chars
python -m hedy.train --pos rope --norm rmsnorm --mlp swiglu   # LLaMA-style block
python -m hedy.train --pos rope --norm rmsnorm --mlp swiglu --n_kv_head 2   # GQA
python -m hedy.bench                       # KV-cache speed-up and cache memory
python -m hedy.prepare --input data/tinystories-valid.txt --out_dir data/tinystories   # tokenize a corpus once
python -m hedy.train --data_dir data/tinystories --grad_accum 4 --dropout 0 --compile --pos rope --norm rmsnorm --mlp swiglu
python -m hedy.train --data_dir data/tinystories --resume    # continue from checkpoints/ckpt.pt
python -m hedy.sample --prompt "ROMEO:"
```

Study notes (read in order): [docs/README.md](docs/README.md). Plan: [docs/roadmap.md](docs/roadmap.md).
