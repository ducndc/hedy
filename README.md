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
python -m hedy.sample --prompt "ROMEO:"
```

Study notes (read in order): [docs/README.md](docs/README.md). Plan: [docs/roadmap.md](docs/roadmap.md).
