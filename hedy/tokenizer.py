"""Tokenizers: character-level (step 01) and byte-level BPE (step 02).

Both expose: vocab_size, encode(str)->list[int], decode(list[int])->str, save(path).
Use `load_tokenizer(path)` to get the right class back from a saved file.
"""
import json
import re
from collections import Counter


class CharTokenizer:
    kind = "char"

    def __init__(self, chars: list[str]):
        self.chars = chars
        self.stoi = {c: i for i, c in enumerate(chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.chars[i] for i in ids)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"type": self.kind, "chars": self.chars}, f)

    @classmethod
    def _from_dict(cls, d):
        return cls(d["chars"])


# Pre-tokenization: split text into chunks first, so merges never cross chunk boundaries
# (e.g. never merge the end of one word with the start of the next). GPT-2-like, ASCII-only
# so we need no third-party `regex` package. Non-ASCII chars fall in the "other" group.
SPLIT_PATTERN = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"
)


def _merge(ids: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    """Replace every non-overlapping occurrence of `pair` in `ids` with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return tuple(out)


class BPETokenizer:
    """Byte-level BPE. Base vocab = the 256 byte values, so any string is encodable (no <unk>)."""

    kind = "bpe"

    def __init__(self, merges: list[tuple[int, int]]):
        self.merges = [tuple(m) for m in merges]            # ordered: merge i creates token 256+i
        self.ranks = {m: i for i, m in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    # ---- training ----
    @classmethod
    def train(cls, text: str, vocab_size: int, verbose: bool = False) -> "BPETokenizer":
        assert vocab_size >= 256
        # Count each distinct chunk once (with frequency): far faster than touching the raw text.
        words = {tuple(c.encode("utf-8")): n for c, n in Counter(SPLIT_PATTERN.findall(text)).items()}
        merges: list[tuple[int, int]] = []
        for new_id in range(256, vocab_size):
            pairs: Counter = Counter()
            for w, n in words.items():
                for p in zip(w, w[1:]):
                    pairs[p] += n
            if not pairs:
                break
            best = max(pairs, key=lambda p: (pairs[p], p))   # tie-break by pair for determinism
            merges.append(best)
            words = {_merge(w, best, new_id): n for w, n in words.items()}
            if verbose:
                print(f"merge {len(merges)}: {best} -> {new_id} ({pairs[best]} occurrences)")
        return cls(merges)

    # ---- encode / decode ----
    def _encode_chunk(self, chunk: str) -> list[int]:
        if chunk in self._cache:
            return self._cache[chunk]
        ids = tuple(chunk.encode("utf-8"))
        while len(ids) > 1:
            # apply the earliest-learned merge present (lowest rank) first, as in training
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, float("inf")))
            if pair not in self.ranks:
                break
            ids = _merge(ids, pair, 256 + self.ranks[pair])
        self._cache[chunk] = list(ids)
        return self._cache[chunk]

    def encode(self, s: str) -> list[int]:
        return [t for chunk in SPLIT_PATTERN.findall(s) for t in self._encode_chunk(chunk)]

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"type": self.kind, "merges": self.merges}, f)

    @classmethod
    def _from_dict(cls, d):
        return cls(d["merges"])


def load_tokenizer(path: str):
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, list):          # legacy step-01 format: a bare list of characters
        return CharTokenizer(d)
    return {"char": CharTokenizer, "bpe": BPETokenizer}[d["type"]]._from_dict(d)
