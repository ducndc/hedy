from hedy.tokenizer import BPETokenizer, load_tokenizer

TEXT = "low lower lowest newer newest wider " * 20 + "héllo wörld 你好 😀"


def test_roundtrip_any_string():
    tok = BPETokenizer.train(TEXT, 300)
    for s in [TEXT, "", "  leading and trailing  ", "héllo 你好 😀\n\n\t", "unseen words zzzqqq"]:
        assert tok.decode(tok.encode(s)) == s


def test_merges_compress():
    tok = BPETokenizer.train(TEXT, 300)
    assert len(tok.encode(TEXT)) < len(TEXT.encode("utf-8"))
    assert tok.vocab_size == 256 + len(tok.merges)


def test_no_merge_across_chunks():
    tok = BPETokenizer.train("ab ab ab ab", 270)
    assert all(b" " not in tok.vocab[t][1:] for t in range(256, tok.vocab_size))


def test_deterministic_and_save_load(tmp_path):
    a, b = BPETokenizer.train(TEXT, 300), BPETokenizer.train(TEXT, 300)
    assert a.merges == b.merges
    a.save(tmp_path / "t.json")
    c = load_tokenizer(tmp_path / "t.json")
    assert c.encode(TEXT) == a.encode(TEXT)
