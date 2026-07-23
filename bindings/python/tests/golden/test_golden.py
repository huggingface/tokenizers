"""Replay the golden inputs on the current build and diff every output.

One test per output domain, parametrized per model, so the failure report
reads as a conformance matrix: which field diverges (or raises) on which
tokenizer archetype. A mismatch means this build disagrees with the released
wheel the goldens were generated from — never edit a golden to make it pass;
regenerate them from the release (`make golden-regen`) or fix the build.
"""

import json
from functools import cache

import pytest
from tokenizers import Tokenizer

from .generate import DATA, GOLDENS, REPO, ids_digest, text_digest

FETCH_HINT = f"run `make -C {REPO / 'tokenizers'} bench-models fixtures`"


@cache
def records(name: str) -> list[dict]:
    return [json.loads(line) for line in (GOLDENS / f"{name}.jsonl").read_text().splitlines()]


@cache
def tokenizer(name: str) -> Tokenizer:
    meta = records(name)[0]
    path = DATA / meta["tokenizer_file"]
    if not path.is_file():
        pytest.skip(f"{path} missing — {FETCH_HINT}")
    return Tokenizer.from_file(str(path))


def golden_models() -> list[str]:
    names = sorted(path.stem for path in GOLDENS.glob("*.jsonl"))
    assert names, f"no golden files in {GOLDENS} — run `make golden-regen` and commit the result"
    return names


@pytest.fixture(params=golden_models())
def model(request):
    return request.param


def samples(name: str) -> list[dict]:
    return [r for r in records(name) if r["kind"] == "sample"]


def test_ids(model):
    tok = tokenizer(model)
    for s in samples(model):
        assert tok.encode(s["text"]).ids == s["ids"], s["source"]
        assert tok.encode(s["text"], add_special_tokens=False).ids == s["ids_no_specials"], (
            f"{s['source']} (no specials)"
        )


def test_tokens(model):
    tok = tokenizer(model)
    for s in samples(model):
        assert tok.encode(s["text"]).tokens == s["tokens"], s["source"]


def test_offsets(model):
    tok = tokenizer(model)
    for s in samples(model):
        assert [list(span) for span in tok.encode(s["text"]).offsets] == s["offsets"], s["source"]


def test_word_ids(model):
    tok = tokenizer(model)
    for s in samples(model):
        assert tok.encode(s["text"]).word_ids == s["word_ids"], s["source"]


def test_type_ids(model):
    tok = tokenizer(model)
    for s in samples(model):
        assert tok.encode(s["text"]).type_ids == s["type_ids"], s["source"]


def test_special_tokens_mask(model):
    tok = tokenizer(model)
    for s in samples(model):
        assert tok.encode(s["text"]).special_tokens_mask == s["special_tokens_mask"], s["source"]


def test_attention_mask(model):
    # Not recorded in the goldens: for a single unpadded sequence it is all
    # ones by definition.
    tok = tokenizer(model)
    for s in samples(model):
        enc = tok.encode(s["text"])
        assert enc.attention_mask == [1] * len(enc.ids), s["source"]


def test_decode(model):
    tok = tokenizer(model)
    for s in samples(model):
        assert tok.decode(s["ids"], skip_special_tokens=True) == s["decoded"], s["source"]
        assert tok.decode(s["ids"], skip_special_tokens=False) == s["decoded_with_specials"], (
            f"{s['source']} (specials)"
        )


def test_pairs(model):
    tok = tokenizer(model)
    for p in (r for r in records(model) if r["kind"] == "pair"):
        enc = tok.encode(p["text"], p["pair"])
        assert enc.ids == p["ids"], p["text"]
        assert enc.tokens == p["tokens"], p["text"]
        assert enc.type_ids == p["type_ids"], p["text"]
        assert enc.sequence_ids == p["sequence_ids"], p["text"]
        assert enc.special_tokens_mask == p["special_tokens_mask"], p["text"]
        assert [list(span) for span in enc.offsets] == p["offsets"], p["text"]


def test_fixture_digests(model):
    # Breadth: the full (capped) fixture corpora, compared as an ids digest.
    # A mismatch says the encoder diverges somewhere in that file; re-run
    # generate.py side by side to find where.
    tok = tokenizer(model)
    for d in (r for r in records(model) if r["kind"] == "digest"):
        path = DATA / d["file"]
        if not path.is_file():
            pytest.skip(f"{path} missing — {FETCH_HINT}")
        text = path.read_text()[: d["cap_chars"]]
        if text_digest(text) != d["text_sha256"]:
            pytest.skip(f"{d['file']} differs from the copy the goldens were generated from")
        ids = tok.encode(text).ids
        assert len(ids) == d["n_tokens"], d["file"]
        assert ids_digest(ids) == d["ids_sha256"], d["file"]
