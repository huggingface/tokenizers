import json
from pathlib import Path

import pytest

from tokenizers import Tokenizer

# `make test` fetches these; see the Makefile's TESTS_RESOURCES.
DATA = Path(__file__).parent.parent / "data"
# Byte-level BPE with a ByteLevel decoder, `<|endoftext|>` as its only special token.
GPT2 = DATA / "gpt2.json"
# WordPiece trained on wikitext: NFD + lowercase + strip accents, `[CLS] $A [SEP]`, no decoder.
BERT = DATA / "bert-wiki.json"
# BPE trained on wikitext with a Whitespace pre-tokenizer, no post-processor, no decoder.
WIKI = DATA / "tokenizer-wiki.json"
# Llama 3: byte-level BPE with a regex pre-tokenizer and `ignore_merges`, 256 special tokens, and a
# post-processor that prepends `<|begin_of_text|>`.
LLAMA3 = DATA / "llama-3-tokenizer.json"
# ALBERT: a normalizer Sequence of Replace, NFKD, StripAccents, Lowercase and Precompiled.
ALBERT = DATA / "albert-base-v1-tokenizer.json"


# Compiling takes over a second on the debug build `make develop` installs, so the tests share one
# compiled tokenizer per file. A compiled tokenizer cannot change, so no test can leak state into another.
@pytest.fixture(scope="session")
def gpt2():
    return Tokenizer.from_file(GPT2).compile()


@pytest.fixture(scope="session")
def bert():
    return Tokenizer.from_file(BERT).compile()


@pytest.fixture(scope="session")
def wiki():
    return Tokenizer.from_file(WIKI).compile()


@pytest.fixture(scope="session")
def llama3():
    return Tokenizer.from_file(LLAMA3).compile()


@pytest.fixture
def gpt2_spec():
    return Tokenizer.from_file(GPT2)


@pytest.fixture
def bert_spec():
    return Tokenizer.from_file(BERT)


@pytest.fixture
def wiki_spec():
    return Tokenizer.from_file(WIKI)


@pytest.fixture
def padded_wiki(tmp_path):
    declared = json.loads(WIKI.read_text())
    declared["padding"] = {
        "strategy": "BatchLongest",
        "direction": "Right",
        "pad_to_multiple_of": None,
        "pad_id": 3,
        "pad_type_id": 0,
        "pad_token": "[PAD]",
    }
    path = tmp_path / "padded.json"
    path.write_text(json.dumps(declared))
    return path


@pytest.fixture
def truncated_wiki(tmp_path):
    declared = json.loads(WIKI.read_text())
    declared["truncation"] = {
        "direction": "Right",
        "max_length": 4,
        "strategy": "LongestFirst",
        "stride": 0,
    }
    path = tmp_path / "truncated.json"
    path.write_text(json.dumps(declared))
    return path
