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


# Loading a file takes over a second on the debug build `make develop` installs, so the tests share
# one tokenizer per file. The padding is its only mutable state; each test starts from none.
@pytest.fixture(scope="session")
def shared_gpt2():
    return Tokenizer.from_file(GPT2)


@pytest.fixture(scope="session")
def shared_bert():
    return Tokenizer.from_file(BERT)


@pytest.fixture(scope="session")
def shared_wiki():
    return Tokenizer.from_file(WIKI)


@pytest.fixture
def gpt2(shared_gpt2):
    yield shared_gpt2
    shared_gpt2.padding = None


@pytest.fixture
def bert(shared_bert):
    yield shared_bert
    shared_bert.padding = None


@pytest.fixture
def wiki(shared_wiki):
    yield shared_wiki
    shared_wiki.padding = None


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
