import numpy as np
import pytest

from tokenizers import Tokenizer, TokenizersError


def test_gpt2_encodes_corpus(gpt2_file, corpus):
    tok = Tokenizer.from_file(gpt2_file)
    batch = tok.encode_batch_ids(corpus, add_special_tokens=False)
    assert sum(len(ids) for ids in batch) > 0
    assert all(ids.dtype == np.uint32 for ids in batch)


def test_bert_special_tokens_gate(bert_file):
    tok = Tokenizer.from_file(bert_file)
    with pytest.raises(NotImplementedError, match="post-process"):
        tok.encode_ids("hello")
    ids = tok.encode_ids("hello", add_special_tokens=False)
    assert len(ids) > 0


def test_metaspace_fails_loudly_at_compile(t5_file):
    tok = Tokenizer.from_file(t5_file)
    with pytest.raises(TokenizersError, match="Metaspace"):
        tok.encode_ids("hello", add_special_tokens=False)


@pytest.mark.network
def test_from_pretrained():
    tok = Tokenizer.from_pretrained("bert-base-uncased")
    ids = tok.encode_ids("hello world", add_special_tokens=False)
    assert [tok.id_to_token(int(i)) for i in ids] == ["hello", "world"]
