import numpy as np
import pytest

from tokenizers import Tokenizer, TokenizersError


def test_gpt2_encodes_corpus(gpt2_file, corpus):
    tok = Tokenizer.from_file(gpt2_file)
    batch = tok.encode_batch_ids(corpus, add_special_tokens=False)
    assert sum(len(ids) for ids in batch) > 0
    assert all(ids.dtype == np.uint32 for ids in batch)


def test_bert_add_special_tokens(bert_file):
    tok = Tokenizer.from_file(bert_file)
    wrapped = tok.encode_ids("hello")  # add_special_tokens=True by default
    plain = tok.encode_ids("hello", add_special_tokens=False)
    # The post-processor wraps the content with [CLS] ... [SEP].
    assert len(wrapped) == len(plain) + 2
    assert tok.id_to_token(int(wrapped[0])) == "[CLS]"
    assert tok.id_to_token(int(wrapped[-1])) == "[SEP]"


def test_metaspace_fails_loudly_at_compile(t5_file):
    tok = Tokenizer.from_file(t5_file)
    with pytest.raises(TokenizersError, match="Metaspace"):
        tok.encode_ids("hello", add_special_tokens=False)


@pytest.mark.network
def test_from_pretrained():
    tok = Tokenizer.from_pretrained("bert-base-uncased")
    ids = tok.encode_ids("hello world", add_special_tokens=False)
    assert [tok.id_to_token(int(i)) for i in ids] == ["hello", "world"]
