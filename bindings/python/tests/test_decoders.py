import json
import pickle

import pytest

from tokenizers import Regex, decoders

TEXT = "Hello world, how are you?"


# The JSON 0.23 writes for each, key order included.
@pytest.mark.parametrize(
    ("decoder", "written"),
    [
        (
            decoders.ByteLevel(),
            '{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}',
        ),
        # transformers passes these; 0.23 ignores them.
        (
            decoders.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False),
            '{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}',
        ),
        (
            decoders.Metaspace(),
            '{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true}',
        ),
        (
            decoders.Metaspace(replacement="_", prepend_scheme="first", split=False),
            '{"type":"Metaspace","replacement":"_","prepend_scheme":"first","split":false}',
        ),
        (decoders.WordPiece(), '{"type":"WordPiece","prefix":"##","cleanup":true}'),
        (decoders.WordPiece(prefix="@@", cleanup=False), '{"type":"WordPiece","prefix":"@@","cleanup":false}'),
        (decoders.BPEDecoder(), '{"type":"BPEDecoder","suffix":"</w>"}'),
        (decoders.BPEDecoder(suffix="_"), '{"type":"BPEDecoder","suffix":"_"}'),
        (decoders.Replace("▁", " "), '{"type":"Replace","pattern":{"String":"▁"},"content":" "}'),
        (
            decoders.Replace(pattern=Regex(r"\s+"), content=" "),
            '{"type":"Replace","pattern":{"Regex":"\\\\s+"},"content":" "}',
        ),
        (decoders.Strip(), '{"type":"Strip","content":" ","start":0,"stop":0}'),
        (decoders.Strip(" ", 1, 0), '{"type":"Strip","content":" ","start":1,"stop":0}'),
        (decoders.Strip(content="x", left=2, right=3), '{"type":"Strip","content":"x","start":2,"stop":3}'),
    ],
    ids=lambda param: type(param).__name__ if isinstance(param, decoders.Decoder) else None,
)
def test_decoder_json(wiki_spec, decoder, written):
    assert json.loads(decoder.__getstate__()) == json.loads(written)
    twin = pickle.loads(pickle.dumps(decoder))
    assert type(twin) is type(decoder)
    tokenizer = wiki_spec
    tokenizer.decoder = decoder
    assert type(tokenizer.decoder) is type(decoder)
    compiled = tokenizer.compile()
    compiled.decode(compiled.encode(TEXT).ids)


def test_metaspace_refuses_an_unknown_prepend_scheme():
    with pytest.raises(ValueError, match="bogus"):
        decoders.Metaspace(prepend_scheme="bogus")


# transformers tests set it to compare decoded text with and without cleanup.
def test_wordpiece_cleanup_writes_through(bert_spec):
    tokenizer = bert_spec
    tokenizer.decoder = decoders.WordPiece()
    assert tokenizer.decoder.cleanup is True
    tokenizer.decoder.cleanup = False
    assert tokenizer.decoder.cleanup is False
    assert json.loads(tokenizer.to_str())["decoder"]["cleanup"] is False


# 0.23 writes each of these as its type alone.
@pytest.mark.parametrize(
    ("key", "cls"),
    [
        ("decoder", decoders.ByteFallback),
        ("decoder", decoders.Fuse),
    ],
    ids=lambda param: param if isinstance(param, str) else param.__name__,
)
def test_stateless_component(wiki_spec, key, cls):
    tokenizer = wiki_spec
    component = cls()
    assert json.loads(component.__getstate__()) == {"type": cls.__name__}
    assert type(pickle.loads(pickle.dumps(component))) is cls
    setattr(tokenizer, key, component)
    assert type(getattr(tokenizer, key)) is cls
    compiled = tokenizer.compile()
    compiled.decode(compiled.encode(TEXT).ids)


# The JSON 0.23 writes for each.
@pytest.mark.parametrize(
    ("key", "sequence", "written"),
    [
        (
            "decoder",
            decoders.Sequence([decoders.ByteFallback(), decoders.Fuse()]),
            '{"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Fuse"}]}',
        ),
    ],
    ids=["decoders"],
)
def test_sequence_json(wiki_spec, key, sequence, written):
    assert json.loads(sequence.__getstate__()) == json.loads(written)
    assert type(pickle.loads(pickle.dumps(sequence))) is type(sequence)
    tokenizer = wiki_spec
    setattr(tokenizer, key, sequence)
    assert type(getattr(tokenizer, key)) is type(sequence)
    compiled = tokenizer.compile()
    compiled.decode(compiled.encode(TEXT).ids)
