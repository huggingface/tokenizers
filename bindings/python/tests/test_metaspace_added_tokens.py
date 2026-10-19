import json

import pytest

from tokenizers import Tokenizer


@pytest.mark.parametrize("prepend_scheme", ["first", "always", "never"])
@pytest.mark.parametrize("normalized", [True, False])
def test_metaspace_after_added_tokens(tmp_path, prepend_scheme, normalized):
    config = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": 1,
                "content": "<s>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": normalized,
                "special": True,
            },
            {
                "id": 7,
                "content": "<raw>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
        ],
        "normalizer": {"type": "Lowercase"},
        "pre_tokenizer": {
            "type": "Metaspace",
            "replacement": "▁",
            "prepend_scheme": prepend_scheme,
            "split": False,
        },
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "BPE",
            "unk_token": "<unk>",
            "vocab": {"<unk>": 0, "<s>": 1, "▁": 2, "▁a": 3, "a": 4, "é": 5, "▁é": 6},
            "merges": [["▁", "a"], ["▁", "é"]],
        },
    }
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    tokenizer = Tokenizer.from_file(path)
    after_token = 3 if prepend_scheme == "always" else 4
    first = 4 if prepend_scheme == "never" else 3
    after_unicode = 6 if prepend_scheme == "always" else 5
    # Recorded from 0.23.1, including controls for genuine leading whitespace.
    cases = {
        "<s> a": [1, 3],
        "<s>a": [1, after_token],
        "<s>é": [1, after_unicode],
        "<s>": [1],
        "<s><s>a": [1, 1, after_token],
        "a <s> a": [first, 2, 1, 3],
        "a<s>a": [first, 1, after_token],
        "▁<s> a": [2, 1, 3],
        " <s> a": [2, 1, 3],
        "<raw><s>a": [7, 1, after_token],
        "A": [first],
    }
    if normalized:
        cases.update({"<S> A": [1, 3], "<S>A": [1, after_token]})
    for text, expected in cases.items():
        assert tokenizer.encode(text).ids == expected, text
        assert tokenizer.encode_batch([text])[0].ids == expected, text
    # Cross the parallel planner's threshold with a raw and a normalized token boundary.
    long = "a" * (16 * 1024)
    expected = [first] + [4] * (len(long) - 1) + [7, 1, after_token]
    assert tokenizer.encode(long + "<raw><s>a").ids == expected
    assert tokenizer.encode_batch([long + "<raw><s>a"])[0].ids == expected
