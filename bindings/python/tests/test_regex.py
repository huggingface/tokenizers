import json

import pytest

from tokenizers import Regex, Tokenizer


def test_regex_refuses_a_pattern_that_does_not_compile():
    with pytest.raises(ValueError):
        Regex("(")


def _split_on_word_regex():
    spec = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Split",
            "pattern": {"Regex": r"\w+"},
            "behavior": "Removed",
            "invert": False,
        },
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "a": 1}, "unk_token": "[UNK]"},
    }
    return Tokenizer.from_str(json.dumps(spec)).compile()


def test_word_regex_matches_the_0_23_character_set():
    r"""Regression for #2742: 0.23's `\w` and UTS#18's differ on eight code points."""
    tokenizer = _split_on_word_regex()

    # Oniguruma (0.23) treated these as word characters; a Split with
    # behavior=Removed must remove them along with the surrounding `a`s.
    for c in "²³¹¼½¾":
        assert tokenizer.encode("a" + c + "a").ids == [], f"U+{ord(c):04X} must be a word char"

    # ZWNJ/ZWJ were not word characters in 0.23, so the removed `\w+` spans
    # leave the join control as the single remaining (unknown) token.
    for c in "\u200c\u200d":
        assert tokenizer.encode("a" + c + "a").ids == [0], f"U+{ord(c):04X} must not be a word char"
