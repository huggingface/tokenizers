import pytest

from tokenizers import Regex


def test_regex_refuses_a_pattern_that_does_not_compile():
    with pytest.raises(ValueError):
        Regex("(")
