from tokenizers import AddedToken


def test_defaults():
    token = AddedToken("<mask>")
    assert token.content == "<mask>"
    assert token.single_word is False
    assert token.lstrip is False
    assert token.rstrip is False
    assert token.special is False
    assert token.normalized is True


def test_special_flips_normalized_default():
    assert AddedToken("<s>", special=True).normalized is False
    assert AddedToken("<s>", special=True, normalized=True).normalized is True


def test_flags():
    token = AddedToken("<mask>", single_word=True, lstrip=True, rstrip=True)
    assert token.single_word is True
    assert token.lstrip is True
    assert token.rstrip is True


def test_repr():
    assert (
        repr(AddedToken("<mask>"))
        == 'AddedToken("<mask>", single_word=False, lstrip=False, rstrip=False, normalized=True, special=False)'
    )
