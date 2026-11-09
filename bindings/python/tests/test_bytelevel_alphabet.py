from tokenizers import normalizers


def test_bytelevel_alphabet_covers_every_byte():
    alphabet = normalizers.ByteLevel.alphabet()
    assert isinstance(alphabet, list)
    assert len(alphabet) == 256
    assert len(set(alphabet)) == 256
    assert all(isinstance(character, str) and len(character) == 1 for character in alphabet)
    # The printable byte ranges keep their Unicode codepoints. The remaining
    # bytes map above Latin-1, including the familiar space and newline markers.
    visible = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    expected = {chr(value) for value in visible} | {chr(256 + offset) for offset in range(256 - len(visible))}
    assert set(alphabet) == expected
    assert {"Ġ", "Ċ", "a", "!", "é"} <= set(alphabet)


def test_bytelevel_alphabet_is_static_and_returns_an_independent_list():
    alphabet = normalizers.ByteLevel.alphabet()
    via_instance = normalizers.ByteLevel().alphabet()
    assert set(via_instance) == set(alphabet)
    alphabet.clear()
    assert len(normalizers.ByteLevel.alphabet()) == 256
