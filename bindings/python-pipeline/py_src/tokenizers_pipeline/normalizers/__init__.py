"""Text cleanup that runs before the text is split."""

from .._native import normalizers as _normalizers

Normalizer = _normalizers.Normalizer
BertNormalizer = _normalizers.BertNormalizer
Lowercase = _normalizers.Lowercase
NFC = _normalizers.NFC
NFD = _normalizers.NFD
NFKC = _normalizers.NFKC
NFKD = _normalizers.NFKD
Prepend = _normalizers.Prepend
Replace = _normalizers.Replace
Sequence = _normalizers.Sequence
Strip = _normalizers.Strip
StripAccents = _normalizers.StripAccents

__all__ = [
    "Normalizer",
    "BertNormalizer",
    "Lowercase",
    "NFC",
    "NFD",
    "NFKC",
    "NFKD",
    "Prepend",
    "Replace",
    "Sequence",
    "Strip",
    "StripAccents",
]
