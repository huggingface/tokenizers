from .. import normalizers

Normalizer = normalizers.Normalizer
BertNormalizer = normalizers.BertNormalizer
NFD = normalizers.NFD
NFKD = normalizers.NFKD
NFC = normalizers.NFC
NFKC = normalizers.NFKC
Sequence = normalizers.Sequence
Lowercase = normalizers.Lowercase
Strip = normalizers.Strip
StripAccents = normalizers.StripAccents
Nmt = normalizers.Nmt
Precompiled = normalizers.Precompiled
Replace = normalizers.Replace

try:
    opencc_enabled = normalizers.opencc_enabled
except:
    def opencc_enabled():
        return False


class NORM_OPTIONS:
    SEPARATE_INTEGERS      = 1 << 1;
    SEPARATE_SYMBOLS       = 1 << 2;
    SIMPL_TO_TRAD          = 1 << 3;
    TRAD_TO_SIMPL          = 1 << 4;
    ZH_NORM_MAPPING        = 1 << 5;

NORMALIZERS = {"nfc": NFC, "nfd": NFD, "nfkc": NFKC, "nfkd": NFKD}


def unicode_normalizer_from_str(normalizer: str) -> Normalizer:
    if normalizer not in NORMALIZERS:
        raise ValueError(
            "{} is not a known unicode normalizer. Available are {}".format(
                normalizer, NORMALIZERS.keys()
            )
        )

    return NORMALIZERS[normalizer]()
