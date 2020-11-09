# Generated content DO NOT EDIT
class Normalizer:
    """
    Base class for all normalizers

    This class is not supposed to be instantiated directly. Instead, any implementation of a
    Normalizer will return an instance of this class when instantiated.
    """

    def normalize(self, normalized):
        """
        Normalize the given NormalizedString in-place
        """
        pass
    def normalize_str(self, sequence):
        """
        Normalize the given str
        """
        pass

class BertNormalizer(Normalizer):
    """
    BertNormalizer

    Takes care of normalizing raw text before giving it to a Bert model.
    This includes cleaning the text, handling accents, chinese chars and lowercasing

    Args:
        clean_text: (`optional`) boolean:
            Whether to clean the text, by removing any control characters
            and replacing all whitespaces by the classic one.

        handle_chinese_chars: (`optional`) boolean:
            Whether to handle chinese chars by putting spaces around them.

        strip_accents: (`optional`) boolean:
            Whether to strip all accents. If this option is not specified (ie == None),
            then it will be determined by the value for `lowercase` (as in the original Bert).

        lowercase: (`optional`) boolean:
            Whether to lowercase.

    Returns:
        Normalizer
    """

    def __init__(
        self, clean_text=True, handle_chinese_chars=True, strip_accents=None, lowercase=True
    ):
        pass
    def normalize(self, normalized):
        """
        Normalize the given NormalizedString in-place
        """
        pass
    def normalize_str(self, sequence):
        """
        Normalize the given str
        """
        pass

class Lowercase(Normalizer):
    """
    Lowercase Normalizer
    """

    def __init__(self):
        pass
    def normalize(self, normalized):
        """
        Normalize the given NormalizedString in-place
        """
        pass
    def normalize_str(self, sequence):
        """
        Normalize the given str
        """
        pass

class NFC(Normalizer):
    """
    NFC Unicode Normalizer
    """

    def __init__(self):
        pass
    def normalize(self, normalized):
        """
        Normalize the given NormalizedString in-place
        """
        pass
    def normalize_str(self, sequence):
        """
        Normalize the given str
        """
        pass

class NFD(Normalizer):
    """
    NFD Unicode Normalizer
    """

    def __init__(self):
        pass
    def normalize(self, normalized):
        """
        Normalize the given NormalizedString in-place
        """
        pass
    def normalize_str(self, sequence):
        """
        Normalize the given str
        """
        pass

class NFKC(Normalizer):
    """
    NFKC Unicode Normalizer
    """

    def __init__(self):
        pass
    def normalize(self, normalized):
        """
        Normalize the given NormalizedString in-place
        """
        pass
    def normalize_str(self, sequence):
        """
        Normalize the given str
        """
        pass

class NFKD(Normalizer):
    """
    NFKD Unicode Normalizer
    """

    def __init__(self):
        pass
    def normalize(self, normalized):
        """
        Normalize the given NormalizedString in-place
        """
        pass
    def normalize_str(self, sequence):
        """
        Normalize the given str
        """
        pass

class Nmt(Normalizer):
    """
    Nmt normalizer
    """

    def __init__(self):
        pass
    def normalize(self, normalized):
        """
        Normalize the given NormalizedString in-place
        """
        pass
    def normalize_str(self, sequence):
        """
        Normalize the given str
        """
        pass

class Precompiled(Normalizer):
    """
    Precompiled normalizer
    Don't use manually it is used for compatiblity for SentencePiece.
    """

    def __init__(self, precompiled_charsmap):
        pass
    def normalize(self, normalized):
        """
        Normalize the given NormalizedString in-place
        """
        pass
    def normalize_str(self, sequence):
        """
        Normalize the given str
        """
        pass

class Replace(Normalizer):
    """
    Replace normalizer
    """

    def __init__(self, pattern, content):
        pass
    def normalize(self, normalized):
        """
        Normalize the given NormalizedString in-place
        """
        pass
    def normalize_str(self, sequence):
        """
        Normalize the given str
        """
        pass

class Sequence(Normalizer):
    """
    Allows concatenating multiple other Normalizer as a Sequence.
    All the normalizers run in sequence in the given order

    Args:
        normalizers: List[Normalizer]:
            A list of Normalizer to be run as a sequence
    """

    def normalize(self, normalized):
        """
        Normalize the given NormalizedString in-place
        """
        pass
    def normalize_str(self, sequence):
        """
        Normalize the given str
        """
        pass

class Strip(Normalizer):
    """
    Strip normalizer
    """

    def __init__(self, left=True, right=True):
        pass
    def normalize(self, normalized):
        """
        Normalize the given NormalizedString in-place
        """
        pass
    def normalize_str(self, sequence):
        """
        Normalize the given str
        """
        pass

class StripAccents(Normalizer):
    def __init__(self):
        pass
    def normalize(self, normalized):
        """
        Normalize the given NormalizedString in-place
        """
        pass
    def normalize_str(self, sequence):
        """
        Normalize the given str
        """
        pass
