from typing import Optional, List

class Normalizer:
    """ Base class for all normalizers

    This class is not supposed to be instantiated directly. Instead, any implementation of a
    Normalizer will return an instance of this class when instantiated.
    """

class BertNormalizer(Normalizer):
    """ BertNormalizer

    Takes care of normalizing raw text before giving it to a Bert model.
    This includes cleaning the text, handling accents, chinese chars and lowercasing
    """

    def __init__(
        self,
        clean_text: Optional[bool] = True,
        handle_chinese_chars: Optional[bool] = True,
        strip_accents: Optional[bool] = True,
        lowercase: Optional[bool] = True,
    ) -> None:
        """ Instantiate a BertNormalizer with the given options.

        Args:
            clean_text: (`optional`) boolean:
                Whether to clean the text, by removing any control characters
                and replacing all whitespaces by the classic one.

            handle_chinese_chars: (`optional`) boolean:
                Whether to handle chinese chars by putting spaces around them.

            strip_accents: (`optional`) boolean:
                Whether to strip all accents.

            lowercase: (`optional`) boolean:
                Whether to lowercase.

        Returns:
            Normalizer
        """
        pass

class NFD(Normalizer):
    """ NFD Unicode Normalizer """

    def __init__(self) -> None:
        """ Instantiate a new NFD Normalizer """
        pass

class NFKD(Normalizer):
    """ NFKD Unicode Normalizer """

    def __init__(self) -> None:
        """ Instantiate a new NFKD Normalizer """
        pass

class NFC(Normalizer):
    """ NFC Unicode Normalizer """

    def __init__(self) -> None:
        """ Instantiate a new NFC Normalizer """
        pass

class NFKC(Normalizer):
    """ NFKC Unicode Normalizer """

    def __init__(self) -> None:
        """ Instantiate a new NFKC Normalizer """
        pass

class Sequence(Normalizer):
    """ Allows concatenating multiple other Normalizer as a Sequence.

    All the normalizers run in sequence in the given order
    """

    def __init__(self, normalizers: List[Normalizer]) -> None:
        """ Instantiate a new normalization Sequence using the given normalizers

        Args:
            normalizers: List[Normalizer]:
                A list of Normalizer to be run as a sequence
        """
        pass

class Lowercase(Normalizer):
    """ Lowercase Normalizer """

    def __init__(self) -> None:
        """ Instantiate a new Lowercase Normalizer """
        pass

class Strip(Normalizer):
    """ Strip normalizer """

    def __init__(self, left: bool = True, right: bool = True) -> Normalizer:
        pass

def unicode_normalizer_from_str(normalizer: str) -> Normalizer:
    """
    Instanciate unicode normalizer from the normalizer name
    :param normalizer: Name of the normalizer
    :return:
    """
    pass
