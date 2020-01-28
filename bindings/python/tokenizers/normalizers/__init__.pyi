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

    @staticmethod
    def new(clean_text: Optional[bool]=True,
            handle_chinese_chars: Optional[bool]=True,
            strip_accents: Optional[bool]=True,
            lowercase: Optional[bool]=True) -> Normalizer:
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

    @staticmethod
    def new() -> Normalizer:
        """ Instantiate a new NFD Normalizer """
        pass

class NFKD(Normalizer):
    """ NFKD Unicode Normalizer """

    @staticmethod
    def new() -> Normalizer:
        """ Instantiate a new NFKD Normalizer """
        pass

class NFC(Normalizer):
    """ NFC Unicode Normalizer """

    @staticmethod
    def new() -> Normalizer:
        """ Instantiate a new NFC Normalizer """
        pass

class NFKC(Normalizer):
    """ NFKC Unicode Normalizer """

    @staticmethod
    def new() -> Normalizer:
        """ Instantiate a new NFKC Normalizer """
        pass

class Sequence(Normalizer):
    """ Allows concatenating multiple other Normalizer as a Sequence.

    All the normalizers run in sequence in the given order
    """

    @staticmethod
    def new(normalizers: List[Normalizer]) -> Normalizer:
        """ Instantiate a new normalization Sequence using the given normalizers

        Args:
            normalizers: List[Normalizer]:
                A list of Normalizer to be run as a sequence
        """
        pass

class Lowercase(Normalizer):
    """ Lowercase Normalizer """

    @staticmethod
    def new() -> Normalizer:
        """ Instantiate a new Lowercase Normalizer """
        pass


def unicode_normalizer_from_str(normalizer: str) -> Normalizer:
    """
    Instanciate unicode normalizer from the normalizer name
    :param normalizer: Name of the normalizer
    :return:
    """
    pass