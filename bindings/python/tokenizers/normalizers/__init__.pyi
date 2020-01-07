from typing import Optional

class Normalizer:
    """ Base class for all normalizers

    This class is not supposed to be instantiated directly. Instead, any implementation of a
    Normalizer will return an instance of this class when instantiated.
    """

class BertNormalizer:
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
