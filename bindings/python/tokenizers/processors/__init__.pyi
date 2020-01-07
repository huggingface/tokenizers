from typing import Tuple

class PostProcessor:
    """ Base class for all post-processors

    This class is not supposed to be instantiated directly. Instead, any implementation of
    a PostProcessor will return an instance of this class when instantiated.
    """

class BertProcessing:
    """ BertProcessing

    This post-processor takes care of adding the special tokens needed by
    a Bert model:
        - a SEP token
        - a CLS token
    """

    @staticmethod
    def new(sep: Tuple[str, int], cls: Tuple[str, int]) -> PostProcessor:
        """ Instantiate a new BertProcessing with the given tokens

        Args:
            sep: Tuple[str, int]:
                A tuple with the string representation of the SEP token, and its id

            cls: Tuple[str, int]:
                A tuple with the string representation of the CLS token, and its id

        Returns:
            PostProcessor
        """
        pass
