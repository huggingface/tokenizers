from typing import Tuple

class PostProcessor:
    """ Base class for all post-processors

    This class is not supposed to be instantiated directly. Instead, any implementation of
    a PostProcessor will return an instance of this class when instantiated.
    """

class BertProcessing(PostProcessor):
    """ BertProcessing

    This post-processor takes care of adding the special tokens needed by
    a Bert model:
        - a SEP token
        - a CLS token
    """

    def __init__(self, sep: Tuple[str, int], cls: Tuple[str, int]) -> None:
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

class RobertaProcessing(PostProcessor):
    """ RobertaProcessing

    This post-processor takes care of adding the special tokens needed by
    a Roberta model:
        - a SEP token
        - a CLS token
    """

    def __init__(self, sep: Tuple[str, int], cls: Tuple[str, int]) -> None:
        """ Instantiate a new RobertaProcessing with the given tokens

        Args:
            sep: Tuple[str, int]:
                A tuple with the string representation of the SEP token, and its id

            cls: Tuple[str, int]:
                A tuple with the string representation of the CLS token, and its id

        Returns:
            PostProcessor
        """
        pass
