from typing import Tuple

class PostProcessor:
    """ Base class for all post-processors

    This class is not supposed to be instantiated directly. Instead, any implementation of
    a PostProcessor will return an instance of this class when instantiated.
    """

    def num_special_tokens_to_add(self, is_pair: bool) -> int:
        """
        Return the number of special tokens that would be added for single/pair sentences.
        :param is_pair: Boolean indicating if the input would be a single sentence or a pair
        :return:
        """
        pass

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

    It also takes care of trimming the offsets.
    By default, the ByteLevel BPE might include whitespaces in the produced tokens. If you don't
    want the offsets to include these whitespaces, then this PostProcessor should be initialized
    with `trim_offsets=True`
    """

    def __init__(
        self,
        sep: Tuple[str, int],
        cls: Tuple[str, int],
        trim_offsets: bool = True,
        add_prefix_space: bool = True,
    ) -> None:
        """ Instantiate a new RobertaProcessing with the given tokens

        Args:
            sep: Tuple[str, int]:
                A tuple with the string representation of the SEP token, and its id

            cls: Tuple[str, int]:
                A tuple with the string representation of the CLS token, and its id

            trim_offsets: bool:
                Whether to trim the whitespaces from the produced offsets.

            add_prefix_space: bool:
                Whether the add_prefix_space option was enabled during pre-tokenization. This
                is relevant because it defines the way the offsets are trimmed out.

        Returns:
            PostProcessor
        """
        pass

class ByteLevel(PostProcessor):
    """ ByteLevel Post processing

    This post-processor takes care of trimming the offsets.
    By default, the ByteLevel BPE might include whitespaces in the produced tokens. If you don't
    want the offsets to include these whitespaces, then this PostProcessor must be used.
    """

    def __init(self, trim_offsets: bool = True) -> None:
        """ Instantiate a new ByteLevel

        Args:
            trim_offsets: bool:
                Whether to trim the whitespaces from the produced offsets.
        """
        pass
