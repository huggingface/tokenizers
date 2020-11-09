# Generated content DO NOT EDIT
class Trainer:
    """
    Base class for all trainers

    This class is not supposed to be instantiated directly. Instead, any implementation of a
    Trainer will return an instance of this class when instantiated.

    Args:
        vocab_size: unsigned int:
            The size of the final vocabulary, including all tokens and alphabet.

        min_frequency: unsigned int:
            The minimum frequency a pair should have in order to be merged.

        show_progress: boolean:
            Whether to show progress bars while training.

        special_tokens: List[Union[str, AddedToken]]:
            A list of special tokens the model should know of.

        limit_alphabet: unsigned int:
            The maximum different characters to keep in the alphabet.

        initial_alphabet: List[str]:
            A list of characters to include in the initial alphabet, even
            if not seen in the training dataset.
            If the strings contain more than one character, only the first one
            is kept.

        continuing_subword_prefix: Optional[str]:
            A prefix to be used for every subword that is not a beginning-of-word.

        end_of_word_suffix: Optional[str]:
            A suffix to be used for every subword that is a end-of-word.

    Returns:
        Trainer
    """

    def __init__(
        self,
        vocab_size=30000,
        min_frequency=0,
        show_progress=True,
        special_tokens=[],
        limit_alphabet=None,
        initial_alphabet=[],
        continuing_subword_prefix=None,
        end_of_word_suffix=None,
    ):
        pass

class BpeTrainer(Trainer):
    """
    Capable of training a BPE model
    """

class UnigramTrainer(Trainer):
    """
    Capable of training a Unigram model

    Args:
        vocab_size: unsigned int:
            The size of the final vocabulary, including all tokens and alphabet.

        show_progress: boolean:
            Whether to show progress bars while training.

        special_tokens: List[Union[str, AddedToken]]:
            A list of special tokens the model should know of.

        initial_alphabet: List[str]:
            A list of characters to include in the initial alphabet, even
            if not seen in the training dataset.
            If the strings contain more than one character, only the first one
            is kept.

    Returns:
        Trainer
    """

    def __init__(self, vocab_size=8000, show_progress=True, special_tokens=[]):
        pass

class WordPieceTrainer(Trainer):
    """
    Capable of training a WordPiece model
    Args:
        vocab_size: unsigned int:
            The size of the final vocabulary, including all tokens and alphabet.

        min_frequency: unsigned int:
            The minimum frequency a pair should have in order to be merged.

        show_progress: boolean:
            Whether to show progress bars while training.

        special_tokens: List[Union[str, AddedToken]]:
            A list of special tokens the model should know of.

        limit_alphabet: unsigned int:
            The maximum different characters to keep in the alphabet.

        initial_alphabet: List[str]:
            A list of characters to include in the initial alphabet, even
            if not seen in the training dataset.
            If the strings contain more than one character, only the first one
            is kept.

        continuing_subword_prefix: Optional[str]:
            A prefix to be used for every subword that is not a beginning-of-word.

        end_of_word_suffix: Optional[str]:
            A suffix to be used for every subword that is a end-of-word.

    Returns:
        Trainer
    """

    def __init__(
        self,
        vocab_size=30000,
        min_frequency=0,
        show_progress=True,
        special_tokens=[],
        limit_alphabet=None,
        initial_alphabet=[],
        continuing_subword_prefix="##",
        end_of_word_suffix=None,
    ):
        pass
