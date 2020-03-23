from .. import AddedToken
from typing import Optional, List, Union

class Trainer:
    """ Base class for all trainers

    This class is not supposed to be instantiated directly. Instead, any implementation of a
    Trainer will return an instance of this class when instantiated.
    """

class BpeTrainer(Trainer):
    """ BpeTrainer

    Capable of training a BPE model
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        min_frequency: int = 0,
        show_progress: bool = True,
        special_tokens: List[Union[str, AddedToken]] = [],
        limit_alphabet: Optional[int] = None,
        initial_alphabet: List[str] = [],
        continuing_subword_prefix: Optional[str] = None,
        end_of_word_suffix: Optional[str] = None,
    ) -> None:
        """ Instantiate a new BpeTrainer with the given options:

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
                If the strings contains more than one character, only the first one
                is kept.

            continuing_subword_prefix: Optional[str]:
                A prefix to be used for every subword that is not a beginning-of-word.

            end_of_word_suffix: Optional[str]:
                A suffix to be used for every subword that is a end-of-word.

        Returns:
            Trainer
        """
        pass

class WordPieceTrainer(Trainer):
    """ WordPieceTrainer

    Capable of training a WordPiece model
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        min_frequency: int = 0,
        show_progress: bool = True,
        special_tokens: List[Union[str, AddedToken]] = [],
        limit_alphabet: Optional[int] = None,
        initial_alphabet: List[str] = [],
        continuing_subword_prefix: Optional[str] = "##",
        end_of_word_suffix: Optional[str] = None,
    ) -> Trainer:
        """ Instantiate a new WordPieceTrainer with the given options:

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
                If the strings contains more than one character, only the first one
                is kept.

            continuing_subword_prefix: Optional[str]:
                A prefix to be used for every subword that is not a beginning-of-word.

            end_of_word_suffix: Optional[str]:
                A suffix to be used for every subword that is a end-of-word.

        Returns:
            Trainer
        """
        pass
