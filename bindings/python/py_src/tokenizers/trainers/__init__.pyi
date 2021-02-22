# Generated content DO NOT EDIT
class Trainer:
    """
    Base class for all trainers

    This class is not supposed to be instantiated directly. Instead, any implementation of a
    Trainer will return an instance of this class when instantiated.
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
    Trainer capable of training a BPE model

    Args:
        vocab_size (:obj:`int`, `optional`):
            The size of the final vocabulary, including all tokens and alphabet.

        min_frequency (:obj:`int`, `optional`):
            The minimum frequency a pair should have in order to be merged.

        show_progress (:obj:`bool`, `optional`):
            Whether to show progress bars while training.

        special_tokens (:obj:`List[Union[str, AddedToken]]`, `optional`):
            A list of special tokens the model should know of.

        limit_alphabet (:obj:`int`, `optional`):
            The maximum different characters to keep in the alphabet.

        initial_alphabet (:obj:`List[str]`, `optional`):
            A list of characters to include in the initial alphabet, even
            if not seen in the training dataset.
            If the strings contain more than one character, only the first one
            is kept.

        continuing_subword_prefix (:obj:`str`, `optional`):
            A prefix to be used for every subword that is not a beginning-of-word.

        end_of_word_suffix (:obj:`str`, `optional`):
            A suffix to be used for every subword that is a end-of-word.
    """

class UnigramTrainer(Trainer):
    """
    Trainer capable of training a Unigram model

    Args:
        vocab_size (:obj:`int`):
            The size of the final vocabulary, including all tokens and alphabet.

        show_progress (:obj:`bool`):
            Whether to show progress bars while training.

        special_tokens (:obj:`List[Union[str, AddedToken]]`):
            A list of special tokens the model should know of.

        initial_alphabet (:obj:`List[str]`):
            A list of characters to include in the initial alphabet, even
            if not seen in the training dataset.
            If the strings contain more than one character, only the first one
            is kept.
    """

    def __init__(self, vocab_size=8000, show_progress=True, special_tokens=[]):
        pass

class WordLevelTrainer(Trainer):
    """
    Trainer capable of training a WorldLevel model

    Args:
        vocab_size (:obj:`int`, `optional`):
            The size of the final vocabulary, including all tokens and alphabet.

        min_frequency (:obj:`int`, `optional`):
            The minimum frequency a pair should have in order to be merged.

        show_progress (:obj:`bool`, `optional`):
            Whether to show progress bars while training.

        special_tokens (:obj:`List[Union[str, AddedToken]]`):
            A list of special tokens the model should know of.
    """

class WordPieceTrainer(Trainer):
    """
    Trainer capable of training a WordPiece model

    Args:
        vocab_size (:obj:`int`, `optional`):
            The size of the final vocabulary, including all tokens and alphabet.

        min_frequency (:obj:`int`, `optional`):
            The minimum frequency a pair should have in order to be merged.

        show_progress (:obj:`bool`, `optional`):
            Whether to show progress bars while training.

        special_tokens (:obj:`List[Union[str, AddedToken]]`, `optional`):
            A list of special tokens the model should know of.

        limit_alphabet (:obj:`int`, `optional`):
            The maximum different characters to keep in the alphabet.

        initial_alphabet (:obj:`List[str]`, `optional`):
            A list of characters to include in the initial alphabet, even
            if not seen in the training dataset.
            If the strings contain more than one character, only the first one
            is kept.

        continuing_subword_prefix (:obj:`str`, `optional`):
            A prefix to be used for every subword that is not a beginning-of-word.

        end_of_word_suffix (:obj:`str`, `optional`):
            A suffix to be used for every subword that is a end-of-word.
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
