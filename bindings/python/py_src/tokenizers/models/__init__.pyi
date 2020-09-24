from .. import Encoding, Offsets, Token
from typing import List, Optional, Union, Tuple, Dict

class Model:
    """Base class for all models

    This class is not supposed to be instantiated directly. Instead, any implementation of
    a Model will return a instance of this class when instantiated.
    """

    def tokenize(self, sequence: str) -> List[Token]:
        """ Tokenize the given sequence """
        pass
    def token_to_id(self, token: str) -> Optional[int]:
        """ Returns the id associated with the given token """
        pass
    def id_to_token(self, id: int) -> Optional[str]:
        """ Returns the token associated with the given id """
        pass
    def save(self, folder: str, name: Optional[str] = None) -> List[str]:
        """Save the current model

        Save the current model in the given folder, using the given name for the various
        files that will get created.
        Any file with the same name that already exist in this folder will be overwritten.
        """
        pass

class BPE(Model):
    """BytePairEncoding model class

    Instantiate a BPE Model from the given vocab and merges.

    Args:
       vocab: ('`optional`) Dict[str, int]:
           A dictionnary of string keys and their ids {"am": 0,...}

       merges: (`optional`) string:
           A list of pairs of tokens [("a", "b"),...]

       cache_capacity: (`optional`) int:
           The number of words that the BPE cache can contain. The cache allows
           to speed-up the process by keeping the result of the merge operations
           for a number of words.

       dropout: (`optional`) Optional[float] [0, 1]:
           The BPE dropout to use. Must be an float between 0 and 1

       unk_token: (`optional`) str:
           The unknown token to be used by the model.

       continuing_subword_prefix: (`optional`) str:
           The prefix to attach to subword units that don't represent a beginning of word.

       end_of_word_suffix: (`optional`) str:
           The suffix to attach to subword units that represent an end of word.

       fuse_unk: (`optional`) bool:
           Multiple unk tokens get fused into only 1
    """

    def __init__(
        self,
        vocab: Optional[Union[str, Dict[str, int]]],
        merges: Optional[Union[str, List[Tuple[str, str]]]],
        cache_capacity: Optional[int],
        dropout: Optional[float],
        unk_token: Optional[str],
        continuing_subword_prefix: Optional[str],
        end_of_word_suffix: Optional[str],
        fuse_unk: Optional[bool],
    ):
        pass
    @staticmethod
    def read_file(vocab_filename: str, merges_filename: str) -> Tuple[Vocab, Merges]:
        pass
    @staticmethod
    def from_file(vocab_filename: str, merges_filename: str, **kwargs) -> BPE:
        """
        Convenient method to intialize a BPE from files
        Roughly equivalent to

        def from_file(vocab_filename, merges_filenames, **kwargs):
            vocab, merges = BPE.read_file(vocab_filename, merges_filename)
            return BPE(vocab, merges, **kwargs)
        """
        pass

class WordPiece(Model):
    """WordPiece model class

    Instantiate a WordPiece Model from the given vocab file.

        Args:
            vocab: (`optional`) string:
                A dictionnary of string keys and their ids {"am": 0,...}

            unk_token: (`optional`) str:
                The unknown token to be used by the model.

            max_input_chars_per_word: (`optional`) int:
                The maximum number of characters to authorize in a single word.
    """

    def __init__(
        self,
        vocab: Optional[Union[str, Dict[str, int]]],
        unk_token: Optional[str],
        max_input_chars_per_word: Optional[int],
    ):
        pass
    @staticmethod
    def read_file(vocab_filename: str) -> Vocab:
        pass
    @staticmethod
    def from_file(vocab_filename: str, **kwargs) -> WordPiece:
        """
        Convenient method to intialize a WordPiece from file
        Roughly equivalent to

        def from_file(vocab_filename, **kwargs):
            vocab, merges = WordPiece.read_file(vocab_filename)
            return WordPiece(vocab, **kwargs)
        """
        pass

class WordLevel(Model):
    """
    Most simple tokenizer model based on mapping token from a vocab file to their corresponding id.

    Instantiate a WordLevel Model from the given vocab file.

        Args:
            vocab: (`optional`) string:
                A dictionnary of string keys and their ids {"am": 0,...}

            unk_token: str:
                The unknown token to be used by the model.
    """

    def __init__(self, vocab: Optional[Union[str, Dict[str, int]]], unk_token: Optional[str]):
        pass
    @staticmethod
    def read_file(vocab_filename: str) -> Vocab:
        pass
    @staticmethod
    def from_file(vocab_filename: str, **kwargs) -> WordLevelg:
        """
        Convenient method to intialize a WordLevelg from file
        Roughly equivalent to

        def from_file(vocab_filename, **kwargs):
            vocab, merges = WordLevelg.read_file(vocab_filename)
            return WordLevelg(vocab, **kwargs)
        """
        pass

class Unigram(Model):
    """UnigramEncoding model class

    Instantiate a Unigram Model from the given model file.

    Args:
       vocab: ('`optional`) string:
           A list of vocabulary items and their relative score [("am", -0.2442),...]

    """

    @staticmethod
    def __init__(self, vocab: Optional[List[Tuple[str, float]]]):
        pass
