from typing import List, Optional

class Model:
    """ Base class for all models

    This class is not supposed to be instantiated directly. Instead, any implementation of
    a Model will return a instance of this class when instantiated.
    """

    def save(self, folder: str, name: str) -> List[str]:
        """ Save the current model

        Save the current model in the given folder, using the given name for the various
        files that will get created.
        Any file with the same name that already exist in this folder will be overwritten.
        """
        pass


class BPE(Model):
    """ BytePairEncoding model class """

    @staticmethod
    def from_files(vocab: str,
                   merges: str,
                   cache_capacity: Optional[int],
                   dropout: Optional[float],
                   unk_token: Optional[str],
                   continuing_subword_prefix: Optional[str],
                   end_of_word_suffix: Optional[str]) -> Model:
        """ Instantiate a BPE Model from the given vocab and merges files.

        Args:
            vocab: string:
                Path to a vocabulary JSON file.

            merges: string:
                Path to a merge file.

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
        """
        pass

    @staticmethod
    def empty() -> Model:
        """ Instantiate an empty BPE Model. """
        pass


class WordPiece(Model):
    """ WordPiece model class """

    @staticmethod
    def from_files(vocab: str,
                   unk_token: Optional[str],
                   max_input_chars_per_word: Optional[int]) -> Model:
        """ Instantiate a WordPiece Model from the given vocab file.

        Args:
            vocab: string:
                Path to a vocabulary file.

            unk_token: (`optional`) str:
                The unknown token to be used by the model.

            max_input_chars_per_word: (`optional`) int:
                The maximum number of characters to authorize in a single word.
        """
        pass

    @staticmethod
    def empty() -> Model:
        """ Instantiate an empty WordPiece Model. """
        pass


class WordLevel(Model):
    """
    Most simple tokenizer model based on mapping token from a vocab file to their corresponding id.
    """

    @staticmethod
    def from_files(vocab: str, unk_token: str) -> Model:
        """ Instantiate a WordLevel Model from the given vocab file.

        Args:
            vocab: string:
                Path to a vocabulary file.

            unk_token: str:
                The unknown token to be used by the model.
        """
        pass