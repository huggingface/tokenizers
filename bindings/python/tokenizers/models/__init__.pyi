from .. import Encoding, Offsets
from typing import List, Optional, Union, Tuple

TokenizedSequence = List[str]
TokenizedSequenceWithOffsets = List[Tuple[str, Offsets]]

class Model:
    """ Base class for all models

    This class is not supposed to be instantiated directly. Instead, any implementation of
    a Model will return a instance of this class when instantiated.
    """

    def save(self, folder: str, name: Optional[str] = None) -> List[str]:
        """ Save the current model

        Save the current model in the given folder, using the given name for the various
        files that will get created.
        Any file with the same name that already exist in this folder will be overwritten.
        """
        pass
    def encode(
        self, sequence: Union[TokenizedSequence, TokenizedSequenceWithOffsets], type_id: int = 0
    ) -> Encoding:
        """ Encode the given sequence.

        A sequence can either be:
            - `TokenizedSequence`: (`List[str]`)
            - `TokenizedSequenceWithOffsets: (`List[Tuple[str, Offsets]]`) where Offsets is
            a Tuple[int, int].

        If the Offsets are not provided, they will be automatically generated, making the hypothesis
        that all the tokens in the `TokenizedSequence` are contiguous in the original string.

        Args:
            sequence: Union[TokenizedSequence, TokenizedSequenceWithOffsets]
                Either a TokenizedSequence or a TokenizedSequenceWithOffsets

            type_id: int:
                The type id of the given sequence

        Returns:
            An Encoding
        """
        pass
    def encode_batch(
        self,
        sequences: Union[List[TokenizedSequence], List[TokenizedSequenceWithOffsets]],
        type_id: int = 0,
    ) -> List[Encoding]:
        """ Encode the given batch of sequence.

        A sequence can either be:
            - `TokenizedSequence`: (`List[str]`)
            - `TokenizedSequenceWithOffsets: (`List[Tuple[str, Offsets]]`) where Offsets is
            a Tuple[int, int].

        If the Offsets are not provided, they will be automatically generated, making the hypothesis
        that all the tokens in the `TokenizedSequence` are contiguous in the original string.

        Args:
            sequences: Union[List[TokenizedSequence], List[TokenizedSequenceWithOffsets]]
                A list of sequence. Each sequence is either a TokenizedSequence or a
                TokenizedSequenceWithOffsets

            type_id: int:
                The type if of the given sequence

        Returns:
            A list of Encoding
        """
        pass

class BPE(Model):
    """BytePairEncoding model class

    Instantiate a BPE Model from the given vocab and merges files.

    Args:
       vocab: ('`optional`) string:
           Path to a vocabulary JSON file.

       merges: (`optional`) string:
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

    @staticmethod
    def __init__(
        self,
        vocab: Optional[str],
        merges: Optional[str],
        cache_capacity: Optional[int],
        dropout: Optional[float],
        unk_token: Optional[str],
        continuing_subword_prefix: Optional[str],
        end_of_word_suffix: Optional[str],
    ):
        pass

class WordPiece(Model):
    """ WordPiece model class

    Instantiate a WordPiece Model from the given vocab file.

        Args:
            vocab: (`optional`) string:
                Path to a vocabulary file.

            unk_token: (`optional`) str:
                The unknown token to be used by the model.

            max_input_chars_per_word: (`optional`) int:
                The maximum number of characters to authorize in a single word.
    """

    def __init__(
        self,
        vocab: Optional[str],
        unk_token: Optional[str],
        max_input_chars_per_word: Optional[int],
    ):
        pass

class WordLevel(Model):
    """
    Most simple tokenizer model based on mapping token from a vocab file to their corresponding id.

    Instantiate a WordLevel Model from the given vocab file.

        Args:
            vocab: (`optional`) string:
                Path to a vocabulary file.

            unk_token: str:
                The unknown token to be used by the model.
    """

    def __init__(self, vocab: Optional[str], unk_token: Optional[str]):
        pass
