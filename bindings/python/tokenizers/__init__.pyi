from .decoders import *
from .models import *
from .normalizers import *
from .pre_tokenizers import *
from .processors import *
from .trainers import *

from .implementations import (
    ByteLevelBPETokenizer as ByteLevelBPETokenizer,
    BPETokenizer as BPETokenizer,
    SentencePieceBPETokenizer as SentencePieceBPETokenizer,
    BertWordPieceTokenizer as BertWordPieceTokenizer
)

from typing import Optional, Union, List, Tuple

Offsets = Tuple[int, int]

class IndexableString:
    """
    Works almost like a `str`, but allows indexing on offsets
    provided on an `Encoding`
    """

    def offsets(self, offsets: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """ Convert the Encoding's offsets to the current string.

        `Encoding` provides a list of offsets that are actually offsets to the Normalized
        version of text. Calling this method with the offsets provided by `Encoding` will make
        sure that said offsets can be used to index the `str` directly.
        """
        pass

class Encoding:
    """ An Encoding as returned by the Tokenizer """

    @property
    def normalized_str(self) -> IndexableString:
        """ The normalized string """
        pass

    @property
    def original_str(self) -> IndexableString:
        """ The original string """
        pass

    @property
    def ids(self) -> List[int]:
        """ The tokenized ids """
        pass

    @property
    def tokens(self) -> List[str]:
        """ The tokenized strings """
        pass

    @property
    def type_ids(self) -> List[int]:
        """ The type ids """
        pass

    @property
    def offsets(self) -> List[Offsets]:
        """ The offsets.
        These offsets can be used to index any `IndexableString` directly. If you want to
        index the original `str`, make sure to retrieve the converted offsets using the `.offsets`
        method on the `original_str`.
        """
        pass

    @property
    def special_tokens_mask(self) -> List[int]:
        """ The special tokens mask """
        pass

    @property
    def attention_mask(self) -> List[int]:
        """ The attention mask """
        pass

    @property
    def overflowing(self) -> Optional[Encoding]:
        """ The overflowing encoding, after truncation """
        pass

    def pad(self,
            length: int,
            pad_id: Optional[int] = 0,
            pad_type_id: Optional[int] = 0,
            pad_token: Optional[str] = "[PAD]",
            direction: Optional[str] = "right"):
        """ Pad the current Encoding at the given length

        Args:
            length: int:
                The length at which to pad

            direction: (`optional`) str:
                Can be one of: `right` or `left`

            pad_id: (`optional`) unsigned int:
                The indice to be used when padding

            pad_type_id: (`optional`) unsigned int:
                The type indice to be used when padding

            pad_token: (`optional`) str:
                The pad token to be used when padding
        """
        pass

    def truncate(self, max_length: int, stride: Optional[int] = 0):
        """ Truncate the current Encoding at the given max_length

        Args:
            max_length: int:
                The maximum length to be kept

            stride: (`optional`) unsigned int:
                The length of the previous first sequence to be included
                in the overflowing sequence
        """
        pass


class Tokenizer:
    """ Tokenizer

    A Tokenizer works as a pipeline, it processes some raw text as input and outputs
    an `Encoding`.

    The various steps of the pipeline are:
        1. The `Normalizer`: in charge of normalizing the text. Common examples of
           normalization are the unicode normalization standards, such as NFD or NFKC.
        2. The `PreTokenizer`: in charge of creating initial words splits in the text.
           The most common way of splitting text is simply on whitespace.
        3. The `Model`: in charge of doing the actual tokenization. An example of a
           `Model` would be `BPE` or `WordPiece`.
        4. The `PostProcessor`: in charge of post-processing the `Encoding` to add anything
           relevant that, for example, a language model would need, such as special tokens.
    """

    def __new__(cls, model: models.Model) -> Tokenizer:
        """ Instantiate a new Tokenizer using the given Model

        Args:
            model: models.Model:
                The model to be used with this Tokenizer

        Returns:
            Tokenizer
        """
        pass


    @property
    def model(self) -> Model:
        """ Get the model in use with this Tokenizer """
        pass

    @model.setter
    def model(self, model: models.Model):
        """ Change the model to use with this Tokenizer """
        pass

    @property
    def pre_tokenizer(self) -> Optional[PreTokenizer]:
        """ Get the pre-tokenizer in use with this model """
        pass

    @pre_tokenizer.setter
    def pre_tokenizer(self, pre_tokenizer: pre_tokenizers.PreTokenizer):
        """ Change the pre tokenizer to use with this Tokenizer """
        pass

    @property
    def decoder(self) -> Optional[Decoder]:
        """ Get the decoder in use with this model """
        pass

    @decoder.setter
    def decoder(self, decoder: decoders.Decoder):
        """ Change the decoder to use with this Tokenizer """
        pass

    @property
    def post_processor(self) -> Optional[PostProcessor]:
        """ Get the post-processor in use with this Tokenizer """
        pass

    @post_processor.setter
    def post_processor(self, processor: processors.PostProcessor):
        """ Change the post processor to use with this Tokenizer """

    @property
    def normalizer(self) -> Optional[Normalizer]:
        """ Get the normalizer in use with this Tokenizer """
        pass

    @normalizer.setter
    def normalizer(self, normalizer: normalizers.Normalizer):
        """ Change the normalizer to use with this Tokenizer """


    def get_vocab_size(self, with_added_tokens: Optional[bool]) -> int:
        """ Returns the size of the vocabulary

        Args:
            with_added_tokens: (`optional`) boolean:
                Whether to include the added tokens in the vocabulary's size
        """
        pass

    def enable_truncation(self,
                          max_length: int,
                          stride: Optional[int],
                          strategy: Optional[str]):
        """ Enable the truncation

        Args:
            max_length: unsigned int:
                The maximum length at which to truncate

            stride: (`optional`) unsigned int:
                The length of the previous first sequence to be included
                in the overflowing sequence

            strategy: (`optional) str:
                Can be one of `longest_first`, `only_first` or `only_second`
        """
        pass

    def no_truncation(self):
        """ Disable truncation """
        pass

    def enable_padding(self,
                       direction: Optional[str] = "right",
                       pad_id: Optional[int] = 0,
                       pad_type_id: Optional[int] = 0,
                       pad_token: Optional[str] = "[PAD]",
                       max_length: Optional[int] = None):
        """ Enable the padding

        Args:
            direction: (`optional`) str:
                Can be one of: `right` or `left`

            pad_id: (`optional`) unsigned int:
                The indice to be used when padding

            pad_type_id: (`optional`) unsigned int:
                The type indice to be used when padding

            pad_token: (`optional`) str:
                The pad token to be used when padding

            max_length: (`optional`) unsigned int:
                If specified, the length at which to pad. If not specified
                we pad using the size of the longest sequence in a batch
        """
        pass

    def no_padding(self):
        """ Disable padding """
        pass

    def encode(self, sequence: str, pair: Optional[str] = None) -> Encoding:
        """ Encode the given sequence

        Args:
            sequence: str:
                The sequence to encode

            pair: (`optional`) Optional[str]:
                The optional pair sequence

        Returns:
            An Encoding
        """
        pass

    def encode_batch(self, sequences: List[Union[str, Tuple[str, str]]]) -> List[Encoding]:
        """ Encode the given sequences or pair of sequences

        Args:
            sequences: List[Union[str, Tuple[str, str]]]:
                A list of sequences or pair of sequences. The list can contain both
                at the same time.

        Returns:
            A list of Encoding
        """
        pass

    def decode(self, ids: List[int], skip_special_tokens: Optional[bool] = True) -> str:
        """ Decode the given list of ids to a string sequence

        Args:
            ids: List[unsigned int]:
                A list of ids to be decoded

            skip_special_tokens: (`optional`) boolean:
                Whether to remove all the special tokens from the output string

        Returns:
            The decoded string
        """
        pass

    def decode_batch(self,
                     sequences: List[List[int]],
                     skip_special_tokens: Optional[bool] = True) -> str:
        """ Decode the list of sequences to a list of string sequences

        Args:
            sequences: List[List[unsigned int]]:
                A list of sequence of ids to be decoded

            skip_special_tokens: (`optional`) boolean:
                Whether to remove all the special tokens from the output strings

        Returns:
            A list of decoded strings
        """
        pass

    def token_to_id(self, token: str) -> Optional[int]:
        """ Convert the given token to its corresponding id

        Args:
            token: str:
                The token to convert

        Returns:
            The corresponding id if it exists, None otherwise
        """
        pass

    def id_to_token(self, id: int) -> Optional[str]:
        """ Convert the given token id to its corresponding string

        Args:
            token: id:
                The token id to convert

        Returns:
            The corresponding string if it exists, None otherwise
        """
        pass

    def add_tokens(self, tokens: List[Union[str, Tuple[str, bool]]]) -> int:
        """ Add the given tokens to the vocabulary

        Args:
            tokens: List[Union[str, Tuple[str, bool]]]:
                A list of tokens to add to the vocabulary. Each token can either be
                a string, or a tuple with a string representing the token, and a boolean
                option representing whether to match on single words only.
                If the boolean is not included, it defaults to False

        Returns:
            The number of tokens that were added to the vocabulary
        """
        pass

    def add_special_tokens(self, tokens: List[str]) -> int:
        """ Add the given special tokens to the vocabulary, and treat them as special tokens.

        The special tokens will never be processed by the model, and will be
        removed while decoding.

        Args:
            tokens: List[str]:
                The list of special tokens to add

        Returns:
            The number of tokens that were added to the vocabulary
        """
        pass
