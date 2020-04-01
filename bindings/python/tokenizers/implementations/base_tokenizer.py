from tokenizers import Tokenizer, Encoding, AddedToken
from tokenizers.models import TokenizedSequence, TokenizedSequenceWithOffsets

from typing import List, Union, Tuple, Optional, Dict

Offsets = Tuple[int, int]


class BaseTokenizer:
    def __init__(self, tokenizer: Tokenizer, parameters=None):
        self._tokenizer = tokenizer
        self._parameters = parameters if parameters is not None else {}

    def __repr__(self):
        return "Tokenizer(vocabulary_size={}, {})".format(
            self._tokenizer.get_vocab_size(),
            ", ".join(k + "=" + str(v) for k, v in self._parameters.items()),
        )

    def num_special_tokens_to_add(self, is_pair: bool) -> int:
        """
        Return the number of special tokens that would be added for single/pair sentences.
        :param is_pair: Boolean indicating if the input would be a single sentence or a pair
        :return:
        """
        return self._tokenizer.num_special_tokens_to_add(is_pair)

    def get_vocab(self, with_added_tokens: bool = True) -> Dict[str, int]:
        """ Returns the vocabulary

        Args:
            with_added_tokens: boolean:
                Whether to include the added tokens in the vocabulary

        Returns:
            The vocabulary
        """
        return self._tokenizer.get_vocab(with_added_tokens=with_added_tokens)

    def get_vocab_size(self, with_added_tokens: bool = True) -> int:
        """ Return the size of vocabulary, with or without added tokens.

        Args:
            with_added_tokens: (`optional`) bool:
                Whether to count in added special tokens or not

        Returns:
            Size of vocabulary
        """
        return self._tokenizer.get_vocab_size(with_added_tokens=with_added_tokens)

    def enable_padding(
        self,
        direction: Optional[str] = "right",
        pad_id: Optional[int] = 0,
        pad_type_id: Optional[int] = 0,
        pad_token: Optional[str] = "[PAD]",
        max_length: Optional[int] = None,
    ):
        """ Change the padding strategy

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
        return self._tokenizer.enable_padding(
            direction=direction,
            pad_id=pad_id,
            pad_type_id=pad_type_id,
            pad_token=pad_token,
            max_length=max_length,
        )

    def no_padding(self):
        """ Disable padding """
        return self._tokenizer.no_padding()

    def enable_truncation(
        self, max_length: int, stride: Optional[int] = 0, strategy: Optional[str] = "longest_first"
    ):
        """ Change the truncation options

        Args:
            max_length: unsigned int:
                The maximum length at which to truncate

            stride: (`optional`) unsigned int:
                The length of the previous first sequence to be included
                in the overflowing sequence

            strategy: (`optional) str:
                Can be one of `longest_first`, `only_first` or `only_second`
        """
        return self._tokenizer.enable_truncation(max_length, stride=stride, strategy=strategy)

    def no_truncation(self):
        """ Disable truncation """
        return self._tokenizer.no_truncation()

    def add_tokens(self, tokens: List[Union[str, AddedToken]]) -> int:
        """ Add the given tokens to the vocabulary

        Args:
            tokens: List[Union[str, AddedToken]]:
                A list of tokens to add to the vocabulary. Each token can either be
                a string, or an instance of AddedToken

        Returns:
            The number of tokens that were added to the vocabulary
        """
        return self._tokenizer.add_tokens(tokens)

    def add_special_tokens(self, special_tokens: List[Union[str, AddedToken]]) -> int:
        """ Add the given special tokens to the vocabulary, and treat them as special tokens.

        The special tokens will never be processed by the model, and will be
        removed while decoding.

        Args:
            tokens: List[Union[str, AddedToken]]:
                A list of special tokens to add to the vocabulary. Each token can either be
                a string, or an instance of AddedToken

        Returns:
            The number of tokens that were added to the vocabulary
        """
        return self._tokenizer.add_special_tokens(special_tokens)

    def normalize(self, sequence: str) -> str:
        """ Normalize the given sequence

        Args:
            sequence: str:
                The sequence to normalize

        Returns:
            The normalized string
        """
        return self._tokenizer.normalize(sequence)

    def encode_tokenized(
        self, sequence: Union[TokenizedSequence, TokenizedSequenceWithOffsets], type_id: int = 0
    ) -> Encoding:
        """ Encode the given sequence. Let us skip the Normalizer and PreTokenizer by providing
        already tokenized substrings.

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
        return self._tokenizer.model.encode(sequence, type_id)

    def encode_tokenized_batch(
        self,
        sequences: Union[List[TokenizedSequence], List[TokenizedSequenceWithOffsets]],
        type_id: int = 0,
    ) -> List[Encoding]:
        """ Encode the given batch of sequence. Let us skip the Normalizer and PreTokenizer by
        providing already tokenized substrings.

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
        return self._tokenizer.model.encode_batch(sequences, type_id)

    def encode(
        self, sequence: str, pair: Optional[str] = None, add_special_tokens: bool = True
    ) -> Encoding:
        """ Encode the given sequence

        Args:
            sequence: str:
                The sequence to encode

            pair: (`optional`) Optional[str]:
                The optional pair sequence

            add_special_tokens: bool:
                Whether to add the special tokens while encoding.

        Returns:
            An Encoding
        """
        if sequence is None:
            raise ValueError("None input is not valid. Should be a string.")

        return self._tokenizer.encode(sequence, pair, add_special_tokens)

    def encode_batch(
        self, sequences: List[Union[str, Tuple[str, str]]], add_special_tokens: bool = True
    ) -> List[Encoding]:
        """ Encode the given sequences or pair of sequences

        Args:
            sequences: List[Union[str, Tuple[str, str]]]:
                A list of sequences or pair of sequences. The list can contain both
                at the same time.

            add_special_tokens: bool:
                Whether to add the special tokens while encoding.

        Returns:
            A list of Encoding
        """

        if sequences is None:
            raise ValueError(
                "None input is not valid. Should be a list of strings or a list of tuple of strings."
            )

        return self._tokenizer.encode_batch(sequences, add_special_tokens)

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
        if ids is None:
            raise ValueError("None input is not valid. Should be a list of integers.")

        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(
        self, sequences: List[List[int]], skip_special_tokens: Optional[bool] = True
    ) -> str:
        """ Decode the list of sequences to a list of string sequences

        Args:
            sequences: List[List[unsigned int]]:
                A list of sequence of ids to be decoded

            skip_special_tokens: (`optional`) boolean:
                Whether to remove all the special tokens from the output strings

        Returns:
            A list of decoded strings
        """
        if sequences is None:
            raise ValueError("None input is not valid. Should be list of list of integers.")

        return self._tokenizer.decode_batch(sequences, skip_special_tokens=skip_special_tokens)

    def token_to_id(self, token: str) -> Optional[int]:
        """ Convert the given token to its corresponding id

        Args:
            token: str:
                The token to convert

        Returns:
            The corresponding id if it exists, None otherwise
        """
        return self._tokenizer.token_to_id(token)

    def id_to_token(self, id: int) -> Optional[str]:
        """ Convert the given token id to its corresponding string

        Args:
            token: id:
                The token id to convert

        Returns:
            The corresponding string if it exists, None otherwise
        """
        return self._tokenizer.id_to_token(id)

    def save(self, directory: str, name: Optional[str] = None):
        """ Save the current model to the given directory

        Args:
            directory: str:
                A path to the destination directory

            name: (Optional) str:
                The name of the tokenizer, to be used in the saved files
        """
        return self._tokenizer.model.save(directory, name=name)

    def post_process(
        self, encoding: Encoding, pair: Optional[Encoding] = None, add_special_tokens: bool = True
    ) -> Encoding:
        """ Apply all the post-processing steps to the given encodings.

        The various steps are:
            1. Truncate according to global params (provided to `enable_truncation`)
            2. Apply the PostProcessor
            3. Pad according to global params. (provided to `enable_padding`)

        Args:
            encoding: Encoding:
                The main Encoding to post process

            pair: Optional[Encoding]:
                An optional pair Encoding

            add_special_tokens: bool:
                Whether to add special tokens

        Returns:
            The resulting Encoding
        """
        return self._tokenizer.post_process(encoding, pair, add_special_tokens)
