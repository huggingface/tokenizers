import _typeshed
import tokenizers
import tokenizers.decoders
import tokenizers.models
import tokenizers.normalizers
import tokenizers.pre_tokenizers
import tokenizers.processors
import tokenizers.trainers
import typing

__version__: typing.Final[str]

class AddedToken:
    def __eq__(self, /, other: tokenizers.AddedToken) -> bool:
        """Return self==value."""
        ...
    def __ge__(self, /, other: tokenizers.AddedToken) -> bool:
        """Return self>=value."""
        ...
    def __getstate__(self, /) -> typing.Any: ...
    def __gt__(self, /, other: tokenizers.AddedToken) -> bool:
        """Return self>value."""
        ...
    def __hash__(self, /) -> int:
        """Return hash(self)."""
        ...
    def __le__(self, /, other: tokenizers.AddedToken) -> bool:
        """Return self<=value."""
        ...
    def __lt__(self, /, other: tokenizers.AddedToken) -> bool:
        """Return self<value."""
        ...
    def __ne__(self, /, other: tokenizers.AddedToken) -> bool:
        """Return self!=value."""
        ...
    def __new__(cls, /, content: str | None = None, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    def __repr__(self, /) -> str:
        """Return repr(self)."""
        ...
    def __setstate__(self, /, state: typing.Any) -> typing.Any: ...
    def __str__(self, /) -> str:
        """Return str(self)."""
        ...
    @property
    def content(self, /) -> str:
        """Get the content of this :obj:`AddedToken`"""
        ...
    @content.setter
    def content(self, /, content: str) -> None:
        """Get the content of this :obj:`AddedToken`"""
        ...
    @property
    def lstrip(self, /) -> bool:
        """Get the value of the :obj:`lstrip` option"""
        ...
    @property
    def normalized(self, /) -> bool:
        """Get the value of the :obj:`normalized` option"""
        ...
    @property
    def rstrip(self, /) -> bool:
        """Get the value of the :obj:`rstrip` option"""
        ...
    @property
    def single_word(self, /) -> bool:
        """Get the value of the :obj:`single_word` option"""
        ...
    @property
    def special(self, /) -> bool:
        """Get the value of the :obj:`special` option"""
        ...
    @special.setter
    def special(self, /, special: bool) -> None:
        """Get the value of the :obj:`special` option"""
        ...

class Encoding:
    def __getstate__(self, /) -> typing.Any: ...
    def __len__(self, /) -> int:
        """Return len(self)."""
        ...
    def __new__(cls, /) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    def __repr__(self, /) -> str:
        """Return repr(self)."""
        ...
    def __setstate__(self, /, state: typing.Any) -> typing.Any: ...
    @property
    def attention_mask(self, /) -> typing.Any:
        """
        The attention mask

        This indicates to the LM which tokens should be attended to, and which should not.
        This is especially important when batching sequences, where we need to applying
        padding.

        Returns:
           :obj:`List[int]`: The attention mask
        """
        ...
    def char_to_token(self, /, char_pos: int, sequence_index: int = 0) -> typing.Any:
        """
        Get the token that contains the char at the given position in the input sequence.

        Args:
            char_pos (:obj:`int`):
                The position of a char in the input string
            sequence_index (:obj:`int`, defaults to :obj:`0`):
                The index of the sequence that contains the target char

        Returns:
            :obj:`int`: The index of the token that contains this char in the encoded sequence
        """
        ...
    def char_to_word(self, /, char_pos: int, sequence_index: int = 0) -> typing.Any:
        """
        Get the word that contains the char at the given position in the input sequence.

        Args:
            char_pos (:obj:`int`):
                The position of a char in the input string
            sequence_index (:obj:`int`, defaults to :obj:`0`):
                The index of the sequence that contains the target char

        Returns:
            :obj:`int`: The index of the word that contains this char in the input sequence
        """
        ...
    @property
    def ids(self, /) -> typing.Any:
        """
        The generated IDs

        The IDs are the main input to a Language Model. They are the token indices,
        the numerical representations that a LM understands.

        Returns:
            :obj:`List[int]`: The list of IDs
        """
        ...
    @staticmethod
    def merge(encodings: typing.Any, growing_offsets: bool = True) -> Encoding:
        """
        Merge the list of encodings into one final :class:`~tokenizers.Encoding`

        Args:
            encodings (A :obj:`List` of :class:`~tokenizers.Encoding`):
                The list of encodings that should be merged in one

            growing_offsets (:obj:`bool`, defaults to :obj:`True`):
                Whether the offsets should accumulate while merging

        Returns:
            :class:`~tokenizers.Encoding`: The resulting Encoding
        """
        ...
    @property
    def n_sequences(self, /) -> int:
        """
        The number of sequences represented

        Returns:
            :obj:`int`: The number of sequences in this :class:`~tokenizers.Encoding`
        """
        ...
    @property
    def offsets(self, /) -> typing.Any:
        """
        The offsets associated to each token

        These offsets let's you slice the input string, and thus retrieve the original
        part that led to producing the corresponding token.

        Returns:
            A :obj:`List` of :obj:`Tuple[int, int]`: The list of offsets
        """
        ...
    @property
    def overflowing(self, /) -> typing.Any:
        """
        A :obj:`List` of overflowing :class:`~tokenizers.Encoding`

        When using truncation, the :class:`~tokenizers.Tokenizer` takes care of splitting
        the output into as many pieces as required to match the specified maximum length.
        This field lets you retrieve all the subsequent pieces.

        When you use pairs of sequences, the overflowing pieces will contain enough
        variations to cover all the possible combinations, while respecting the provided
        maximum length.
        """
        ...
    def pad(self, /, length: int, **kwargs) -> None:
        """
        Pad the :class:`~tokenizers.Encoding` at the given length

        Args:
            length (:obj:`int`):
                The desired length

            direction: (:obj:`str`, defaults to :obj:`right`):
                The expected padding direction. Can be either :obj:`right` or :obj:`left`

            pad_id (:obj:`int`, defaults to :obj:`0`):
                The ID corresponding to the padding token

            pad_type_id (:obj:`int`, defaults to :obj:`0`):
                The type ID corresponding to the padding token

            pad_token (:obj:`str`, defaults to `[PAD]`):
                The pad token to use
        """
        ...
    @property
    def sequence_ids(self, /) -> typing.Any:
        """
        The generated sequence indices.

        They represent the index of the input sequence associated to each token.
        The sequence id can be None if the token is not related to any input sequence,
        like for example with special tokens.

        Returns:
            A :obj:`List` of :obj:`Optional[int]`: A list of optional sequence index.
        """
        ...
    def set_sequence_id(self, /, sequence_id: int) -> None:
        """
        Set the given sequence index

        Set the given sequence index for the whole range of tokens contained in this
        :class:`~tokenizers.Encoding`.
        """
        ...
    @property
    def special_tokens_mask(self, /) -> typing.Any:
        """
        The special token mask

        This indicates which tokens are special tokens, and which are not.

        Returns:
            :obj:`List[int]`: The special tokens mask
        """
        ...
    def token_to_chars(self, /, token_index: int) -> typing.Any:
        """
        Get the offsets of the token at the given index.

        The returned offsets are related to the input sequence that contains the
        token.  In order to determine in which input sequence it belongs, you
        must call :meth:`~tokenizers.Encoding.token_to_sequence()`.

        Args:
            token_index (:obj:`int`):
                The index of a token in the encoded sequence.

        Returns:
            :obj:`Tuple[int, int]`: The token offsets :obj:`(first, last + 1)`
        """
        ...
    def token_to_sequence(self, /, token_index: int) -> typing.Any:
        """
        Get the index of the sequence represented by the given token.

        In the general use case, this method returns :obj:`0` for a single sequence or
        the first sequence of a pair, and :obj:`1` for the second sequence of a pair

        Args:
            token_index (:obj:`int`):
                The index of a token in the encoded sequence.

        Returns:
            :obj:`int`: The sequence id of the given token
        """
        ...
    def token_to_word(self, /, token_index: int) -> typing.Any:
        """
        Get the index of the word that contains the token in one of the input sequences.

        The returned word index is related to the input sequence that contains
        the token.  In order to determine in which input sequence it belongs, you
        must call :meth:`~tokenizers.Encoding.token_to_sequence()`.

        Args:
            token_index (:obj:`int`):
                The index of a token in the encoded sequence.

        Returns:
            :obj:`int`: The index of the word in the relevant input sequence.
        """
        ...
    @property
    def tokens(self, /) -> typing.Any:
        """
        The generated tokens

        They are the string representation of the IDs.

        Returns:
            :obj:`List[str]`: The list of tokens
        """
        ...
    def truncate(self, /, max_length: int, stride: int = 0, direction: str = "right") -> None:
        """
        Truncate the :class:`~tokenizers.Encoding` at the given length

        If this :class:`~tokenizers.Encoding` represents multiple sequences, when truncating
        this information is lost. It will be considered as representing a single sequence.

        Args:
            max_length (:obj:`int`):
                The desired length

            stride (:obj:`int`, defaults to :obj:`0`):
                The length of previous content to be included in each overflowing piece

            direction (:obj:`str`, defaults to :obj:`right`):
                Truncate direction
        """
        ...
    @property
    def type_ids(self, /) -> typing.Any:
        """
        The generated type IDs

        Generally used for tasks like sequence classification or question answering,
        these tokens let the LM know which input sequence corresponds to each tokens.

        Returns:
            :obj:`List[int]`: The list of type ids
        """
        ...
    @property
    def word_ids(self, /) -> typing.Any:
        """
        The generated word indices.

        They represent the index of the word associated to each token.
        When the input is pre-tokenized, they correspond to the ID of the given input label,
        otherwise they correspond to the words indices as defined by the
        :class:`~tokenizers.pre_tokenizers.PreTokenizer` that was used.

        For special tokens and such (any token that was generated from something that was
        not part of the input), the output is :obj:`None`

        Returns:
            A :obj:`List` of :obj:`Optional[int]`: A list of optional word index.
        """
        ...
    def word_to_chars(self, /, word_index: int, sequence_index: int = 0) -> typing.Any:
        """
        Get the offsets of the word at the given index in one of the input sequences.

        Args:
            word_index (:obj:`int`):
                The index of a word in one of the input sequences.
            sequence_index (:obj:`int`, defaults to :obj:`0`):
                The index of the sequence that contains the target word

        Returns:
            :obj:`Tuple[int, int]`: The range of characters (span) :obj:`(first, last + 1)`
        """
        ...
    def word_to_tokens(self, /, word_index: int, sequence_index: int = 0) -> typing.Any:
        """
        Get the encoded tokens corresponding to the word at the given index
        in one of the input sequences.

        Args:
            word_index (:obj:`int`):
                The index of a word in one of the input sequences.
            sequence_index (:obj:`int`, defaults to :obj:`0`):
                The index of the sequence that contains the target word

        Returns:
            :obj:`Tuple[int, int]`: The range of tokens: :obj:`(first, last + 1)`
        """
        ...
    @property
    def words(self, /) -> typing.Any:
        """
        The generated word indices.

        .. warning::
            This is deprecated and will be removed in a future version.
            Please use :obj:`~tokenizers.Encoding.word_ids` instead.

        They represent the index of the word associated to each token.
        When the input is pre-tokenized, they correspond to the ID of the given input label,
        otherwise they correspond to the words indices as defined by the
        :class:`~tokenizers.pre_tokenizers.PreTokenizer` that was used.

        For special tokens and such (any token that was generated from something that was
        not part of the input), the output is :obj:`None`

        Returns:
            A :obj:`List` of :obj:`Optional[int]`: A list of optional word index.
        """
        ...

class NormalizedString:
    def __getitem__(self, /, range: int | tuple[int, int] | typing.Any) -> typing.Any:
        """Return self[key]."""
        ...
    def __new__(cls, /, sequence: str) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    def __repr__(self, /) -> str:
        """Return repr(self)."""
        ...
    def __str__(self, /) -> str:
        """Return str(self)."""
        ...
    def append(self, /, s: str) -> None:
        """Append the given sequence to the string"""
        ...
    def clear(self, /) -> None:
        """Clears the string"""
        ...
    def filter(self, /, func: typing.Any) -> typing.Any:
        """Filter each character of the string using the given func"""
        ...
    def for_each(self, /, func: typing.Any) -> typing.Any:
        """Calls the given function for each character of the string"""
        ...
    def lowercase(self, /) -> None:
        """Lowercase the string"""
        ...
    def lstrip(self, /) -> None:
        """Strip the left of the string"""
        ...
    def map(self, /, func: typing.Any) -> typing.Any:
        """
        Calls the given function for each character of the string

        Replaces each character of the string using the returned value. Each
        returned value **must** be a str of length 1 (ie a character).
        """
        ...
    def nfc(self, /) -> None:
        """Runs the NFC normalization"""
        ...
    def nfd(self, /) -> None:
        """Runs the NFD normalization"""
        ...
    def nfkc(self, /) -> None:
        """Runs the NFKC normalization"""
        ...
    def nfkd(self, /) -> None:
        """Runs the NFKD normalization"""
        ...
    @property
    def normalized(self, /) -> str:
        """The normalized part of the string"""
        ...
    @property
    def original(self, /) -> str: ...
    def prepend(self, /, s: str) -> None:
        """Prepend the given sequence to the string"""
        ...
    def replace(self, /, pattern: str | tokenizers.Regex, content: str) -> typing.Any:
        """
        Replace the content of the given pattern with the provided content

        Args:
            pattern: Pattern:
                A pattern used to match the string. Usually a string or a Regex

            content: str:
                The content to be used as replacement
        """
        ...
    def rstrip(self, /) -> None:
        """Strip the right of the string"""
        ...
    def slice(self, /, range: int | tuple[int, int] | typing.Any) -> typing.Any:
        """Slice the string using the given range"""
        ...
    def split(self, /, pattern: str | tokenizers.Regex, behavior: typing.Any) -> typing.Any:
        """
        Split the NormalizedString using the given pattern and the specified behavior

        Args:
            pattern: Pattern:
                A pattern used to split the string. Usually a string or a regex built with `tokenizers.Regex`

            behavior: SplitDelimiterBehavior:
                The behavior to use when splitting.
                Choices: "removed", "isolated", "merged_with_previous", "merged_with_next",
                "contiguous"

        Returns:
            A list of NormalizedString, representing each split
        """
        ...
    def strip(self, /) -> None:
        """Strip both ends of the string"""
        ...
    def uppercase(self, /) -> None:
        """Uppercase the string"""
        ...

class PreTokenizedString:
    def __new__(cls, /, s: str) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    def get_splits(self, /, offset_referential: typing.Any = ..., offset_type: typing.Any = ...) -> typing.Any:
        """
        Get the splits currently managed by the PreTokenizedString

        Args:
            offset_referential: :obj:`str`
                Whether the returned splits should have offsets expressed relative
                to the original string, or the normalized one. choices: "original", "normalized".

            offset_type: :obj:`str`
                Whether the returned splits should have offsets expressed in bytes or chars.
                When slicing an str, we usually want to use chars, which is the default value.
                Now in some cases it might be interesting to get these offsets expressed in bytes,
                so it is possible to change this here.
                choices: "char", "bytes"

        Returns
            A list of splits
        """
        ...
    def normalize(self, /, func: typing.Any) -> typing.Any:
        """
        Normalize each split of the `PreTokenizedString` using the given `func`

        Args:
            func: Callable[[NormalizedString], None]:
                The function used to normalize each underlying split. This function
                does not need to return anything, just calling the methods on the provided
                NormalizedString allow its modification.
        """
        ...
    def split(self, /, func: typing.Any) -> typing.Any:
        """
        Split the PreTokenizedString using the given `func`

        Args:
            func: Callable[[index, NormalizedString], List[NormalizedString]]:
                The function used to split each underlying split.
                It is expected to return a list of `NormalizedString`, that represent the new
                splits. If the given `NormalizedString` does not need any splitting, we can
                just return it directly.
                In order for the offsets to be tracked accurately, any returned `NormalizedString`
                should come from calling either `.split` or `.slice` on the received one.
        """
        ...
    def to_encoding(self, /, type_id: int = 0, word_idx: int | None = None) -> Encoding:
        """
        Return an Encoding generated from this PreTokenizedString

        Args:
            type_id: int = 0:
                The type_id to be used on the generated Encoding.

            word_idx: Optional[int] = None:
                An optional word index to be used for each token of this Encoding. If provided,
                all the word indices in the generated Encoding will use this value, instead
                of the one automatically tracked during pre-tokenization.

        Returns:
            An Encoding
        """
        ...
    def tokenize(self, /, func: typing.Any) -> typing.Any:
        """
        Tokenize each split of the `PreTokenizedString` using the given `func`

        Args:
            func: Callable[[str], List[Token]]:
                The function used to tokenize each underlying split. This function must return
                a list of Token generated from the input str.
        """
        ...

class Regex:
    def __new__(cls, /, s: str) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...

class Token:
    def __new__(cls, /, id: int, value: str, offsets: typing.Any) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    def as_tuple(self, /) -> typing.Any: ...
    @property
    def id(self, /) -> int: ...
    @property
    def offsets(self, /) -> typing.Any: ...
    @property
    def value(self, /) -> str: ...

class Tokenizer:
    def __getnewargs__(self, /) -> typing.Any: ...
    def __getstate__(self, /) -> typing.Any: ...
    def __new__(cls, /, model: tokenizers.models.Model) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    def __repr__(self, /) -> str:
        """Return repr(self)."""
        ...
    def __setstate__(self, /, state: typing.Any) -> typing.Any: ...
    def __str__(self, /) -> str:
        """Return str(self)."""
        ...
    def add_special_tokens(self, /, tokens: typing.Any) -> int:
        """
        Add the given special tokens to the Tokenizer.

        If these tokens are already part of the vocabulary, it just let the Tokenizer know about
        them. If they don't exist, the Tokenizer creates them, giving them a new id.

        These special tokens will never be processed by the model (ie won't be split into
        multiple tokens), and they can be removed from the output when decoding.

        Args:
            tokens (A :obj:`List` of :class:`~tokenizers.AddedToken` or :obj:`str`):
                The list of special tokens we want to add to the vocabulary. Each token can either
                be a string or an instance of :class:`~tokenizers.AddedToken` for more
                customization.

        Returns:
            :obj:`int`: The number of tokens that were created in the vocabulary
        """
        ...
    def add_tokens(self, /, tokens: typing.Any) -> int:
        """
        Add the given tokens to the vocabulary

        The given tokens are added only if they don't already exist in the vocabulary.
        Each token then gets a new attributed id.

        Args:
            tokens (A :obj:`List` of :class:`~tokenizers.AddedToken` or :obj:`str`):
                The list of tokens we want to add to the vocabulary. Each token can be either a
                string or an instance of :class:`~tokenizers.AddedToken` for more customization.

        Returns:
            :obj:`int`: The number of tokens that were created in the vocabulary
        """
        ...
    def async_decode_batch(self, /, sequences: typing.Any, skip_special_tokens: bool = True) -> typing.Any:
        """
        Decode a batch of ids back to their corresponding string

        Args:
            sequences (:obj:`List` of :obj:`List[int]`):
                The batch of sequences we want to decode

            skip_special_tokens (:obj:`bool`, defaults to :obj:`True`):
                Whether the special tokens should be removed from the decoded strings

        Returns:
            :obj:`List[str]`: A list of decoded strings
        """
        ...
    def async_encode(
        self,
        /,
        sequence: typing.Any,
        pair: typing.Any | None = None,
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
    ) -> typing.Any:
        """
        Asynchronously encode the given input with character offsets.

        This is an async version of encode that can be awaited in async Python code.

        Example:
            Here are some examples of the inputs that are accepted::

                await async_encode("A single sequence")

        Args:
            sequence (:obj:`~tokenizers.InputSequence`):
                The main input sequence we want to encode. This sequence can be either raw
                text or pre-tokenized, according to the ``is_pretokenized`` argument:

                - If ``is_pretokenized=False``: :class:`~tokenizers.TextInputSequence`
                - If ``is_pretokenized=True``: :class:`~tokenizers.PreTokenizedInputSequence`

            pair (:obj:`~tokenizers.InputSequence`, `optional`):
                An optional input sequence. The expected format is the same that for ``sequence``.

            is_pretokenized (:obj:`bool`, defaults to :obj:`False`):
                Whether the input is already pre-tokenized

            add_special_tokens (:obj:`bool`, defaults to :obj:`True`):
                Whether to add the special tokens

        Returns:
            :class:`~tokenizers.Encoding`: The encoded result
        """
        ...
    def async_encode_batch(
        self, /, input: typing.Any, is_pretokenized: bool = False, add_special_tokens: bool = True
    ) -> typing.Any:
        """
        Asynchronously encode the given batch of inputs with character offsets.

        This is an async version of encode_batch that can be awaited in async Python code.

        Example:
            Here are some examples of the inputs that are accepted::

                await async_encode_batch([
                    "A single sequence",
                    ("A tuple with a sequence", "And its pair"),
                    [ "A", "pre", "tokenized", "sequence" ],
                    ([ "A", "pre", "tokenized", "sequence" ], "And its pair")
                ])

        Args:
            input (A :obj:`List`/:obj:`Tuple` of :obj:`~tokenizers.EncodeInput`):
                A list of single sequences or pair sequences to encode. Each sequence
                can be either raw text or pre-tokenized, according to the ``is_pretokenized``
                argument:

                - If ``is_pretokenized=False``: :class:`~tokenizers.TextEncodeInput`
                - If ``is_pretokenized=True``: :class:`~tokenizers.PreTokenizedEncodeInput`

            is_pretokenized (:obj:`bool`, defaults to :obj:`False`):
                Whether the input is already pre-tokenized

            add_special_tokens (:obj:`bool`, defaults to :obj:`True`):
                Whether to add the special tokens

        Returns:
            A :obj:`List` of :class:`~tokenizers.Encoding`: The encoded batch
        """
        ...
    def async_encode_batch_fast(
        self, /, input: typing.Any, is_pretokenized: bool = False, add_special_tokens: bool = True
    ) -> typing.Any:
        """
        Asynchronously encode the given batch of inputs without tracking character offsets.

        This is an async version of encode_batch_fast that can be awaited in async Python code.

        Example:
            Here are some examples of the inputs that are accepted::

                await async_encode_batch_fast([
                    "A single sequence",
                    ("A tuple with a sequence", "And its pair"),
                    [ "A", "pre", "tokenized", "sequence" ],
                    ([ "A", "pre", "tokenized", "sequence" ], "And its pair")
                ])

        Args:
            input (A :obj:`List`/:obj:`Tuple` of :obj:`~tokenizers.EncodeInput`):
                A list of single sequences or pair sequences to encode. Each sequence
                can be either raw text or pre-tokenized, according to the ``is_pretokenized``
                argument:

                - If ``is_pretokenized=False``: :class:`~tokenizers.TextEncodeInput`
                - If ``is_pretokenized=True``: :class:`~tokenizers.PreTokenizedEncodeInput`

            is_pretokenized (:obj:`bool`, defaults to :obj:`False`):
                Whether the input is already pre-tokenized

            add_special_tokens (:obj:`bool`, defaults to :obj:`True`):
                Whether to add the special tokens

        Returns:
            A :obj:`List` of :class:`~tokenizers.Encoding`: The encoded batch
        """
        ...
    def decode(self, /, ids: typing.Any, skip_special_tokens: bool = True) -> str:
        """
        Decode the given list of ids back to a string

        This is used to decode anything coming back from a Language Model

        Args:
            ids (A :obj:`List/Tuple` of :obj:`int`):
                The list of ids that we want to decode

            skip_special_tokens (:obj:`bool`, defaults to :obj:`True`):
                Whether the special tokens should be removed from the decoded string

        Returns:
            :obj:`str`: The decoded string
        """
        ...
    def decode_batch(self, /, sequences: typing.Any, skip_special_tokens: bool = True) -> list[str]:
        """
        Decode a batch of ids back to their corresponding string

        Args:
            sequences (:obj:`List` of :obj:`List[int]`):
                The batch of sequences we want to decode

            skip_special_tokens (:obj:`bool`, defaults to :obj:`True`):
                Whether the special tokens should be removed from the decoded strings

        Returns:
            :obj:`List[str]`: A list of decoded strings
        """
        ...
    @property
    def decoder(self, /) -> typing.Any:
        """The `optional` :class:`~tokenizers.decoders.Decoder` in use by the Tokenizer"""
        ...
    @decoder.setter
    def decoder(self, /, decoder: tokenizers.decoders.Decoder | None) -> None:
        """The `optional` :class:`~tokenizers.decoders.Decoder` in use by the Tokenizer"""
        ...
    def enable_padding(self, /, **kwargs) -> None:
        """
        Enable the padding

        Args:
            direction (:obj:`str`, `optional`, defaults to :obj:`right`):
                The direction in which to pad. Can be either ``right`` or ``left``

            pad_to_multiple_of (:obj:`int`, `optional`):
                If specified, the padding length should always snap to the next multiple of the
                given value. For example if we were going to pad witha length of 250 but
                ``pad_to_multiple_of=8`` then we will pad to 256.

            pad_id (:obj:`int`, defaults to 0):
                The id to be used when padding

            pad_type_id (:obj:`int`, defaults to 0):
                The type id to be used when padding

            pad_token (:obj:`str`, defaults to :obj:`[PAD]`):
                The pad token to be used when padding

            length (:obj:`int`, `optional`):
                If specified, the length at which to pad. If not specified we pad using the size of
                the longest sequence in a batch.
        """
        ...
    def enable_truncation(self, /, max_length: int, **kwargs) -> None:
        """
        Enable truncation

        Args:
            max_length (:obj:`int`):
                The max length at which to truncate

            stride (:obj:`int`, `optional`):
                The length of the previous first sequence to be included in the overflowing
                sequence

            strategy (:obj:`str`, `optional`, defaults to :obj:`longest_first`):
                The strategy used to truncation. Can be one of ``longest_first``, ``only_first`` or
                ``only_second``.

            direction (:obj:`str`, defaults to :obj:`right`):
                Truncate direction
        """
        ...
    def encode(
        self,
        /,
        sequence: typing.Any,
        pair: typing.Any | None = None,
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
    ) -> Encoding:
        """
        Encode the given sequence and pair. This method can process raw text sequences
        as well as already pre-tokenized sequences.

        Example:
            Here are some examples of the inputs that are accepted::

                encode("A single sequence")`
                encode("A sequence", "And its pair")`
                encode([ "A", "pre", "tokenized", "sequence" ], is_pretokenized=True)`
                encode(
                    [ "A", "pre", "tokenized", "sequence" ], [ "And", "its", "pair" ],
                    is_pretokenized=True
                )

        Args:
            sequence (:obj:`~tokenizers.InputSequence`):
                The main input sequence we want to encode. This sequence can be either raw
                text or pre-tokenized, according to the ``is_pretokenized`` argument:

                - If ``is_pretokenized=False``: :class:`~tokenizers.TextInputSequence`
                - If ``is_pretokenized=True``: :class:`~tokenizers.PreTokenizedInputSequence`

            pair (:obj:`~tokenizers.InputSequence`, `optional`):
                An optional input sequence. The expected format is the same that for ``sequence``.

            is_pretokenized (:obj:`bool`, defaults to :obj:`False`):
                Whether the input is already pre-tokenized

            add_special_tokens (:obj:`bool`, defaults to :obj:`True`):
                Whether to add the special tokens

        Returns:
            :class:`~tokenizers.Encoding`: The encoded result
        """
        ...
    def encode_batch(
        self, /, input: typing.Any, is_pretokenized: bool = False, add_special_tokens: bool = True
    ) -> list[Encoding]:
        """
        Encode the given batch of inputs. This method accept both raw text sequences
        as well as already pre-tokenized sequences. The reason we use `PySequence` is
        because it allows type checking with zero-cost (according to PyO3) as we don't
        have to convert to check.

        Example:
            Here are some examples of the inputs that are accepted::

                encode_batch([
                    "A single sequence",
                    ("A tuple with a sequence", "And its pair"),
                    [ "A", "pre", "tokenized", "sequence" ],
                    ([ "A", "pre", "tokenized", "sequence" ], "And its pair")
                ])

        Args:
            input (A :obj:`List`/:obj:`Tuple` of :obj:`~tokenizers.EncodeInput`):
                A list of single sequences or pair sequences to encode. Each sequence
                can be either raw text or pre-tokenized, according to the ``is_pretokenized``
                argument:

                - If ``is_pretokenized=False``: :class:`~tokenizers.TextEncodeInput`
                - If ``is_pretokenized=True``: :class:`~tokenizers.PreTokenizedEncodeInput`

            is_pretokenized (:obj:`bool`, defaults to :obj:`False`):
                Whether the input is already pre-tokenized

            add_special_tokens (:obj:`bool`, defaults to :obj:`True`):
                Whether to add the special tokens

        Returns:
            A :obj:`List` of :class:`~tokenizers.Encoding`: The encoded batch
        """
        ...
    def encode_batch_fast(
        self, /, input: typing.Any, is_pretokenized: bool = False, add_special_tokens: bool = True
    ) -> list[Encoding]:
        """
        Encode the given batch of inputs. This method is faster than `encode_batch`
        because it doesn't keep track of offsets, they will be all zeros.

        Example:
            Here are some examples of the inputs that are accepted::

                encode_batch_fast([
                    "A single sequence",
                    ("A tuple with a sequence", "And its pair"),
                    [ "A", "pre", "tokenized", "sequence" ],
                    ([ "A", "pre", "tokenized", "sequence" ], "And its pair")
                ])

        Args:
            input (A :obj:`List`/:obj:`Tuple` of :obj:`~tokenizers.EncodeInput`):
                A list of single sequences or pair sequences to encode. Each sequence
                can be either raw text or pre-tokenized, according to the ``is_pretokenized``
                argument:

                - If ``is_pretokenized=False``: :class:`~tokenizers.TextEncodeInput`
                - If ``is_pretokenized=True``: :class:`~tokenizers.PreTokenizedEncodeInput`

            is_pretokenized (:obj:`bool`, defaults to :obj:`False`):
                Whether the input is already pre-tokenized

            add_special_tokens (:obj:`bool`, defaults to :obj:`True`):
                Whether to add the special tokens

        Returns:
            A :obj:`List` of :class:`~tokenizers.Encoding`: The encoded batch
        """
        ...
    @property
    def encode_special_tokens(self, /) -> bool:
        """
        Modifies the tokenizer in order to use or not the special tokens
        during encoding.

        Args:
            value (:obj:`bool`):
                Whether to use the special tokens or not
        """
        ...
    @encode_special_tokens.setter
    def encode_special_tokens(self, /, value: bool) -> None:
        """
        Modifies the tokenizer in order to use or not the special tokens
        during encoding.

        Args:
            value (:obj:`bool`):
                Whether to use the special tokens or not
        """
        ...
    @staticmethod
    def from_buffer(buffer: typing.Any) -> Tokenizer:
        """
        Instantiate a new :class:`~tokenizers.Tokenizer` from the given buffer.

        Args:
            buffer (:obj:`bytes`):
                A buffer containing a previously serialized :class:`~tokenizers.Tokenizer`

        Returns:
            :class:`~tokenizers.Tokenizer`: The new tokenizer
        """
        ...
    @staticmethod
    def from_file(path: str) -> Tokenizer:
        """
        Instantiate a new :class:`~tokenizers.Tokenizer` from the file at the given path.

        Args:
            path (:obj:`str`):
                A path to a local JSON file representing a previously serialized
                :class:`~tokenizers.Tokenizer`

        Returns:
            :class:`~tokenizers.Tokenizer`: The new tokenizer
        """
        ...
    @staticmethod
    def from_pretrained(identifier: str, revision: str = ..., token: str | None = None) -> Tokenizer:
        """
        Instantiate a new :class:`~tokenizers.Tokenizer` from an existing file on the
        Hugging Face Hub.

        Args:
            identifier (:obj:`str`):
                The identifier of a Model on the Hugging Face Hub, that contains
                a tokenizer.json file
            revision (:obj:`str`, defaults to `main`):
                A branch or commit id
            token (:obj:`str`, `optional`, defaults to `None`):
                An optional auth token used to access private repositories on the
                Hugging Face Hub

        Returns:
            :class:`~tokenizers.Tokenizer`: The new tokenizer
        """
        ...
    @staticmethod
    def from_str(json: str) -> Tokenizer:
        """
        Instantiate a new :class:`~tokenizers.Tokenizer` from the given JSON string.

        Args:
            json (:obj:`str`):
                A valid JSON string representing a previously serialized
                :class:`~tokenizers.Tokenizer`

        Returns:
            :class:`~tokenizers.Tokenizer`: The new tokenizer
        """
        ...
    def get_added_tokens_decoder(self, /) -> dict[int, AddedToken]:
        """
        Get the underlying vocabulary

        Returns:
            :obj:`Dict[int, AddedToken]`: The vocabulary
        """
        ...
    def get_vocab(self, /, with_added_tokens: bool = True) -> dict[str, int]:
        """
        Get the underlying vocabulary

        Args:
            with_added_tokens (:obj:`bool`, defaults to :obj:`True`):
                Whether to include the added tokens

        Returns:
            :obj:`Dict[str, int]`: The vocabulary
        """
        ...
    def get_vocab_size(self, /, with_added_tokens: bool = True) -> int:
        """
        Get the size of the underlying vocabulary

        Args:
            with_added_tokens (:obj:`bool`, defaults to :obj:`True`):
                Whether to include the added tokens

        Returns:
            :obj:`int`: The size of the vocabulary
        """
        ...
    def id_to_token(self, /, id: int) -> str | None:
        """
        Convert the given id to its corresponding token if it exists

        Args:
            id (:obj:`int`):
                The id to convert

        Returns:
            :obj:`Optional[str]`: An optional token, :obj:`None` if out of vocabulary
        """
        ...
    @property
    def model(self, /) -> typing.Any:
        """The :class:`~tokenizers.models.Model` in use by the Tokenizer"""
        ...
    @model.setter
    def model(self, /, model: tokenizers.models.Model) -> None:
        """The :class:`~tokenizers.models.Model` in use by the Tokenizer"""
        ...
    def no_padding(self, /) -> None:
        """Disable padding"""
        ...
    def no_truncation(self, /) -> None:
        """Disable truncation"""
        ...
    @property
    def normalizer(self, /) -> typing.Any:
        """The `optional` :class:`~tokenizers.normalizers.Normalizer` in use by the Tokenizer"""
        ...
    @normalizer.setter
    def normalizer(self, /, normalizer: tokenizers.normalizers.Normalizer | None) -> None:
        """The `optional` :class:`~tokenizers.normalizers.Normalizer` in use by the Tokenizer"""
        ...
    def num_special_tokens_to_add(self, /, is_pair: bool) -> int:
        """
        Return the number of special tokens that would be added for single/pair sentences.
        :param is_pair: Boolean indicating if the input would be a single sentence or a pair
        :return:
        """
        ...
    @property
    def padding(self, /) -> typing.Any:
        """
        Get the current padding parameters

        `Cannot be set, use` :meth:`~tokenizers.Tokenizer.enable_padding` `instead`

        Returns:
            (:obj:`dict`, `optional`):
                A dict with the current padding parameters if padding is enabled
        """
        ...
    def post_process(
        self,
        /,
        encoding: tokenizers.Encoding,
        pair: tokenizers.Encoding | None = None,
        add_special_tokens: bool = True,
    ) -> tokenizers.Encoding:
        """
        Apply all the post-processing steps to the given encodings.

        The various steps are:

            1. Truncate according to the set truncation params (provided with
               :meth:`~tokenizers.Tokenizer.enable_truncation`)
            2. Apply the :class:`~tokenizers.processors.PostProcessor`
            3. Pad according to the set padding params (provided with
               :meth:`~tokenizers.Tokenizer.enable_padding`)

        Args:
            encoding (:class:`~tokenizers.Encoding`):
                The :class:`~tokenizers.Encoding` corresponding to the main sequence.

            pair (:class:`~tokenizers.Encoding`, `optional`):
                An optional :class:`~tokenizers.Encoding` corresponding to the pair sequence.

            add_special_tokens (:obj:`bool`):
                Whether to add the special tokens

        Returns:
            :class:`~tokenizers.Encoding`: The final post-processed encoding
        """
        ...
    @property
    def post_processor(self, /) -> typing.Any:
        """The `optional` :class:`~tokenizers.processors.PostProcessor` in use by the Tokenizer"""
        ...
    @post_processor.setter
    def post_processor(self, /, processor: tokenizers.processors.PostProcessor | None) -> None:
        """The `optional` :class:`~tokenizers.processors.PostProcessor` in use by the Tokenizer"""
        ...
    @property
    def pre_tokenizer(self, /) -> typing.Any:
        """The `optional` :class:`~tokenizers.pre_tokenizers.PreTokenizer` in use by the Tokenizer"""
        ...
    @pre_tokenizer.setter
    def pre_tokenizer(self, /, pretok: tokenizers.pre_tokenizers.PreTokenizer | None) -> None:
        """The `optional` :class:`~tokenizers.pre_tokenizers.PreTokenizer` in use by the Tokenizer"""
        ...
    def save(self, /, path: str, pretty: bool = True) -> None:
        """
        Save the :class:`~tokenizers.Tokenizer` to the file at the given path.

        Args:
            path (:obj:`str`):
                A path to a file in which to save the serialized tokenizer.

            pretty (:obj:`bool`, defaults to :obj:`True`):
                Whether the JSON file should be pretty formatted.
        """
        ...
    def to_str(self, /, pretty: bool = False) -> str:
        """
        Gets a serialized string representing this :class:`~tokenizers.Tokenizer`.

        Args:
            pretty (:obj:`bool`, defaults to :obj:`False`):
                Whether the JSON string should be pretty formatted.

        Returns:
            :obj:`str`: A string representing the serialized Tokenizer
        """
        ...
    def token_to_id(self, /, token: str) -> int | None:
        """
        Convert the given token to its corresponding id if it exists

        Args:
            token (:obj:`str`):
                The token to convert

        Returns:
            :obj:`Optional[int]`: An optional id, :obj:`None` if out of vocabulary
        """
        ...
    def train(self, /, files: typing.Any, trainer: tokenizers.trainers.Trainer | None = None) -> typing.Any:
        """
        Train the Tokenizer using the given files.

        Reads the files line by line, while keeping all the whitespace, even new lines.
        If you want to train from data store in-memory, you can check
        :meth:`~tokenizers.Tokenizer.train_from_iterator`

        Args:
            files (:obj:`List[str]`):
                A list of path to the files that we should use for training

            trainer (:obj:`~tokenizers.trainers.Trainer`, `optional`):
                An optional trainer that should be used to train our Model
        """
        ...
    def train_from_iterator(
        self, /, iterator: typing.Any, trainer: tokenizers.trainers.Trainer | None = None, length: int | None = None
    ) -> typing.Any:
        """
        Train the Tokenizer using the provided iterator.

        You can provide anything that is a Python Iterator

            * A list of sequences :obj:`List[str]`
            * A generator that yields :obj:`str` or :obj:`List[str]`
            * A Numpy array of strings
            * ...

        Args:
            iterator (:obj:`Iterator`):
                Any iterator over strings or list of strings

            trainer (:obj:`~tokenizers.trainers.Trainer`, `optional`):
                An optional trainer that should be used to train our Model

            length (:obj:`int`, `optional`):
                The total number of sequences in the iterator. This is used to
                provide meaningful progress tracking
        """
        ...
    @property
    def truncation(self, /) -> typing.Any:
        """
        Get the currently set truncation parameters

        `Cannot set, use` :meth:`~tokenizers.Tokenizer.enable_truncation` `instead`

        Returns:
            (:obj:`dict`, `optional`):
                A dict with the current truncation parameters if truncation is enabled
        """
        ...

def __getattr__(name: str) -> _typeshed.Incomplete: ...
