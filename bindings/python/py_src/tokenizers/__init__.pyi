# Generated content DO NOT EDIT
class AddedToken:
    """
    Represents a token that can be be added to a :class:`~tokenizers.Tokenizer`.
    It can have special options that defines the way it should behave.

    Args:
        content (:obj:`str`): The content of the token

        single_word (:obj:`bool`, defaults to :obj:`False`):
            Defines whether this token should only match single words. If :obj:`True`, this
            token will never match inside of a word. For example the token ``ing`` would match
            on ``tokenizing`` if this option is :obj:`False`, but not if it is :obj:`True`.
            The notion of "`inside of a word`" is defined by the word boundaries pattern in
            regular expressions (ie. the token should start and end with word boundaries).

        lstrip (:obj:`bool`, defaults to :obj:`False`):
            Defines whether this token should strip all potential whitespaces on its left side.
            If :obj:`True`, this token will greedily match any whitespace on its left. For
            example if we try to match the token ``[MASK]`` with ``lstrip=True``, in the text
            ``"I saw a [MASK]"``, we would match on ``" [MASK]"``. (Note the space on the left).

        rstrip (:obj:`bool`, defaults to :obj:`False`):
            Defines whether this token should strip all potential whitespaces on its right
            side. If :obj:`True`, this token will greedily match any whitespace on its right.
            It works just like :obj:`lstrip` but on the right.

        normalized (:obj:`bool`, defaults to :obj:`True` with :meth:`~tokenizers.Tokenizer.add_tokens` and :obj:`False` with :meth:`~tokenizers.Tokenizer.add_special_tokens`):
            Defines whether this token should match against the normalized version of the input
            text. For example, with the added token ``"yesterday"``, and a normalizer in charge of
            lowercasing the text, the token could be extract from the input ``"I saw a lion
            Yesterday"``.

    """

    def __init__(self, content, single_word=False, lstrip=False, rstrip=False, normalized=True):
        pass

class Encoding:
    """
    The :class:`~tokenizers.Encoding` represents the output of a :class:`~tokenizers.Tokenizer`.
    """

    def char_to_token(self, char_pos, sequence_index=0):
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
        pass
    def char_to_word(self, char_pos, sequence_index=0):
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
        pass
    @staticmethod
    def merge(encodings, growing_offsets=True):
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
        pass
    def pad(self, length, direction="right", pad_id=0, pad_type_id=0, pad_token="[PAD]"):
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
        pass
    def set_sequence_id(self, sequence_id):
        """
        Set the given sequence index

        Set the given sequence index for the whole range of tokens contained in this
        :class:`~tokenizers.Encoding`.
        """
        pass
    def token_to_chars(self, token_index):
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
        pass
    def token_to_sequence(self, token_index):
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
        pass
    def token_to_word(self, token_index):
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
        pass
    def truncate(self, max_length, stride=0):
        """
        Truncate the :class:`~tokenizers.Encoding` at the given length

        If this :class:`~tokenizers.Encoding` represents multiple sequences, when truncating
        this information is lost. It will be considered as representing a single sequence.

        Args:
            max_length (:obj:`int`):
                The desired length

            stride (:obj:`int`, defaults to :obj:`0`):
                The length of previous content to be included in each overflowing piece
        """
        pass
    def word_to_chars(self, word_index, sequence_index=0):
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
        pass
    def word_to_tokens(self, word_index, sequence_index=0):
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
        pass

class NormalizedString:
    pass

class PreTokenizedString:
    pass

class Regex:
    pass

class Token:
    pass

class Tokenizer:
    """
    A :obj:`Tokenizer` works as a pipeline. It processes some raw text as input
    and outputs an :class:`~tokenizers.Encoding`.

    Args:
        model (:class:`~tokenizers.models.Model`):
            The core algorithm that this :obj:`Tokenizer` should be using.

    """

    def __init__(model):
        pass
    def add_special_tokens(self, tokens):
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
        pass
    def add_tokens(self, tokens):
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
        pass
    def decode(self, ids, skip_special_tokens=True):
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
        pass
    def decode_batch(self, sequences, skip_special_tokens=True):
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
        pass
    def enable_padding(
        self,
        direction="right",
        pad_id=0,
        pad_type_id=0,
        pad_token="[PAD]",
        length=None,
        pad_to_multiple_of=None,
    ):
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
        pass
    def enable_truncation(self, max_length, stride=0, strategy="longest_first"):
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
        """
        pass
    def encode(self, sequence, pair=None, is_pretokenized=False, add_special_tokens=True):
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
        pass
    def encode_batch(self, input, is_pretokenized=False, add_special_tokens=True):
        """
        Encode the given batch of inputs. This method accept both raw text sequences
        as well as already pre-tokenized sequences.

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
        pass
    @staticmethod
    def from_buffer(buffer):
        """
        Instantiate a new :class:`~tokenizers.Tokenizer` from the given buffer.

        Args:
            buffer (:obj:`bytes`):
                A buffer containing a previously serialized :class:`~tokenizers.Tokenizer`

        Returns:
            :class:`~tokenizers.Tokenizer`: The new tokenizer
        """
        pass
    @staticmethod
    def from_file(path):
        """
        Instantiate a new :class:`~tokenizers.Tokenizer` from the file at the given path.

        Args:
            path (:obj:`str`):
                A path to a local JSON file representing a previously serialized
                :class:`~tokenizers.Tokenizer`

        Returns:
            :class:`~tokenizers.Tokenizer`: The new tokenizer
        """
        pass
    @staticmethod
    def from_str(json):
        """
        Instantiate a new :class:`~tokenizers.Tokenizer` from the given JSON string.

        Args:
            json (:obj:`str`):
                A valid JSON string representing a previously serialized
                :class:`~tokenizers.Tokenizer`

        Returns:
            :class:`~tokenizers.Tokenizer`: The new tokenizer
        """
        pass
    def get_vocab(self, with_added_tokens=True):
        """
        Get the underlying vocabulary

        Args:
            with_added_tokens (:obj:`bool`, defaults to :obj:`True`):
                Whether to include the added tokens

        Returns:
            :obj:`Dict[str, int]`: The vocabulary
        """
        pass
    def get_vocab_size(self, with_added_tokens=True):
        """
        Get the size of the underlying vocabulary

        Args:
            with_added_tokens (:obj:`bool`, defaults to :obj:`True`):
                Whether to include the added tokens

        Returns:
            :obj:`int`: The size of the vocabulary
        """
        pass
    def id_to_token(self, id):
        """
        Convert the given id to its corresponding token if it exists

        Args:
            id (:obj:`int`):
                The id to convert

        Returns:
            :obj:`Optional[str]`: An optional token, :obj:`None` if out of vocabulary
        """
        pass
    def no_padding(self):
        """
        Disable padding
        """
        pass
    def no_truncation(self):
        """
        Disable truncation
        """
        pass
    def post_process(self, encoding, pair=None, add_special_tokens=True):
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
        pass
    def save(self, pretty=False):
        """
        Save the :class:`~tokenizers.Tokenizer` to the file at the given path.

        Args:
            path (:obj:`str`):
                A path to a file in which to save the serialized tokenizer.

            pretty (:obj:`bool`, defaults to :obj:`False`):
                Whether the JSON file should be pretty formatted.
        """
        pass
    def to_str(self, pretty=False):
        """
        Gets a serialized string representing this :class:`~tokenizers.Tokenizer`.

        Args:
            pretty (:obj:`bool`, defaults to :obj:`False`):
                Whether the JSON string should be pretty formatted.

        Returns:
            :obj:`str`: A string representing the serialized Tokenizer
        """
        pass
    def token_to_id(self, token):
        """
        Convert the given token to its corresponding id if it exists

        Args:
            token (:obj:`str`):
                The token to convert

        Returns:
            :obj:`Optional[int]`: An optional id, :obj:`None` if out of vocabulary
        """
        pass
