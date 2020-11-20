# Generated content DO NOT EDIT
class Decoder:
    """
    Base class for all decoders

    This class is not supposed to be instantiated directly. Instead, any implementation of
    a Decoder will return an instance of this class when instantiated.
    """

    def decode(self, tokens):
        """
        Decode the given list of tokens to a final string

        Args:
            tokens (:obj:`List[str]`):
                The list of tokens to decode

        Returns:
            :obj:`str`: The decoded string
        """
        pass

class BPEDecoder(Decoder):
    """
    BPEDecoder Decoder

    Args:
        suffix (:obj:`str`, `optional`, defaults to :obj:`</w>`):
            The suffix that was used to caracterize an end-of-word. This suffix will
            be replaced by whitespaces during the decoding
    """

    def __init__(self, suffix="</w>"):
        pass
    def decode(self, tokens):
        """
        Decode the given list of tokens to a final string

        Args:
            tokens (:obj:`List[str]`):
                The list of tokens to decode

        Returns:
            :obj:`str`: The decoded string
        """
        pass

class ByteLevel(Decoder):
    """
    ByteLevel Decoder

    This decoder is to be used in tandem with the :class:`~tokenizers.pre_tokenizers.ByteLevel`
    :class:`~tokenizers.pre_tokenizers.PreTokenizer`.
    """

    def __init__(self):
        pass
    def decode(self, tokens):
        """
        Decode the given list of tokens to a final string

        Args:
            tokens (:obj:`List[str]`):
                The list of tokens to decode

        Returns:
            :obj:`str`: The decoded string
        """
        pass

class Metaspace(Decoder):
    """
    Metaspace Decoder

    Args:
        replacement (:obj:`str`, `optional`, defaults to :obj:`▁`):
            The replacement character. Must be exactly one character. By default we
            use the `▁` (U+2581) meta symbol (Same as in SentencePiece).

        add_prefix_space (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to add a space to the first word if there isn't already one. This
            lets us treat `hello` exactly like `say hello`.
    """

    def __init__(self, replacement="▁", add_prefix_space=True):
        pass
    def decode(self, tokens):
        """
        Decode the given list of tokens to a final string

        Args:
            tokens (:obj:`List[str]`):
                The list of tokens to decode

        Returns:
            :obj:`str`: The decoded string
        """
        pass

class WordPiece(Decoder):
    """
    WordPiece Decoder

    Args:
        prefix (:obj:`str`, `optional`, defaults to :obj:`##`):
            The prefix to use for subwords that are not a beginning-of-word

        cleanup (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to cleanup some tokenization artifacts. Mainly spaces before punctuation,
            and some abbreviated english forms.
    """

    def __init__(self, prefix="##", cleanup=True):
        pass
    def decode(self, tokens):
        """
        Decode the given list of tokens to a final string

        Args:
            tokens (:obj:`List[str]`):
                The list of tokens to decode

        Returns:
            :obj:`str`: The decoded string
        """
        pass
