from .. import Encoding
from typing import Tuple, Union, List

class PostProcessor:
    """Base class for all post-processors

    This class is not supposed to be instantiated directly. Instead, any implementation of
    a PostProcessor will return an instance of this class when instantiated.
    """

    def num_special_tokens_to_add(self, is_pair: bool) -> int:
        """
        Return the number of special tokens that would be added for single/pair sentences.
        :param is_pair: Boolean indicating if the input would be a single sentence or a pair
        :return:
        """
        pass
    def process(
        self, encoding: Encoding, pair: Optional[Encoding] = None, add_special_tokens: bool = True
    ) -> Encoding:
        """ Post-process the given encodings, generating the final one """
        pass

class BertProcessing(PostProcessor):
    """BertProcessing

    This post-processor takes care of adding the special tokens needed by
    a Bert model:
        - a SEP token
        - a CLS token
    """

    def __init__(self, sep: Tuple[str, int], cls: Tuple[str, int]) -> None:
        """Instantiate a new BertProcessing with the given tokens

        Args:
            sep: Tuple[str, int]:
                A tuple with the string representation of the SEP token, and its id

            cls: Tuple[str, int]:
                A tuple with the string representation of the CLS token, and its id

        Returns:
            PostProcessor
        """
        pass

class RobertaProcessing(PostProcessor):
    """RobertaProcessing

    This post-processor takes care of adding the special tokens needed by
    a Roberta model:
        - a SEP token
        - a CLS token

    It also takes care of trimming the offsets.
    By default, the ByteLevel BPE might include whitespaces in the produced tokens. If you don't
    want the offsets to include these whitespaces, then this PostProcessor should be initialized
    with `trim_offsets=True`
    """

    def __init__(
        self,
        sep: Tuple[str, int],
        cls: Tuple[str, int],
        trim_offsets: bool = True,
        add_prefix_space: bool = True,
    ) -> None:
        """Instantiate a new RobertaProcessing with the given tokens

        Args:
            sep: Tuple[str, int]:
                A tuple with the string representation of the SEP token, and its id

            cls: Tuple[str, int]:
                A tuple with the string representation of the CLS token, and its id

            trim_offsets: bool:
                Whether to trim the whitespaces from the produced offsets.

            add_prefix_space: bool:
                Whether the add_prefix_space option was enabled during pre-tokenization. This
                is relevant because it defines the way the offsets are trimmed out.

        Returns:
            PostProcessor
        """
        pass

class ByteLevel(PostProcessor):
    """ByteLevel Post processing

    This post-processor takes care of trimming the offsets.
    By default, the ByteLevel BPE might include whitespaces in the produced tokens. If you don't
    want the offsets to include these whitespaces, then this PostProcessor must be used.
    """

    def __init__(self, trim_offsets: bool = True) -> None:
        """Instantiate a new ByteLevel

        Args:
            trim_offsets: bool:
                Whether to trim the whitespaces from the produced offsets.
        """
        pass

Template = Union[str, List[str]]
Tokens = List[Union[Tuple[int, str], Tuple[str, int], dict]]

class TemplateProcessing(PostProcessor):
    """TemplateProcessing

    Provides a way to specify templates in order to add the special tokens to each
    input sequence as relevant.

    Let's take `BERT` tokenizer as an example. It uses two special tokens, used to
    delimitate each sequence. `[CLS]` is always used at the beginning of the first
    sequence, and `[SEP]` is added at the end of both the first, and the pair
    sequences. The final result looks like this:
        - Single sequence: `[CLS] Hello there [SEP]`
        - Pair sequences: `[CLS] My name is Anthony [SEP] What is my name? [SEP]`
    With the type ids as following:
    ```markdown
    [CLS]   ...   [SEP]   ...   [SEP]
      0      0      0      1      1
    ```

    You can achieve such behavior using a TemplateProcessing:
    ```
    TemplateProcessing(
        single="[CLS] $0 [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 1), ("[SEP]", 0)],
    )
    ```

    In this example, each input sequence is identified using a `$` construct. This identifier
    lets us specify each input sequence, and the type_id to use. When nothing is specified,
    it uses the default values. Here are the different ways to specify it:
    - Specifying the sequence, with default `type_id == 0`: `$A` or `$B`
    - Specifying the `type_id` with default `sequence == A`: `$0`, `$1`, `$2`, ...
    - Specifying both: `$A:0`, `$B:1`, ...

    The same construct is used for special tokens: `<identifier>(:<type_id>)?`.

    **Warning**: You must ensure that you are giving the correct tokens/ids as these
    will be added to the Encoding without any further check. If the given ids correspond
    to something totally different in a `Tokenizer` using this `PostProcessor`, it
    might lead to unexpected results.
    """

    def __init__(self, single: Template, pair: Template, special_tokens: Tokens) -> None:
        """Instantiate a new TemplateProcessing

        Args:
            single: Template
                The template used for single sequences

            pair: Template:
                The template used when both sequences are specified

            special_tokens: Tokens:
                The list of special tokens used in each sequences

        Template: Union[str, List[str]]:
            - If a `str` is provided, the whitespace is used as delimiter between tokens
            - If a `List[str]` is provided, a list of tokens

        Tokens: List[Union[Tuple[int, str], Tuple[str, int], dict]]:
            - A Tuple with both a token and its associated ID, in any order
            - A dict with the following keys:
                - "id": str => The special token id, as specified in the Template
                - "ids": List[int] => The associated IDs
                - "tokens": List[str] => The associated tokens
             The given dict expects the provided `ids` and `tokens` lists to have
             the same length.
        """
        pass
