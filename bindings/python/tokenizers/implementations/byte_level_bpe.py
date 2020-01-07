from tokenizers import Tokenizer, pre_tokenizers, decoders
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from .base_tokenizer import BaseTokenizer

from typing import Optional

class ByteLevelBPETokenizer(BaseTokenizer):
    """ ByteLevelBPETokenizer

    Represents a Byte-level BPE as introduced by OpenAI with their GPT-2 model
    """

    def __init__(self,
                 vocab_file: Optional[str]=None,
                 merges_file: Optional[str]=None,
                 add_prefix_space: bool=False):
        if vocab_file is not None and merges_file is not None:
            tokenizer = Tokenizer(BPE.from_files(vocab_file, merges_file))
        else:
            tokenizer = Tokenizer(BPE.empty())

        tokenizer.normalizer = NFKC.new()
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel.new(add_prefix_space=add_prefix_space)
        tokenizer.decoder = decoders.ByteLevel.new()

        super().__init__(tokenizer)
