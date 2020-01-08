from tokenizers import Tokenizer, pre_tokenizers, decoders
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from .base_tokenizer import BaseTokenizer

from typing import Optional

class SentencePieceBPETokenizer(BaseTokenizer):
    """ SentencePiece BPE Tokenizer

    Represents the BPE algorithm, with the pretokenization used by SentencePiece
    """

    def __init__(self,
                 vocab_file: Optional[str]=None,
                 merges_file: Optional[str]=None,
                 unk_token: str="<unk>",
                 replacement: str="▁",
                 add_prefix_space: bool=True,
                 dropout: Optional[float]=None):
        if vocab_file is not None and merges_file is not None:
            tokenizer = Tokenizer(BPE.from_files(vocab_file,
                                                 merges_file,
                                                 dropout=dropout,
                                                 unk_token=unk_token))
        else:
            tokenizer = Tokenizer(BPE.empty())

        tokenizer.normalizer = NFKC.new()
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace.new(replacement=replacement,
                                                               add_prefix_space=add_prefix_space)
        tokenizer.decoder = decoders.Metaspace.new(replacement=replacement,
                                                   add_prefix_space=add_prefix_space)

        parameters = {
            "model": "SentencePieceBPE",
            "unk_token": unk_token,
            "replacement": replacement,
            "add_prefix_space": add_prefix_space,
            "dropout": dropout,
        }

        super().__init__(tokenizer, parameters)
