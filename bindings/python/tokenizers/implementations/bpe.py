from tokenizers import Tokenizer, pre_tokenizers, decoders
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC, Sequence, Lowercase
from .base_tokenizer import BaseTokenizer

from typing import Optional

class BPETokenizer(BaseTokenizer):
    """ Original BPE Tokenizer

    Represents the BPE algorithm, as introduced by Rico Sennrich (https://arxiv.org/abs/1508.07909)
    """

    def __init__(self,
                 vocab_file: Optional[str]=None,
                 merges_file: Optional[str]=None,
                 unk_token: Optional[str]="<unk>",
                 suffix: Optional[str]="</w>",
                 dropout: Optional[float]=None):
        if vocab_file is not None and merges_file is not None:
            tokenizer = Tokenizer(BPE.from_files(vocab_file,
                                                 merges_file,
                                                 dropout=dropout,
                                                 unk_token=unk_token,
                                                 end_of_word_suffix=suffix))
        else:
            tokenizer = Tokenizer(BPE.empty())

        tokenizer.normalizer = Sequence.new([
            NFKC.new(),
            Lowercase.new()
        ])
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace.new()
        tokenizer.decoder = decoders.BPEDecoder.new(suffix=suffix)

        parameters = {
            "model": "BPE",
            "unk_token": unk_token,
            "suffix": suffix,
            "dropout": dropout,
        }

        super().__init__(tokenizer, parameters)
