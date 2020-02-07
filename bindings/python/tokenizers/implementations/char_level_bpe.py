from .. import Tokenizer, pre_tokenizers, decoders, trainers
from ..models import BPE
from ..normalizers import Sequence, Lowercase, unicode_normalizer_from_str
from .base_tokenizer import BaseTokenizer

from typing import Optional, List, Union


class CharBPETokenizer(BaseTokenizer):
    """ Original BPE Tokenizer

    Represents the BPE algorithm, as introduced by Rico Sennrich (https://arxiv.org/abs/1508.07909)
    """

    def __init__(self,
                 vocab_file: Optional[str]=None,
                 merges_file: Optional[str]=None,
                 unk_token: Optional[str]="<unk>",
                 suffix: Optional[str]="</w>",
                 dropout: Optional[float]=None,
                 do_lowercase: bool = False,
                 unicode_normalizer: Optional[str] = None):
        if vocab_file is not None and merges_file is not None:
            tokenizer = Tokenizer(
                BPE.from_files(
                    vocab_file,
                    merges_file,
                    dropout=dropout,
                    unk_token=unk_token,
                    end_of_word_suffix=suffix
                )
            )
        else:
            tokenizer = Tokenizer(BPE.empty())

        tokenizer.add_special_tokens([ unk_token ])

        # Check for Unicode normalization first (before everything else)
        normalizers = []

        if unicode_normalizer:
            normalizers += [unicode_normalizer_from_str(unicode_normalizer)]

        if do_lowercase:
            normalizers += [Lowercase()]

        # Create the normalizer structure
        if len(normalizers) > 0:
            if len(normalizers) > 1:
                tokenizer.normalizer = Sequence(normalizers)
            else:
                tokenizer.normalizer = normalizers[0]

        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer.decoder = decoders.BPEDecoder(suffix=suffix)

        parameters = {
            "model": "BPE",
            "unk_token": unk_token,
            "suffix": suffix,
            "dropout": dropout,
        }

        super().__init__(tokenizer, parameters)

    def train(
        self,
        files: Union[str, List[str]],
        vocab_size: int = 30000,
        min_frequency: int = 2,
        special_tokens: List[str] = ["<unk>"],
        limit_alphabet: int = 1000,
        initial_alphabet: List[str] = [],
        suffix: Optional[str] = "</w>",
        show_progress: bool = True,
    ):
        """ Train the model using the given files """

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            limit_alphabet=limit_alphabet,
            initial_alphabet=initial_alphabet,
            end_of_word_suffix=suffix,
            show_progress=show_progress,
        )
        if isinstance(files, str):
            files = [files]
        self._tokenizer.train(trainer, files)
