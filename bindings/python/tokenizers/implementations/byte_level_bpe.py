from tokenizers import Tokenizer, pre_tokenizers, decoders, trainers
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from .base_tokenizer import BaseTokenizer

from typing import Optional, List, Union

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

        parameters = {
            "model": "ByteLevelBPE",
            "add_prefix_space": add_prefix_space,
        }

        super().__init__(tokenizer, parameters)

    def train(self, files: Union[str, List[str]],
              vocab_size: int=30000,
              min_frequency: int=2,
              show_progress: bool=True,
              special_tokens: List[str]=[]):
        """ Train the model using the given files """

        trainer = trainers.BpeTrainer.new(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=show_progress,
            special_tokens=special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )
        if isinstance(files, str):
            files = [files]
        self._tokenizer.train(trainer, files)
