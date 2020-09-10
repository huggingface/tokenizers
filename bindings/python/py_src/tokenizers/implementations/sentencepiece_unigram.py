from tokenizers import (
    Tokenizer,
    AddedToken,
    pre_tokenizers,
    decoders,
    trainers,
    normalizers,
)
from tokenizers.models import Unigram
from .base_tokenizer import BaseTokenizer

from typing import Optional, List, Union


class SentencePieceUnigramTokenizer(BaseTokenizer):
    """SentencePiece Unigram Tokenizer

    Represents the Unigram algorithm, with the pretokenization used by SentencePiece
    """

    def __init__(
        self, vocab: Optional[str] = None, replacement: str = "▁", add_prefix_space: bool = True,
    ):
        if vocab is not None:
            tokenizer = Tokenizer(Unigram(vocab))
        else:
            tokenizer = Tokenizer(Unigram())

        tokenizer.normalizer = normalizers.Sequence(normalizers.Nmt(), normalizers.NFKC(),)
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.WhitespaceSplit(),
                pre_tokenizers.Metaspace(
                    replacement=replacement, add_prefix_space=add_prefix_space
                ),
            ]
        )
        tokenizer.decoder = decoders.Metaspace(
            replacement=replacement, add_prefix_space=add_prefix_space
        )

        parameters = {
            "model": "SentencePieceUnigram",
            "replacement": replacement,
            "add_prefix_space": add_prefix_space,
        }

        super().__init__(tokenizer, parameters)

    def train(
        self,
        files: Union[str, List[str]],
        vocab_size: int = 8000,
        show_progress: bool = True,
        special_tokens: List[Union[str, AddedToken]] = [],
    ):
        """ Train the model using the given files """

        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size, special_tokens=special_tokens, show_progress=show_progress,
        )

        if isinstance(files, str):
            files = [files]
        self._tokenizer.train(trainer, files)

    @staticmethod
    def from_spm(filename: str):
        try:
            import sentencepiece as spm
        except Exception:
            raise Exception(
                "We need `sentencepiece` package to load via this method try installing it with `pip install sentencepiece`"
            )

        sp = spm.SentencePieceProcessor()
        sp.Load(filename)

        from . import sentencepiece_model_pb2 as model

        m = model.ModelProto()
        m.ParseFromString(open(filename, "rb").read())

        precompiled_charsmap = m.normalizer_spec.precompiled_charsmap

        vocab = [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.piece_size())]
        tokenizer = Tokenizer(Unigram(vocab))
        tokenizer.normalizer = normalizers.Precompiled(precompiled_charsmap)

        parameters = {
            "model": "SentencePieceUnigram",
        }

        return super().__init__(tokenizer, parameters)
