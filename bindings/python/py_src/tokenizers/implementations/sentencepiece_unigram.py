from tokenizers import (
    Tokenizer,
    AddedToken,
    pre_tokenizers,
    decoders,
    trainers,
    normalizers,
)
import os
from tokenizers.models import Unigram
import json
from .base_tokenizer import BaseTokenizer

from typing import Optional, List, Union


class SentencePieceUnigramTokenizer(BaseTokenizer):
    """SentencePiece Unigram Tokenizer

    Represents the Unigram algorithm, with the pretokenization used by SentencePiece
    """

    def __init__(
        self,
        vocab: Optional[str] = None,
        replacement: str = "▁",
        add_prefix_space: bool = True,
    ):
        if vocab is not None:
            # Let Unigram(..) fail if only one of them is None
            tokenizer = Tokenizer(Unigram(vocab))
        else:
            tokenizer = Tokenizer(Unigram())

        tokenizer.normalizer = normalizers.Sequence(
            [
                normalizers.Nmt(),
                normalizers.NFKC(),
            ]
        )
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
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            show_progress=show_progress,
        )

        if isinstance(files, str):
            files = [files]
        self._tokenizer.train(trainer, files)

    @staticmethod
    def from_spm(filename: str):
        try:
            import sys

            sys.path.append(".")

            import sentencepiece_model_pb2 as model
        except Exception:
            raise Exception(
                "You don't seem to have the required protobuf file, in order to use this function you need to run `pip install protobuf` and `wget https://raw.githubusercontent.com/google/sentencepiece/master/python/sentencepiece_model_pb2.py` for us to be able to read the intrinsics of your spm_file. `pip install sentencepiece` is not required."
            )

        m = model.ModelProto()
        m.ParseFromString(open(filename, "rb").read())

        precompiled_charsmap = m.normalizer_spec.precompiled_charsmap
        vocab = [(piece.piece, piece.score) for piece in m.pieces]
        unk_id = m.trainer_spec.unk_id
        model_type = m.trainer_spec.model_type
        if model_type != 1:
            raise Exception(
                "You're trying to run a `Unigram` model but you're file was trained with a different algorithm"
            )

        replacement = "▁"
        add_prefix_space = True

        tokenizer = Tokenizer(Unigram(vocab, unk_id))

        tokenizer.normalizer = normalizers.Precompiled(precompiled_charsmap)
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
        }

        obj = BaseTokenizer.__new__(SentencePieceUnigramTokenizer, tokenizer, parameters)
        BaseTokenizer.__init__(obj, tokenizer, parameters)
        return obj
