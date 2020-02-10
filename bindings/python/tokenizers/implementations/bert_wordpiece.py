from tokenizers import Tokenizer, decoders, trainers
from tokenizers.models import WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.processors import BertProcessing
from .base_tokenizer import BaseTokenizer

from typing import Optional, List, Union


class BertWordPieceTokenizer(BaseTokenizer):
    """ Bert WordPiece Tokenizer """

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        add_special_tokens: bool = True,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        cls_token: str = "[CLS]",
        clean_text: bool = True,
        handle_chinese_chars: bool = True,
        strip_accents: bool = True,
        lowercase: bool = True,
        wordpieces_prefix: str = "##",
    ):

        if vocab_file is not None:
            tokenizer = Tokenizer(WordPiece.from_files(vocab_file, unk_token=unk_token))
        else:
            tokenizer = Tokenizer(WordPiece.empty())

        tokenizer.add_special_tokens([unk_token, sep_token, cls_token])
        tokenizer.normalizer = BertNormalizer(
            clean_text=clean_text,
            handle_chinese_chars=handle_chinese_chars,
            strip_accents=strip_accents,
            lowercase=lowercase,
        )
        tokenizer.pre_tokenizer = BertPreTokenizer()

        if add_special_tokens and vocab_file is not None:
            sep_token_id = tokenizer.token_to_id(sep_token)
            if sep_token_id is None:
                raise TypeError("sep_token not found in the vocabulary")
            cls_token_id = tokenizer.token_to_id(cls_token)
            if cls_token_id is None:
                raise TypeError("cls_token not found in the vocabulary")

            tokenizer.post_processor = BertProcessing(
                (sep_token, sep_token_id), (cls_token, cls_token_id)
            )
        tokenizer.decoders = decoders.WordPiece(prefix=wordpieces_prefix)

        parameters = {
            "model": "BertWordPiece",
            "add_special_tokens": add_special_tokens,
            "unk_token": unk_token,
            "sep_token": sep_token,
            "cls_token": cls_token,
            "clean_text": clean_text,
            "handle_chinese_chars": handle_chinese_chars,
            "strip_accents": strip_accents,
            "lowercase": lowercase,
            "wordpieces_prefix": wordpieces_prefix,
        }

        super().__init__(tokenizer, parameters)

    def train(
        self,
        files: Union[str, List[str]],
        vocab_size: int = 30000,
        min_frequency: int = 2,
        limit_alphabet: int = 1000,
        initial_alphabet: List[str] = [],
        special_tokens: List[str] = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        show_progress: bool = True,
        wordpieces_prefix: str = "##",
    ):
        """ Train the model using the given files """

        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            limit_alphabet=limit_alphabet,
            initial_alphabet=initial_alphabet,
            special_tokens=special_tokens,
            show_progress=show_progress,
            continuing_subword_prefix=wordpieces_prefix,
        )
        if isinstance(files, str):
            files = [files]
        self._tokenizer.train(trainer, files)
