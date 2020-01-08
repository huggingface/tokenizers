from tokenizers import Tokenizer, decoders
from tokenizers.models import WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.processors import BertProcessing
from .base_tokenizer import BaseTokenizer

from typing import Optional

class BertWordPieceTokenizer(BaseTokenizer):
    """ Bert WordPiece Tokenizer """

    def __init__(self,
                 vocab_file: Optional[str]=None,
                 add_special_tokens: bool=True,
                 unk_token: str="[UNK]",
                 sep_token: str="[SEP]",
                 cls_token: str="[CLS]",
                 clean_text: bool=True,
                 handle_chinese_chars: bool=True,
                 strip_accents: bool=True,
                 lowercase: bool=True,
                 prefix: str="##"):
        if vocab_file is not None:
            tokenizer = Tokenizer(WordPiece.from_files(vocab_file, unk_token=unk_token))
        else:
            tokenizer = Tokenizer(WordPiece.empty())

        tokenizer.normalizer = BertNormalizer.new(clean_text=clean_text,
                                                  handle_chinese_chars=handle_chinese_chars,
                                                  strip_accents=strip_accents,
                                                  lowercase=lowercase)
        tokenizer.pre_tokenizer = BertPreTokenizer.new()

        sep_token_id = tokenizer.token_to_id(sep_token)
        if sep_token_id is None:
            raise TypeError("sep_token not found in the vocabulary")
        cls_token_id = tokenizer.token_to_id(cls_token)
        if cls_token_id is None:
            raise TypeError("cls_token not found in the vocabulary")

        if add_special_tokens:
            tokenizer.post_processor = BertProcessing.new(
                (sep_token, sep_token_id),
                (cls_token, cls_token_id)
            )
        tokenizer.decoders = decoders.WordPiece.new(prefix=prefix)

        super().__init__(tokenizer)

