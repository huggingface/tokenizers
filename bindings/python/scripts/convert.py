import transformers
from tokenizers.implementations import SentencePieceUnigramTokenizer, BaseTokenizer
from tokenizers.processors import TemplateProcessing
from tokenizers.models import Unigram, BPE
from tokenizers import decoders
from tokenizers import Tokenizer
from tokenizers.normalizers import (
    StripAccents,
    NFKD,
    Lowercase,
    Sequence,
    BertNormalizer,
    Precompiled,
    Replace,
)
from tokenizers.pre_tokenizers import (
    Digits,
    WhitespaceSplit,
    Metaspace,
    Sequence as PSequence,
)
import json
import unicodedata
import sys
import os
import datetime
import argparse

sys.path.append(".")

from spm_parity_check import check_details
from sentencepiece_extractor import SentencePieceExtractor


def check_number_comma(piece: str) -> bool:
    return len(piece) < 2 or piece[-1] != "," or not piece[-2].isdigit()


def get_proto(filename: str):
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
    return m


class Converter:
    def __init__(self, original_tokenizer):
        self.original_tokenizer = original_tokenizer

    def converted(self) -> Tokenizer:
        raise NotImplementedError()


class SpmConverter(Converter):
    def __init__(self, *args):
        super().__init__(*args)
        self.proto = get_proto(self.original_tokenizer.vocab_file)

    def vocab(self, proto):
        return [(piece.piece, piece.score) for piece in proto.pieces]

    def unk_id(self, proto):
        return proto.trainer_spec.unk_id

    def tokenizer(self, proto):
        model_type = proto.trainer_spec.model_type
        vocab = self.vocab(proto)
        unk_id = self.unk_id(proto)
        if model_type == 1:
            tokenizer = Tokenizer(Unigram(vocab, unk_id))
        elif model_type == 2:
            vocab, merges = SentencePieceExtractor(self.original_tokenizer.vocab_file).extract()
            tokenizer = Tokenizer(
                BPE(vocab, merges, unk_token=proto.trainer_spec.unk_piece, fuse_unk=True,)
            )
        else:
            raise Exception(
                "You're trying to run a `Unigram` model but you're file was trained with a different algorithm"
            )

        return tokenizer

    def normalizer(self, proto):
        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
        return Precompiled(precompiled_charsmap)

    def post_processor(self, tokenizer):
        return None

    def converted(self):
        tokenizer = self.tokenizer(self.proto)

        # Tokenizer assemble
        tokenizer.normalizer = self.normalizer(self.proto)

        replacement = "▁"
        add_prefix_space = True
        tokenizer.pre_tokenizer = PSequence(
            [
                WhitespaceSplit(),
                Metaspace(replacement=replacement, add_prefix_space=add_prefix_space),
            ]
        )
        tokenizer.decoder = decoders.Metaspace(
            replacement=replacement, add_prefix_space=add_prefix_space
        )
        post_processor = self.post_processor(tokenizer)
        if post_processor:
            tokenizer.post_processor = post_processor

        # TODO what parameters should we give ?
        parameters = {}

        return BaseTokenizer(tokenizer, parameters)


class AlbertConverter(SpmConverter):
    def vocab(self, proto):
        return [
            (piece.piece, piece.score)
            if check_number_comma(piece.piece)
            else (piece.piece, piece.score - 100)
            for piece in proto.pieces
        ]

    def normalizer(self, proto):
        normalizers = [Replace("``", '"'), Replace("''", '"')]
        if not self.original_tokenizer.keep_accents:
            normalizers.append(NFKD())
            normalizers.append(StripAccents())
        if self.original_tokenizer.do_lower_case:
            normalizers.append(Lowercase())

        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
        normalizers.append(Precompiled(precompiled_charsmap))
        return Sequence(normalizers)

    def post_processor(self, tokenizer):
        return TemplateProcessing(
            seq_a=["[CLS]", "$0", "[SEP]"],
            seq_b=["$1", "[SEP]"],
            special_tokens=[
                ("[CLS]", tokenizer.get_vocab()["[CLS]"]),
                ("[SEP]", tokenizer.get_vocab()["[SEP]"]),
            ],
        )


class CamembertConverter(SpmConverter):
    def vocab(self, proto):
        vocab = [
            ("<s>NOTUSED", 0.0),
            ("<pad>", 0.0),
            ("</s>NOTUSED", 0.0),
            ("<unk>", 0.0),
        ]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces]
        return vocab

    def unk_id(self, proto):
        # See vocab unk position
        return 3

    def post_processor(self, tokenizer):
        return TemplateProcessing(
            seq_a=["<s>", "$0", "</s>"],
            seq_b=["$1", "</s>"],
            special_tokens=[
                ("<s>", tokenizer.get_vocab()["<s>"]),
                ("</s>", tokenizer.get_vocab()["</s>"]),
            ],
        )


class MBartConverter(SpmConverter):
    def vocab(self, proto):
        vocab = [
            ("<s>", 0.0),
            ("<pad>", 0.0),
            ("</s>", 0.0),
            ("<unk>", 0.0),
        ]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        vocab += [
            ("ar_AR", 0.0),
            ("cs_CZ", 0.0),
            ("de_DE", 0.0),
            ("en_XX", 0.0),
            ("es_XX", 0.0),
            ("et_EE", 0.0),
            ("fi_FI", 0.0),
            ("fr_XX", 0.0),
            ("gu_IN", 0.0),
            ("hi_IN", 0.0),
            ("it_IT", 0.0),
            ("ja_XX", 0.0),
            ("kk_KZ", 0.0),
            ("ko_KR", 0.0),
            ("lt_LT", 0.0),
            ("lv_LV", 0.0),
            ("my_MM", 0.0),
            ("ne_NP", 0.0),
            ("nl_XX", 0.0),
            ("ro_RO", 0.0),
            ("ru_RU", 0.0),
            ("si_LK", 0.0),
            ("tr_TR", 0.0),
            ("vi_VN", 0.0),
            ("zh_CN", 0.0),
        ]
        return vocab

    def unk_id(self, proto):
        return 3

    def post_processor(self, tokenizer):
        return TemplateProcessing(
            seq_a=["$0", "</s>", "en_XX"],
            seq_b=["$1", "</s>"],
            special_tokens=[
                ("en_XX", tokenizer.get_vocab()["en_XX"]),
                ("</s>", tokenizer.get_vocab()["</s>"]),
            ],
        )


class XLMRobertaConverter(SpmConverter):
    def vocab(self, proto):
        vocab = [
            ("<s>", 0.0),
            ("<pad>", 0.0),
            ("</s>", 0.0),
            ("<unk>", 0.0),
        ]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        return vocab

    def unk_id(self, proto):
        unk_id = 3
        return unk_id

    def post_processor(self, tokenizer):
        return TemplateProcessing(
            seq_a=["<s>", "$0", "</s>"],
            seq_b=["$1", "</s>"],
            special_tokens=[
                ("<s>", tokenizer.get_vocab()["<s>"]),
                ("</s>", tokenizer.get_vocab()["</s>"]),
            ],
        )


class XLNetConverter(SpmConverter):
    def vocab(self, proto):
        return [
            (piece.piece, piece.score)
            if check_number_comma(piece.piece)
            else (piece.piece, piece.score - 100)
            for piece in proto.pieces
        ]

    def normalizer(self, proto):
        normalizers = [Replace("``", '"'), Replace("''", '"')]
        if not self.original_tokenizer.keep_accents:
            normalizers.append(NFKD())
            normalizers.append(StripAccents())
        if self.original_tokenizer.do_lower_case:
            normalizers.append(Lowercase())

        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
        normalizers.append(Precompiled(precompiled_charsmap))
        return Sequence(normalizers)

    def post_processor(self, tokenizer):
        return TemplateProcessing(
            seq_a=["$0", "<sep>", "<cls>"],
            seq_b=["$1", "<sep>"],
            special_tokens=[
                ("<sep>", tokenizer.get_vocab()["<sep>"]),
                ("<cls>", tokenizer.get_vocab()["<cls>"]),
            ],
        )


class ReformerConverter(SpmConverter):
    pass


class PegasusConverter(SpmConverter):
    offset = 103

    def vocab(self, proto):
        vocab = [
            (self.original_tokenizer.pad_token, 0),
            (self.original_tokenizer.eos_token, 0),
        ]
        vocab += [(f"unk_{i}", -100) for i in range(2, 2 + self.offset)]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[2:]]
        return vocab

    def unk_id(self, proto):
        return proto.trainer_spec.unk_id + self.offset

    def post_processor(self, tokenizer):
        eos = self.original_tokenizer.eos_token
        return TemplateProcessing(
            seq_a=["$0", eos],
            seq_b=["$1", eos],
            special_tokens=[(eos, tokenizer.get_vocab()[eos]),],
        )


class T5Converter(SpmConverter):
    def post_processor(self, tokenizer):
        return TemplateProcessing(
            seq_a=["$0", "</s>"],
            seq_b=["$1", "</s>"],
            special_tokens=[("</s>", tokenizer.get_vocab()["</s>"]),],
        )


CONVERTERS = {
    "AlbertTokenizer": AlbertConverter,
    "CamembertTokenizer": CamembertConverter,
    "XLMRobertaTokenizer": XLMRobertaConverter,
    "MBartTokenizer": MBartConverter,
    "XLNetTokenizer": XLNetConverter,
    "ReformerTokenizer": ReformerConverter,
    "PegasusTokenizer": PegasusConverter,
    "T5Tokenizer": T5Converter,
}


def check(pretrained, filename):
    transformer_tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained)
    converter_class = CONVERTERS[transformer_tokenizer.__class__.__name__]
    tokenizer = converter_class(transformer_tokenizer).converted()

    now = datetime.datetime.now
    trans_total_time = datetime.timedelta(seconds=0)
    tok_total_time = datetime.timedelta(seconds=0)

    with open(filename, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()

            start = now()
            ids = transformer_tokenizer.encode(line)
            trans = now()
            tok_ids = tokenizer.encode(line).ids
            tok = now()

            trans_total_time += trans - start
            tok_total_time += tok - trans

            if ids != tok_ids:
                if check_details(line, ids, tok_ids, transformer_tokenizer, tokenizer):
                    continue
            assert ids == tok_ids, f"Error in line {i}: {line} {ids} != {tok_ids}"

    tokenizer.save(f"{pretrained.replace('/', '-')}.json")
    return ("OK", trans_total_time / tok_total_time)


def main():
    pretraineds = [
        "albert-base-v1",
        "albert-large-v1",
        "albert-xlarge-v1",
        "albert-xxlarge-v1",
        "albert-base-v2",
        "albert-large-v2",
        "albert-xlarge-v2",
        "albert-xxlarge-v2",
        "camembert-base",
        "xlm-roberta-base",
        "xlm-roberta-large",
        "xlm-roberta-large-finetuned-conll02-dutch",
        "xlm-roberta-large-finetuned-conll02-spanish",
        "xlm-roberta-large-finetuned-conll03-english",
        "xlm-roberta-large-finetuned-conll03-german",
        "facebook/mbart-large-en-ro",
        "facebook/mbart-large-cc25",
        "xlnet-base-cased",
        "xlnet-large-cased",
        "google/reformer-crime-and-punishment",
        "t5-small",
        "google/pegasus-large",
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename",
        required=True,
        type=str,
        help="The filename that we are going to encode in both versions to check that conversion worked",
    )
    parser.add_argument(
        "--models",
        type=lambda s: s.split(","),
        default=pretraineds,
        help=f"The pretrained tokenizers you want to test agains, (default: {pretraineds})",
    )
    args = parser.parse_args()

    print(args.filename)

    model_len = 50
    status_len = 6
    speedup_len = 8
    print(f"|{'Model':^{model_len}}|{'Status':^{status_len}}|{'Speedup':^{speedup_len}}|")
    print(f"|{'-'*model_len}|{'-'*status_len}|{'-'*speedup_len}|")
    for pretrained in args.models:
        status, speedup = check(pretrained, args.filename)
        print(
            f"|{pretrained:<{model_len}}|{status:^{status_len}}|{speedup:^{speedup_len - 1}.2f}x|"
        )


if __name__ == "__main__":
    main()
