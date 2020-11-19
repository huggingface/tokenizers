from ..utils import data_dir, doc_wiki_tokenizer, doc_pipeline_bert_tokenizer
from tokenizers import Tokenizer


disable_printing = True
original_print = print


def print(*args, **kwargs):
    if not disable_printing:
        original_print(*args, **kwargs)


class TestPipeline:
    def test_pipeline(self, doc_wiki_tokenizer):
        try:
            # START reload_tokenizer
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file("data/tokenizer-wiki.json")
            # END reload_tokenizer
        except Exception:
            tokenizer = Tokenizer.from_file(doc_wiki_tokenizer)

        # START setup_normalizer
        from tokenizers import normalizers
        from tokenizers.normalizers import NFD, StripAccents

        normalizer = normalizers.Sequence([NFD(), StripAccents()])
        # END setup_normalizer
        # START test_normalizer
        normalizer.normalize_str("Héllò hôw are ü?")
        # "Hello how are u?"
        # END test_normalizer
        assert normalizer.normalize_str("Héllò hôw are ü?") == "Hello how are u?"
        # START replace_normalizer
        tokenizer.normalizer = normalizer
        # END replace_normalizer
        # START setup_pre_tokenizer
        from tokenizers.pre_tokenizers import Whitespace

        pre_tokenizer = Whitespace()
        pre_tokenizer.pre_tokenize_str("Hello! How are you? I'm fine, thank you.")
        # [("Hello", (0, 5)), ("!", (5, 6)), ("How", (7, 10)), ("are", (11, 14)), ("you", (15, 18)),
        #  ("?", (18, 19)), ("I", (20, 21)), ("'", (21, 22)), ('m', (22, 23)), ("fine", (24, 28)),
        #  (",", (28, 29)), ("thank", (30, 35)), ("you", (36, 39)), (".", (39, 40))]
        # END setup_pre_tokenizer
        assert pre_tokenizer.pre_tokenize_str("Hello! How are you? I'm fine, thank you.") == [
            ("Hello", (0, 5)),
            ("!", (5, 6)),
            ("How", (7, 10)),
            ("are", (11, 14)),
            ("you", (15, 18)),
            ("?", (18, 19)),
            ("I", (20, 21)),
            ("'", (21, 22)),
            ("m", (22, 23)),
            ("fine", (24, 28)),
            (",", (28, 29)),
            ("thank", (30, 35)),
            ("you", (36, 39)),
            (".", (39, 40)),
        ]
        # START combine_pre_tokenizer
        from tokenizers import pre_tokenizers
        from tokenizers.pre_tokenizers import Digits

        pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(individual_digits=True)])
        pre_tokenizer.pre_tokenize_str("Call 911!")
        # [("Call", (0, 4)), ("9", (5, 6)), ("1", (6, 7)), ("1", (7, 8)), ("!", (8, 9))]
        # END combine_pre_tokenizer
        assert pre_tokenizer.pre_tokenize_str("Call 911!") == [
            ("Call", (0, 4)),
            ("9", (5, 6)),
            ("1", (6, 7)),
            ("1", (7, 8)),
            ("!", (8, 9)),
        ]
        # START replace_pre_tokenizer
        tokenizer.pre_tokenizer = pre_tokenizer
        # END replace_pre_tokenizer
        # START setup_processor
        from tokenizers.processors import TemplateProcessing

        tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
        )
        # END setup_processor
        # START test_decoding
        output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
        print(output.ids)
        # [1, 27253, 16, 93, 11, 5097, 5, 7961, 5112, 6218, 0, 35, 2]

        tokenizer.decode([1, 27253, 16, 93, 11, 5097, 5, 7961, 5112, 6218, 0, 35, 2])
        # "Hello , y ' all ! How are you ?"
        # END test_decoding
        assert output.ids == [1, 27253, 16, 93, 11, 5097, 5, 7961, 5112, 6218, 0, 35, 2]
        assert (
            tokenizer.decode([1, 27253, 16, 93, 11, 5097, 5, 7961, 5112, 6218, 0, 35, 2])
            == "Hello , y ' all ! How are you ?"
        )

    @staticmethod
    def slow_train():
        # START bert_setup_tokenizer
        from tokenizers import Tokenizer
        from tokenizers.models import WordPiece

        bert_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        # END bert_setup_tokenizer
        # START bert_setup_normalizer
        from tokenizers import normalizers
        from tokenizers.normalizers import Lowercase, NFD, StripAccents

        bert_tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
        # END bert_setup_normalizer
        # START bert_setup_pre_tokenizer
        from tokenizers.pre_tokenizers import Whitespace

        bert_tokenizer.pre_tokenizer = Whitespace()
        # END bert_setup_pre_tokenizer
        # START bert_setup_processor
        from tokenizers.processors import TemplateProcessing

        bert_tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", 1),
                ("[SEP]", 2),
            ],
        )
        # END bert_setup_processor
        # START bert_train_tokenizer
        from tokenizers.trainers import WordPieceTrainer

        trainer = WordPieceTrainer(
            vocab_size=30522, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        )
        files = [f"data/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]
        bert_tokenizer.train(files, trainer)

        bert_tokenizer.save("data/bert-wiki.json")
        # END bert_train_tokenizer

    def test_bert_example(self, doc_pipeline_bert_tokenizer):
        try:
            bert_tokenizer = Tokenizer.from_file("data/bert-wiki.json")
        except Exception:
            bert_tokenizer = Tokenizer.from_file(doc_pipeline_bert_tokenizer)

        # START bert_test_decoding
        output = bert_tokenizer.encode("Welcome to the 🤗 Tokenizers library.")
        print(output.tokens)
        # ["[CLS]", "welcome", "to", "the", "[UNK]", "tok", "##eni", "##zer", "##s", "library", ".", "[SEP]"]

        bert_tokenizer.decode(output.ids)
        # "welcome to the tok ##eni ##zer ##s library ."
        # END bert_test_decoding
        assert bert_tokenizer.decode(output.ids) == "welcome to the tok ##eni ##zer ##s library ."
        # START bert_proper_decoding
        from tokenizers import decoders

        bert_tokenizer.decoder = decoders.WordPiece()
        bert_tokenizer.decode(output.ids)
        # "welcome to the tokenizers library."
        # END bert_proper_decoding
        assert bert_tokenizer.decode(output.ids) == "welcome to the tokenizers library."


if __name__ == "__main__":
    from urllib import request
    from zipfile import ZipFile
    import os

    disable_printing = False
    if not os.path.isdir("data/wikitext-103-raw"):
        print("Downloading wikitext-103...")
        wiki_text, _ = request.urlretrieve(
            "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"
        )
        with ZipFile(wiki_text, "r") as z:
            print("Unzipping in data...")
            z.extractall("data")

    print("Now training...")
    TestPipeline.slow_train()
