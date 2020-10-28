from ..utils import data_dir, doc_wiki_tokenizer
from tokenizers import Tokenizer


class TestPipeline:
    def test_pipeline(self, doc_wiki_tokenizer):
        def print(*args, **kwargs):
            pass

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
