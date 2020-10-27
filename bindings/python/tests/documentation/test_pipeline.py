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
