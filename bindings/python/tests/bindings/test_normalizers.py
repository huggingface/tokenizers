import pickle

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import Normalizer, BertNormalizer, Sequence, Lowercase, Strip


class TestBertNormalizer:
    def test_instantiate(self):
        assert isinstance(BertNormalizer(), Normalizer)
        assert isinstance(BertNormalizer(), BertNormalizer)
        assert isinstance(pickle.loads(pickle.dumps(BertNormalizer())), BertNormalizer)

    def test_strip_accents(self):
        normalizer = BertNormalizer(
            strip_accents=True, lowercase=False, handle_chinese_chars=False, clean_text=False
        )

        output = normalizer.normalize_str("Héllò")
        assert output == "Hello"

    def test_handle_chinese_chars(self):
        normalizer = BertNormalizer(
            strip_accents=False, lowercase=False, handle_chinese_chars=True, clean_text=False
        )

        output = normalizer.normalize_str("你好")
        assert output == " 你  好 "

    def test_clean_text(self):
        normalizer = BertNormalizer(
            strip_accents=False, lowercase=False, handle_chinese_chars=False, clean_text=True
        )

        output = normalizer.normalize_str("\ufeffHello")
        assert output == "Hello"

    def test_lowercase(self):
        normalizer = BertNormalizer(
            strip_accents=False, lowercase=True, handle_chinese_chars=False, clean_text=False
        )

        output = normalizer.normalize_str("Héllò")
        assert output == "héllò"


class TestSequence:
    def test_instantiate(self):
        assert isinstance(Sequence([]), Normalizer)
        assert isinstance(Sequence([]), Sequence)
        assert isinstance(pickle.loads(pickle.dumps(Sequence([]))), Sequence)

    def test_can_make_sequences(self):
        normalizer = Sequence([Lowercase(), Strip()])

        output = normalizer.normalize_str("  HELLO  ")
        assert output == "hello"


class TestLowercase:
    def test_instantiate(self):
        assert isinstance(Lowercase(), Normalizer)
        assert isinstance(Lowercase(), Lowercase)
        assert isinstance(pickle.loads(pickle.dumps(Lowercase())), Lowercase)

    def test_lowercase(self):
        normalizer = Lowercase()

        output = normalizer.normalize_str("HELLO")
        assert output == "hello"


class TestStrip:
    def test_instantiate(self):
        assert isinstance(Strip(), Normalizer)
        assert isinstance(Strip(), Strip)
        assert isinstance(pickle.loads(pickle.dumps(Strip())), Strip)

    def test_left_strip(self):
        normalizer = Strip(left=True, right=False)

        output = normalizer.normalize_str("  hello  ")
        assert output == "hello  "

    def test_right_strip(self):
        normalizer = Strip(left=False, right=True)

        output = normalizer.normalize_str("  hello  ")
        assert output == "  hello"

    def test_full_strip(self):
        normalizer = Strip(left=True, right=True)

        output = normalizer.normalize_str("  hello  ")
        assert output == "hello"
