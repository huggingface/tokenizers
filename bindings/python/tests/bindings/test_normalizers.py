import pickle
import pytest

from tokenizers import Tokenizer, NormalizedString
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

    def test_can_modify(self):
        normalizer = BertNormalizer(
            clean_text=True, handle_chinese_chars=True, strip_accents=True, lowercase=True
        )

        assert normalizer.clean_text == True
        assert normalizer.handle_chinese_chars == True
        assert normalizer.strip_accents == True
        assert normalizer.lowercase == True

        # Modify these
        normalizer.clean_text = False
        assert normalizer.clean_text == False
        normalizer.handle_chinese_chars = False
        assert normalizer.handle_chinese_chars == False
        normalizer.strip_accents = None
        assert normalizer.strip_accents == None
        normalizer.lowercase = False
        assert normalizer.lowercase == False


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

    def test_can_modify(self):
        normalizer = Strip(left=True, right=True)

        assert normalizer.left == True
        assert normalizer.right == True

        # Modify these
        normalizer.left = False
        assert normalizer.left == False
        normalizer.right = False
        assert normalizer.right == False


class TestCustomNormalizer:
    class BadCustomNormalizer:
        def normalize(self, normalized, wrong):
            pass

    class GoodCustomNormalizer:
        def normalize(self, normalized):
            self.kept_normalized = normalized
            normalized.replace("there", "you")

        def use_after_normalize(self):
            self.kept_normalized.replace("something", "else")

    def test_instantiate(self):
        bad = Normalizer.custom(TestCustomNormalizer.BadCustomNormalizer())
        good_custom = TestCustomNormalizer.GoodCustomNormalizer()
        good = Normalizer.custom(good_custom)

        assert isinstance(bad, Normalizer)
        assert isinstance(good, Normalizer)
        with pytest.raises(Exception, match="TypeError: normalize()"):
            bad.normalize_str("Hey there!")
        assert good.normalize_str("Hey there!") == "Hey you!"
        with pytest.raises(
            Exception, match="Cannot use a NormalizedStringRefMut outside `normalize`"
        ):
            good_custom.use_after_normalize()

    def test_normalizer_interface(self):
        normalizer = Normalizer.custom(TestCustomNormalizer.GoodCustomNormalizer())

        normalized = NormalizedString("Hey there!")
        normalizer.normalize(normalized)

        assert repr(normalized) == 'NormalizedString(original="Hey there!", normalized="Hey you!")'
        assert str(normalized) == "Hey you!"
