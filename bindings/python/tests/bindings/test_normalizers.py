import pickle

import pytest

from tokenizers import NormalizedString
from tokenizers.normalizers import (
    BertNormalizer,
    Lowercase,
    Normalizer,
    Precompiled,
    Sequence,
    Strip,
    Prepend,
    Replace,
    UnicodeFilter,
)


class TestBertNormalizer:
    def test_instantiate(self):
        assert isinstance(BertNormalizer(), Normalizer)
        assert isinstance(BertNormalizer(), BertNormalizer)
        assert isinstance(pickle.loads(pickle.dumps(BertNormalizer())), BertNormalizer)

    def test_strip_accents(self):
        normalizer = BertNormalizer(strip_accents=True, lowercase=False, handle_chinese_chars=False, clean_text=False)

        output = normalizer.normalize_str("Héllò")
        assert output == "Hello"

    def test_handle_chinese_chars(self):
        normalizer = BertNormalizer(strip_accents=False, lowercase=False, handle_chinese_chars=True, clean_text=False)

        output = normalizer.normalize_str("你好")
        assert output == " 你  好 "

    def test_clean_text(self):
        normalizer = BertNormalizer(strip_accents=False, lowercase=False, handle_chinese_chars=False, clean_text=True)

        output = normalizer.normalize_str("\ufeffHello")
        assert output == "Hello"

    def test_lowercase(self):
        normalizer = BertNormalizer(strip_accents=False, lowercase=True, handle_chinese_chars=False, clean_text=False)

        output = normalizer.normalize_str("Héllò")
        assert output == "héllò"

    def test_can_modify(self):
        normalizer = BertNormalizer(clean_text=True, handle_chinese_chars=True, strip_accents=True, lowercase=True)

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

    def test_set_item(self):
        normalizers = Sequence(
            [
                BertNormalizer(True, True),
                Prepend(prepend="test"),
            ]
        )
        assert normalizers[0].__class__ == BertNormalizer
        assert normalizers[1].__class__ == Prepend
        normalizers[1] = Strip()
        assert normalizers[1].__class__ == Strip
        with pytest.raises(IndexError):
            print(normalizers[2])

    def test_item_getters_and_setters(self):
        normalizers = Sequence(
            [
                BertNormalizer(clean_text=True, handle_chinese_chars=True, strip_accents=True, lowercase=True),
                Strip(left=True, right=True),
                Prepend(prepend="_"),
                Replace(pattern="something", content="else"),
            ]
        )

        assert normalizers[0].__class__ == BertNormalizer
        normalizers[0].clean_text = False
        normalizers[0].handle_chinese_chars = False
        normalizers[0].strip_accents = False
        normalizers[0].lowercase = False
        assert not normalizers[0].clean_text
        assert not normalizers[0].handle_chinese_chars
        assert not normalizers[0].strip_accents
        assert not normalizers[0].lowercase

        assert normalizers[1].__class__ == Strip
        normalizers[1].left = False
        normalizers[1].right = False
        assert not normalizers[1].left
        assert not normalizers[1].right

        assert normalizers[2].__class__ == Prepend
        normalizers[2].prepend = " "
        assert normalizers[2].prepend == " "

        assert normalizers[3].__class__ == Replace
        with pytest.raises(Exception):
            normalizers[3].pattern = "test"
        with pytest.raises(Exception):
            print(normalizers[3].pattern)
        normalizers[3].content = "test"
        assert normalizers[3].content == "test"


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


class TestPrepend:
    def test_instantiate(self):
        assert isinstance(Prepend("▁"), Normalizer)
        assert isinstance(Prepend("▁"), Prepend)
        assert isinstance(pickle.loads(pickle.dumps(Prepend("▁"))), Prepend)

    def test_prepend(self):
        normalizer = Prepend(prepend="▁")

        output = normalizer.normalize_str("hello")
        assert output == "▁hello"

    def test_can_modify(self):
        normalizer = Prepend("▁")

        assert normalizer.prepend == "▁"

        # Modify these
        normalizer.prepend = "-"
        assert normalizer.prepend == "-"


class TestUnicodeFilter:
    def test_instantiate(self):
        assert isinstance(UnicodeFilter(), Normalizer)
        assert isinstance(UnicodeFilter(), UnicodeFilter)
        assert isinstance(pickle.loads(pickle.dumps(UnicodeFilter())), UnicodeFilter)

    def test_default_filtering(self):
        normalizer = UnicodeFilter()  # Default filters out Unassigned, PrivateUse, Surrogate
        output = normalizer.normalize_str("Hello\uE000\U000F0000\U0010FFFF")  # Hello + Private Use + Private Use B + Unassigned
        assert output == "Hello"  # Only valid chars remain

    def test_custom_filtering(self):
        # Only filter private use areas
        normalizer = UnicodeFilter(
            filter_unassigned=False,
            filter_private_use=True,
            filter_surrogate=False
        )
        output = normalizer.normalize_str("Hello\uE000\U000F0000\U0010FFFF")  
        assert output == "Hello\U0010FFFF"  # Private use removed, others kept

    def test_can_modify(self):
        normalizer = UnicodeFilter()
        output = normalizer.normalize_str("Hello\uE000\U000F0000\U0010FFFF")  
        assert output == "Hello"  # All filtered by default

        # Disable all filtering
        normalizer = UnicodeFilter(
            filter_unassigned=False,
            filter_private_use=False,
            filter_surrogate=False
        )
        output = normalizer.normalize_str("Hello\uE000\U000F0000\U0010FFFF")
        assert output == "Hello\uE000\U000F0000\U0010FFFF"  # Nothing filtered


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
        with pytest.raises(Exception, match="TypeError:.*normalize()"):
            bad.normalize_str("Hey there!")
        assert good.normalize_str("Hey there!") == "Hey you!"
        with pytest.raises(Exception, match="Cannot use a NormalizedStringRefMut outside `normalize`"):
            good_custom.use_after_normalize()

    def test_normalizer_interface(self):
        normalizer = Normalizer.custom(TestCustomNormalizer.GoodCustomNormalizer())

        normalized = NormalizedString("Hey there!")
        normalizer.normalize(normalized)

        assert repr(normalized) == 'NormalizedString(original="Hey there!", normalized="Hey you!")'
        assert str(normalized) == "Hey you!"
