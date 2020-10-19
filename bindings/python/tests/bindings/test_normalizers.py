import pickle
import pytest

from tokenizers import Tokenizer, NormalizedString
from tokenizers.models import BPE
from tokenizers.normalizers import Normalizer, BertNormalizer, Sequence, Lowercase, Strip, NORM_OPTIONS, opencc_enabled


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

    def test_handle_separate_numbers(self):
        normalizer = BertNormalizer(
            strip_accents=False, lowercase=False, handle_chinese_chars=True, clean_text=False, norm_options=NORM_OPTIONS.SEPARATE_INTEGERS
        )

        output = normalizer.normalize_str("你好123 is 123")
        assert output == " 你  好  1  2  3  is  1  2  3 "

    def test_special_chars(self):
        import subprocess, sys
        cmd = sys.executable + r''' << END
from tokenizers.normalizers import BertNormalizer, NORM_OPTIONS
normalizer = BertNormalizer(
    strip_accents=False, lowercase=False, handle_chinese_chars=True, clean_text=False, norm_options=NORM_OPTIONS.SEPARATE_SYMBOLS
)
output = normalizer.normalize_str("\$100 and 0.5% \$\$ %%" )
print(output)
END'''
        output = subprocess.check_output(cmd, shell=True).decode().rstrip('\n')
        assert output == " $ 100 and 0 . 5 %   $  $   %  % ", output


    def test_zh_norm(self):
        import subprocess, sys
        if not opencc_enabled():
            return
        cmd = sys.executable + ''' << END
from tokenizers.normalizers import BertNormalizer, NORM_OPTIONS
normalizer = BertNormalizer(
    strip_accents=False, lowercase=False, handle_chinese_chars=True, clean_text=False, norm_options=NORM_OPTIONS.ZH_NORM_MAPPING | NORM_OPTIONS.SIMPL_TO_TRAD
)
output = normalizer.normalize_str("系列 聯系 « 联系 𠱁 氹 𥱊 栄 梊 𠹌 <n> "+chr(0) )
print(output)
END'''
        output = subprocess.check_output(cmd, shell=True).decode().rstrip('\n')

        assert output == " 系  列   聯  系  <<  聯  繫   o 氹   氹   席   榮   折  木   o 能  <n>  ", repr(output)

        cmd = sys.executable + ''' << END
from tokenizers.normalizers import BertNormalizer, NORM_OPTIONS
normalizer = BertNormalizer(
    strip_accents=False, lowercase=False, handle_chinese_chars=True, clean_text=False, norm_options=NORM_OPTIONS.ZH_NORM_MAPPING | NORM_OPTIONS.SIMPL_TO_TRAD
)
output = normalizer.normalize_str("头部" )
print(output)
END'''
        output = subprocess.check_output(cmd, shell=True).decode().rstrip('\n')
        assert output == " 頭  部 ", output


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
