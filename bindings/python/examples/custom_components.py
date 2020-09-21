import jieba

from tokenizers import Tokenizer, Regex
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.normalizers import Normalizer
from tokenizers.decoders import Decoder


class JiebaPreTokenizer:
    def jieba_split(self, i, normalized):
        return [normalized[w[1] : w[2]] for w in jieba.tokenize(str(normalized))]

    def pre_tokenize(self, pretok):
        # Let's call split on the PreTokenizedString to split using `self.split`
        # Here we can call `pretok.split` multiple times if we want to apply
        # different algorithm
        pretok.split(self.jieba_split)


class CustomDecoder:
    def decode(self, tokens):
        return "".join(tokens)


class CustomNormalizer:
    def normalize(self, normalized):
        normalized.nfkc()
        normalized.replace(Regex("\s+"), " ")
        normalized.lowercase()


# This section shows how to attach these custom components to the Tokenizer
tok = Tokenizer(BPE())
tok.normalizer = Normalizer.custom(CustomNormalizer())
tok.pre_tokenizer = PreTokenizer.custom(JiebaPreTokenizer())
tok.decoder = Decoder.custom(CustomDecoder())

input1 = "永和服装饰品有限公司"
print("PreTokenize:", input1)
print(tok.pre_tokenizer.pre_tokenize_str(input1))
# [('永和', (0, 2)), ('服装', (2, 4)), ('饰品', (4, 6)), ('有限公司', (6, 10))]

input2 = "ℌ𝔢𝔩𝔩𝔬    𝔱𝔥𝔢𝔯𝔢 𝓂𝓎 𝒹ℯ𝒶𝓇 𝕕𝕖𝕒𝕣    𝕗𝕣𝕚𝕖𝕟𝕕!"
print("Normalize:", input2)
print(tok.normalizer.normalize_str(input2))
# hello there my dear dear friend!
