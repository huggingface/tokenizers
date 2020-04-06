import argparse

from tokenizers import Tokenizer, models, pre_tokenizers, decoders

parser = argparse.ArgumentParser()
parser.add_argument("--vocab", default=None, type=str, required=True, help="The vocab.json file")
parser.add_argument("--merges", default=None, type=str, required=True, help="The merges.txt file")
args = parser.parse_args()


class GoodCustom:
    """GoodCustom
    This class represents a good custom PreTokenizer that will be called
    by `tokenizers` when needed
    """

    def pre_tokenize(self, sentence):
        return sentence.split(" ")

    def decode(self, tokens):
        return ", ".join(tokens)


class BadCustom:
    """Bad Pretok
    This class represents a bad custom PreTokenizer that will trigger an exception
    when called by `tokenizers`
    """

    def pre_tokenize(self, sentence):
        return None

    def decode(self, tokens):
        return None


def tokenize(sentence):
    output = tokenizer.encode(sentence).tokens
    print(f"`{sentence}` tokenized to {output}")
    return output


# Create a Tokenizer using a BPE model
bpe = models.BPE(args.vocab, args.merges)
tokenizer = Tokenizer(bpe)

# Test the good custom classes
good_custom = GoodCustom()
good_pretok = pre_tokenizers.PreTokenizer.custom(good_custom)
good_decoder = decoders.Decoder.custom(good_custom)

tokenizer.pre_tokenizer = good_pretok
tokenizer.decoder = good_decoder

print("Tokenization will work with good custom:")
encoding = tokenizer.encode("Hey friend!")
print(f"IDS: {encoding.ids}")
print(f"TOKENS: {encoding.tokens}")
print(f"OFFSETS: {encoding.offsets}")
decoded = tokenizer.decode(encoding.ids)
print(f"DECODED: {decoded}")

# Now test with the bad custom classes
bad_custom = BadCustom()
bad_pretok = pre_tokenizers.PreTokenizer.custom(bad_custom)
bad_decoder = decoders.Decoder.custom(bad_custom)

tokenizer.pre_tokenizer = bad_pretok
tokenizer.decoder = bad_decoder
try:
    encoding = tokenizer.encode("Hey friend!")
except:
    print("Bad tokenizer didn't work")
