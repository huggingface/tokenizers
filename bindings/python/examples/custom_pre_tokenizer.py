import argparse

from tokenizers import Tokenizer, models, pre_tokenizers

parser = argparse.ArgumentParser()
parser.add_argument("--vocab", default=None, type=str, required=True, help="The vocab.json file")
parser.add_argument("--merges", default=None, type=str, required=True, help="The merges.txt file")
args = parser.parse_args()

class MyPreTok:
    """
    This class represents a custom PreTokenizer that will be called
    by `tokenizers` when needed
    """
    def pre_tokenize(self, sentence):
        if sentence.startswith("Hello"):
            # This will generate an error
            return None

        # Prepend "Haha"
        return sum([ [ "Haha" ], sentence.split(" ") ], [])


# Create a PreTokenizer from our custom one
mypretok = MyPreTok()
pretok = pre_tokenizers.PreTokenizer.custom(mypretok)

# Create a Tokenizer using a BPE model
bpe = models.BPE.from_files(args.vocab, args.merges)
tokenizer = Tokenizer(bpe)

# And attach our PreTokenizer
tokenizer.with_pre_tokenizer(pretok)


def tokenize(sentence):
    output = [ token.value for token in tokenizer.encode(sentence) ]
    print(f"`{sentence}` tokenized to {output}")
    return output


## Good example
# Our PreTokenizer has been used as expected
assert(tokenize("Hey friend") == [ "H", "aha", "Hey", "friend" ])

## Bad example
# In this case, our PreTokenizer returns None instead of a List[str]
# So it doesn't work as expected, and we get a empty list back, with an error printed
assert(tokenize("Hello friend") == [])
