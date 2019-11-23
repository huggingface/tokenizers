import os
import time
import argparse

from tokenizers import Tokenizer, models, pre_tokenizers, decoders
from transformers import GPT2Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--file", default=None, type=str, help="The file to encode")
parser.add_argument("--vocab", default=None, type=str, required=True, help="The vocab.json file")
parser.add_argument("--merges", default=None, type=str, required=True, help="The merges.txt file")
args = parser.parse_args()

if args.file is not None:
    current_dir = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(current_dir, args.file)

    with open(path, "r") as fp:
        text = [ line.strip() for line in fp ]
else:
    text = """
The Zen of Python, by Tim Peters
Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!
""".split("\n")


tok_p = GPT2Tokenizer.from_pretrained('gpt2')

# Create a Tokenizer using BPE
tok_r = Tokenizer(models.BPE.from_files(args.vocab, args.merges))
# Use ByteLevel PreTokenizer
tok_r.with_pre_tokenizer(pre_tokenizers.ByteLevel.new())
# Use ByteLevel Decoder
tok_r.with_decoder(decoders.ByteLevel.new())

def tokenize_r():
    # return [ tok_r.encode(sentence) for sentence in text]
    return tok_r.encode_batch(text);

def tokenize_p():
    return [tok_p.encode(sentence) for sentence in text]

print(f"Tokenizing {len(text)} lines")

# Rust version
start = time.time()
encoded_r = tokenize_r()
end = time.time()
time_r = end - start
print(f"Rust tokenizer took: {time_r} sec")

# Python version
start = time.time()
encoded_p = tokenize_p()
end = time.time()
time_p = end - start
print(f"Transformer tokenizer took: {time_p} sec")

print(f"SpeedUp Ratio: {time_p / time_r}")

ids_r = [ [ token.id for token in sentence ] for sentence in encoded_r ]
assert(ids_r == encoded_p)

decoded_r = tok_r.decode_batch(ids_r)
for i in range(0, len(text)):
    if decoded_r[i] != text[i]:
        print(decoded_r[i])
        print(text[i])
        print("")

assert(decoded_r == text)
