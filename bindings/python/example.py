import os
import time
import argparse

from tokenizers import Tokenizer
from transformers import GPT2Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--file", default=None, type=str, help="The file to encode")
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
tok_r = Tokenizer.bpe_from_files(
    "../../data/gpt2-vocab.json",
    "../../data/gpt2-merges.txt",
    pre_tokenizer="ByteLevel",
)

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
print(f"Rust tokenizer took: {end - start} sec")

# Python version
start = time.time()
encoded_p = tokenize_p()
end = time.time()
print(f"Transformer tokenizer took: {end - start} sec")

assert([ [ token.id for token in sentence] for sentence in encoded_r ] == encoded_p)
