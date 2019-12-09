import time
import argparse
from tqdm import tqdm

import logging
logging.getLogger('transformers').disabled = True
logging.getLogger('transformers.tokenization_utils').disabled = True

from tokenizers import Tokenizer, models, pre_tokenizers, decoders
from transformers import GPT2Tokenizer, BertTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--type", default="gpt2", type=str, help="The type of tokenizer (bert|gpt2)")
parser.add_argument("--file", default=None, type=str, help="The file to encode")
parser.add_argument("--vocab", default=None, type=str, required=True, help="The vocab file")
parser.add_argument("--merges", default=None, type=str, help="The merges.txt file")
args = parser.parse_args()

if args.type == "gpt2" and args.merges is None:
    raise Exception("Expected merges.txt file")

if args.file is not None:
    with open(args.file, "r") as fp:
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

if args.type == "gpt2":
    tok_p = GPT2Tokenizer.from_pretrained('gpt2')

    # Create a Tokenizer using BPE
    tok_r = Tokenizer(models.BPE.from_files(args.vocab, args.merges))
    # Use ByteLevel PreTokenizer
    tok_r.with_pre_tokenizer(pre_tokenizers.ByteLevel.new())
    # Use ByteLevel Decoder
    tok_r.with_decoder(decoders.ByteLevel.new())
elif args.type == "bert":
    tok_p = BertTokenizer.from_pretrained('bert-base-uncased')

    tok_r = Tokenizer(models.WordPiece.from_files(args.vocab))
    tok_r.with_pre_tokenizer(pre_tokenizers.BasicPreTokenizer.new())
    tok_r.with_decoder(decoders.WordPiece.new())
else:
    raise Exception(f"Unknown type {args.type}")

def tokenize_r():
    # return [ tok_r.encode(sentence) for sentence in text]
    return tok_r.encode_batch(text);

def tokenize_p():
    return [tok_p.encode(sentence) for sentence in tqdm(text)]

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
diff = 0
for i in range(0, len(ids_r)):
    if ids_r[i] != encoded_p[i]:
        diff += 1
        print("".join([ token.value for token in encoded_r[i] ]))
        print("".join(tok_p.tokenize(text[i])))
        print(text[i])
        print("")
        #print(ids_r[i])
        #print(encoded_p[i])
print(f"DIFF: {diff}")
assert(ids_r == encoded_p)

exit()
decoded_r = tok_r.decode_batch(ids_r)
decoded_p = [ tok_p.decode(en) for en in encoded_p ]
for i in range(0, len(text)):
    if decoded_r[i] != decoded_p[i]: #text[i]:
        print(decoded_r[i])
        print(decoded_p[i])
        #print(text[i])
        print("")

assert(decoded_r == text)
