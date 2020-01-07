import time
import argparse
from tqdm import tqdm

import logging
logging.getLogger('transformers').disabled = True
logging.getLogger('transformers.tokenization_utils').disabled = True

from tokenizers import Tokenizer, pre_tokenizers, decoders
from tokenizers.models import BPE, WordPiece
from tokenizers.processors import BertProcessing
from tokenizers.normalizers import BertNormalizer

from transformers import GPT2Tokenizer, BertTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--type", default="gpt2", type=str, help="The type of tokenizer (bert|gpt2)")
parser.add_argument("--file", default=None, type=str, help="The file to encode")
parser.add_argument("--vocab", default=None, type=str, required=True, help="The vocab file")
parser.add_argument("--merges", default=None, type=str, help="The merges.txt file")
parser.add_argument("--debug", default=False, type=bool, help="Verbose output")
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
    print("Running GPT-2 tokenizer")
    tok_p = GPT2Tokenizer.from_pretrained('gpt2')

    # Create a Tokenizer using BPE
    tok_r = Tokenizer(BPE.from_files(args.vocab, args.merges))
    # Use ByteLevel PreTokenizer
    tok_r.with_pre_tokenizer(pre_tokenizers.ByteLevel.new(add_prefix_space=False))
    # Use ByteLevel Decoder
    tok_r.with_decoder(decoders.ByteLevel.new())
elif args.type == "bert":
    print("Running Bert tokenizer")
    tok_p = BertTokenizer.from_pretrained('bert-base-uncased')

    tok_r = Tokenizer(WordPiece.from_files(
        args.vocab,
        unk_token="[UNK]",
        max_input_chars_per_word=100)
    )
    tok_r.with_normalizer(BertNormalizer.new(
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=True,
        lowercase=True,
    ))
    # tok_r.with_pre_tokenizer(pre_tokenizers.Whitespace.new())
    tok_r.with_pre_tokenizer(pre_tokenizers.BertPreTokenizer.new())
    tok_r.with_decoder(decoders.WordPiece.new())
    tok_r.with_post_processor(BertProcessing.new(
        ("[SEP]", tok_r.token_to_id("[SEP]")),
        ("[CLS]", tok_r.token_to_id("[CLS]")),
    ))
else:
    raise Exception(f"Unknown type {args.type}")


def tokenize_r():
    return tok_r.encode_batch(text)


def tokenize_p():
    return [tok_p.encode(sentence, add_special_tokens=True) for sentence in tqdm(text)]


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

ids_r = [sentence.ids for sentence in encoded_r]
diff_ids = 0
for i in range(0, len(encoded_r)):
    if encoded_r[i].ids != encoded_p[i]:
        diff_ids += 1
        if args.debug:
            print(encoded_r[i].ids)
            print(encoded_p[i])
            print(encoded_r[i].tokens)
            print(tok_p.tokenize(text[i]))
            print(text[i])
            print("")
print(f"Ids differences: {diff_ids}")

decoded_r = tok_r.decode_batch([sentence.ids for sentence in encoded_r], skip_special_tokens=False)
decoded_p = [tok_p.decode(en) for en in encoded_p]
diff_decoded = 0
for i in range(0, len(text)):
    if decoded_r[i] != decoded_p[i]:
        diff_decoded += 1
        if args.debug:
            print(f"Original:  {text[i]}")
            print(f"Rust:      {decoded_r[i]}")
            print(f"Python:    {decoded_p[i]}")
            print("")
print(f"Decoding differences: {diff_decoded}")
