"""
Pad a batch of encodings, every way padding can be set

    python examples/batch_padding.py

Runs on the GPT-2 fixture `make test` fetches into `data/`. The `padding` attribute applies to
every encode; the `padding=` keyword of `encode` and `encode_batch` applies to that one call.
`None` switches padding off in both places. The expected ids come from released tokenizers 0.23.1.
"""

from pathlib import Path

from tokenizers import Padding, Tokenizer

path = Path(__file__).parent.parent / "data" / "gpt2.json"
prompts = ["Hello, I'm a", "The weather today is quite a bit warmer than expected"]

# GPT-2 has no padding token, so its end-of-text token pads.
EOT = 50256
EOT_TOKEN = "<|endoftext|>"
SHORT = [15496, 11, 314, 1101, 257]
LONG = [464, 6193, 1909, 318, 2407, 257, 1643, 23254, 621, 2938]


def show(title, encodings):
    print(f"== {title}")
    for encoding in encodings:
        print(f"  {len(encoding):>2} ids  {encoding.ids}")
        print(f"          mask {encoding.attention_mask}")


tokenizer = Tokenizer.from_file(path)
assert tokenizer.padding is None

short, long = tokenizer.encode_batch(prompts)
show("no padding", (short, long))
assert short.ids == SHORT
assert long.ids == LONG

tokenizer.padding = Padding(direction="left", pad_id=EOT, pad_token=EOT_TOKEN)
short, long = tokenizer.encode_batch(prompts)
show("attribute: left, to the longest in the batch", (short, long))
assert short.ids == [EOT] * 5 + SHORT
assert short.attention_mask == [0] * 5 + [1] * 5
assert long.ids == LONG
assert long.attention_mask == [1] * 10

short, long = tokenizer.encode_batch(prompts, padding=Padding(length=16, pad_id=EOT, pad_token=EOT_TOKEN))
show("keyword, this call only: right, to a fixed length of 16", (short, long))
assert short.ids == SHORT + [EOT] * 11
assert short.attention_mask == [1] * 5 + [0] * 11
assert long.ids == LONG + [EOT] * 6
assert long.attention_mask == [1] * 10 + [0] * 6
assert tokenizer.padding == Padding(direction="left", pad_id=EOT, pad_token=EOT_TOKEN)

short, long = tokenizer.encode_batch(prompts, padding=None)
show("keyword None, this call only: switched off", (short, long))
assert short.ids == SHORT
assert long.ids == LONG
assert tokenizer.encode_batch(prompts)[0].ids == [EOT] * 5 + SHORT

tokenizer.padding = Padding(pad_to_multiple_of=8, pad_id=EOT, pad_token=EOT_TOKEN)
short, long = tokenizer.encode_batch(prompts)
show("attribute: right, to a multiple of 8", (short, long))
assert short.ids == SHORT + [EOT] * 11
assert long.ids == LONG + [EOT] * 6

encoding = tokenizer.encode(prompts[0], padding=Padding(length=8, pad_id=EOT, pad_token=EOT_TOKEN))
show("keyword on a single encode: right, to a fixed length of 8", (encoding,))
assert encoding.ids == SHORT + [EOT] * 3
assert encoding.attention_mask == [1] * 5 + [0] * 3

tokenizer.padding = None
short, long = tokenizer.encode_batch(prompts)
show("attribute None: switched off", (short, long))
assert short.ids == SHORT
assert long.ids == LONG
