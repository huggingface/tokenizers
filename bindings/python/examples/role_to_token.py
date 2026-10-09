"""
Read and replace the `role_to_token` map, then pad with the token that plays the `pad_token` role

    python examples/role_to_token.py

Runs on the GPT-2 fixture `make test` fetches into `data/`. Its `tokenizer.json` declares no roles.
"""

from pathlib import Path

from tokenizers import Padding, Tokenizer

path = Path(__file__).parent.parent / "data" / "gpt2.json"

tokenizer = Tokenizer.from_file(path, role_to_token={"eos_token": "<|endoftext|>"})
print(tokenizer.role_to_token)

# GPT-2 has no pad token: reuse eos. The original tokenizer keeps its own map.
padded = tokenizer.with_role_to_token({**tokenizer.role_to_token, "pad_token": "<|endoftext|>"})
print(padded.role_to_token)
assert "pad_token" not in tokenizer.role_to_token

pad_token = padded.role_to_token["pad_token"]
pad_id = padded.token_to_id(pad_token)
assert pad_id == 50256

padded.padding = Padding(pad_id=pad_id, pad_token=pad_token)
short, long = padded.encode_batch(["Hello", "Hello, world"])
print(short.ids)
print(padded.decode(short, False))
assert short.ids == [15496, 50256, 50256]
assert long.ids == [15496, 11, 995]
