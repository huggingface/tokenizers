"""
Read and replace the `role_to_token` map, which says which token plays which role

    python examples/role_to_token.py

Runs on the GPT-2 fixture `make test` fetches into `data/`. Its `tokenizer.json` declares no roles.
"""

from pathlib import Path

from tokenizers import Tokenizer

path = Path(__file__).parent.parent / "data" / "gpt2.json"

tokenizer = Tokenizer.from_file(path)
print(tokenizer.role_to_token)
assert tokenizer.role_to_token == {}

# Replace the file's map at load time. `from_pretrained` takes the same keyword.
tokenizer = Tokenizer.from_file(path, role_to_token={"eos_token": "<|endoftext|>"})
print(tokenizer.role_to_token)
assert tokenizer.role_to_token == {"eos_token": "<|endoftext|>"}

eos = tokenizer.role_to_token["eos_token"]
assert tokenizer.encode(eos).ids == [50256]

# Get a new tokenizer with another map. The original keeps its own.
rebuilt = tokenizer.with_role_to_token({"eos_token": "<|endoftext|>", "pad_token": "<|endoftext|>"})
print(rebuilt.role_to_token)
assert rebuilt.role_to_token["pad_token"] == "<|endoftext|>"
assert "pad_token" not in tokenizer.role_to_token
