"""
Load a tokenizer from a model repo on the Hugging Face Hub

    python examples/from_pretrained.py [repo_id] [revision]

Defaults to openai-community/gpt2 at main. `huggingface_hub` downloads `tokenizer.json` into
its cache (`~/.cache/huggingface/hub` unless `HF_HOME` says otherwise), next to what
transformers downloads, and reads it from there on the next run.
"""

import sys

from tokenizers import Tokenizer

repo_id = sys.argv[1] if len(sys.argv) > 1 else "openai-community/gpt2"
revision = sys.argv[2] if len(sys.argv) > 2 else "main"

tokenizer = Tokenizer.from_pretrained(repo_id, revision=revision)

encoding = tokenizer.encode("Hello there, how are you?")
print(f"{repo_id}@{revision}")
print("ids:    ", encoding.ids)
print("decoded:", tokenizer.decode(encoding.ids))
