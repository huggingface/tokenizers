### Python Bindings

```
# This expect the rust chain to be nightly
rustup update
rustup default nightly

# In this folder:
python3 -m venv .env
source .env/bin/activate
pip install maturin
maturin develop --release

# Then test:
pip install transformers

# Download vocab/merges from GPT-2
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt

python examples/example.py --file <FILE_PATH> --merges gpt2-merges.txt --vocab gpt2-vocab.json
python custom_pre_tokenizer.py --merges gpt2-merges.txt --vocab gpt2-vocab.json
```
