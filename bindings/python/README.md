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

python example.py --file <FILE_PATH>
# or
python example.py
```
