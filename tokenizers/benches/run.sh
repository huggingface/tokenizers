#!/bin/bash

set -e

echo 'Updating benchmark fixtures...'
[[ -f ./benches/gpt2-vocab.json ]] || wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json -O ./benches/gpt2-vocab.json
[[ -f ./benches/gpt2-merges.txt ]] || wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt -O ./benches/gpt2-merges.txt
[[ -f ./benches/big.txt ]] || wget https://norvig.com/big.txt -O ./benches/big.txt

echo 'Running bencmarks...'
cargo bench -- --verbose
