# See 
# https://github.com/huggingface/transformers/issues/1083
# https://huggingface.co/transformers/v1.1.0/_modules/pytorch_transformers/tokenization_roberta.html

wget https://huggingface.co/roberta-base/resolve/main/vocab.json -O roberta-base-vocab.json
wget https://huggingface.co/roberta-base/resolve/main/merges.txt -O roberta-base-merges.txt
wget https://huggingface.co/roberta-base/resolve/main/tokenizer.json -O roberta-base-tokenizer.json
