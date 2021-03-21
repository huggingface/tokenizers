
module Main where

import Data.Map

vocabMap = fromList [
        ("bert-base-uncased", "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt"),
        ("bert-large-uncased", "https://huggingface.co/bert-large-uncased/resolve/main/vocab.txt"),
        ("bert-base-cased", "https://huggingface.co/bert-base-cased/resolve/main/vocab.txt"),
        ("bert-large-cased", "https://huggingface.co/bert-large-cased/resolve/main/vocab.txt"),
        ("bert-base-multilingual-uncased", "https://huggingface.co/bert-base-multilingual-uncased/resolve/main/vocab.txt"),
        ("bert-base-multilingual-cased", "https://huggingface.co/bert-base-multilingual-cased/resolve/main/vocab.txt"),
        ("bert-base-chinese", "https://huggingface.co/bert-base-chinese/resolve/main/vocab.txt"),
        ("bert-base-german-cased", "https://int-deepset-models-bert.s3.eu-central-1.amazonaws.com/pytorch/bert-base-german-cased-vocab.txt"),
        ("bert-large-uncased-whole-word-masking", "https://huggingface.co/bert-large-uncased-whole-word-masking/resolve/main/vocab.txt"),
        ("bert-large-cased-whole-word-masking", "https://huggingface.co/bert-large-cased-whole-word-masking/resolve/main/vocab.txt"),
        ("bert-large-uncased-whole-word-masking-finetuned-squad", "https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/vocab.txt"),
        ("bert-large-cased-whole-word-masking-finetuned-squad", "https://huggingface.co/bert-large-cased-whole-word-masking-finetuned-squad/resolve/main/vocab.txt"),
        ("bert-base-cased-finetuned-mrpc", "https://huggingface.co/bert-base-cased-finetuned-mrpc/resolve/main/vocab.txt"),
        ("bert-base-german-dbmdz-cased", "https://huggingface.co/bert-base-german-dbmdz-cased/resolve/main/vocab.txt"),
        ("bert-base-german-dbmdz-uncased", "https://huggingface.co/bert-base-german-dbmdz-uncased/resolve/main/vocab.txt"),
        ("TurkuNLP/bert-base-finnish-cased-v1", "https://huggingface.co/TurkuNLP/bert-base-finnish-cased-v1/resolve/main/vocab.txt"),
        ("TurkuNLP/bert-base-finnish-uncased-v1", "https://huggingface.co/TurkuNLP/bert-base-finnish-uncased-v1/resolve/main/vocab.txt"),
        ("wietsedv/bert-base-dutch-cased", "https://huggingface.co/wietsedv/bert-base-dutch-cased/resolve/main/vocab.txt"),
        ("t5-small", "https://huggingface.co/t5-small/resolve/main/spiece.model"),
        ("t5-base", "https://huggingface.co/t5-base/resolve/main/spiece.model"),
        ("t5-large", "https://huggingface.co/t5-large/resolve/main/spiece.model"),
        ("t5-3b", "https://huggingface.co/t5-3b/resolve/main/spiece.model"),
        ("t5-11b", "https://huggingface.co/t5-11b/resolve/main/spiece.model"),
        ("roberta-base", "https://huggingface.co/roberta-base/resolve/main/vocab.json"),
        ("roberta-large", "https://huggingface.co/roberta-large/resolve/main/vocab.json"),
        ("roberta-large-mnli", "https://huggingface.co/roberta-large-mnli/resolve/main/vocab.json"),
        ("distilroberta-base", "https://huggingface.co/distilroberta-base/resolve/main/vocab.json"),
        ("roberta-base-openai-detector", "https://huggingface.co/roberta-base/resolve/main/vocab.json"),
        ("roberta-large-openai-detector", "https://huggingface.co/roberta-large/resolve/main/vocab.json")
        ]

mergesMap = fromList [
        ("roberta-base", "https://huggingface.co/roberta-base/resolve/main/merges.txt"),
        ("roberta-large", "https://huggingface.co/roberta-large/resolve/main/merges.txt"),
        ("roberta-large-mnli", "https://huggingface.co/roberta-large-mnli/resolve/main/merges.txt"),
        ("distilroberta-base", "https://huggingface.co/distilroberta-base/resolve/main/merges.txt"),
        ("roberta-base-openai-detector", "https://huggingface.co/roberta-base/resolve/main/merges.txt"),
        ("roberta-large-openai-detector", "https://huggingface.co/roberta-large/resolve/main/merges.txt")
        ]
    
tokenizerMap = fromList [
        ("bert-base-uncased", "https://huggingface.co/bert-base-uncased/resolve/main/tokenizer.json"),
        ("bert-large-uncased", "https://huggingface.co/bert-large-uncased/resolve/main/tokenizer.json"),
        ("bert-base-cased", "https://huggingface.co/bert-base-cased/resolve/main/tokenizer.json"),
        ("bert-large-cased", "https://huggingface.co/bert-large-cased/resolve/main/tokenizer.json"),
        ("bert-base-multilingual-uncased", "https://huggingface.co/bert-base-multilingual-uncased/resolve/main/tokenizer.json"),
        ("bert-base-multilingual-cased", "https://huggingface.co/bert-base-multilingual-cased/resolve/main/tokenizer.json"),
        ("bert-base-chinese", "https://huggingface.co/bert-base-chinese/resolve/main/tokenizer.json"),
        ("bert-base-german-cased", "https://int-deepset-models-bert.s3.eu-central-1.amazonaws.com/pytorch/bert-base-german-cased-tokenizer.json"),
        ("bert-large-uncased-whole-word-masking", "https://huggingface.co/bert-large-uncased-whole-word-masking/resolve/main/tokenizer.json"),
        ("bert-large-cased-whole-word-masking", "https://huggingface.co/bert-large-cased-whole-word-masking/resolve/main/tokenizer.json"),
        ("bert-large-uncased-whole-word-masking-finetuned-squad", "https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/tokenizer.json"),
        ("bert-large-cased-whole-word-masking-finetuned-squad", "https://huggingface.co/bert-large-cased-whole-word-masking-finetuned-squad/resolve/main/tokenizer.json"),
        ("bert-base-cased-finetuned-mrpc", "https://huggingface.co/bert-base-cased-finetuned-mrpc/resolve/main/tokenizer.json"),
        ("bert-base-german-dbmdz-cased", "https://huggingface.co/bert-base-german-dbmdz-cased/resolve/main/tokenizer.json"),
        ("bert-base-german-dbmdz-uncased", "https://huggingface.co/bert-base-german-dbmdz-uncased/resolve/main/tokenizer.json"),
        ("TurkuNLP/bert-base-finnish-cased-v1", "https://huggingface.co/TurkuNLP/bert-base-finnish-cased-v1/resolve/main/tokenizer.json"),
        ("TurkuNLP/bert-base-finnish-uncased-v1", "https://huggingface.co/TurkuNLP/bert-base-finnish-uncased-v1/resolve/main/tokenizer.json"),
        ("wietsedv/bert-base-dutch-cased", "https://huggingface.co/wietsedv/bert-base-dutch-cased/resolve/main/tokenizer.json"),
        ("t5-small", "https://huggingface.co/t5-small/resolve/main/tokenizer.json"),
        ("t5-base", "https://huggingface.co/t5-base/resolve/main/tokenizer.json"),
        ("t5-large", "https://huggingface.co/t5-large/resolve/main/tokenizer.json"),
        ("t5-3b", "https://huggingface.co/t5-3b/resolve/main/tokenizer.json"),
        ("t5-11b", "https://huggingface.co/t5-11b/resolve/main/tokenizer.json"),
        ("roberta-base", "https://huggingface.co/roberta-base/resolve/main/tokenizer.json"),
        ("roberta-large", "https://huggingface.co/roberta-large/resolve/main/tokenizer.json"),
        ("roberta-large-mnli", "https://huggingface.co/roberta-large-mnli/resolve/main/tokenizer.json"),
        ("distilroberta-base", "https://huggingface.co/distilroberta-base/resolve/main/tokenizer.json"),
        ("roberta-base-openai-detector", "https://huggingface.co/roberta-base/resolve/main/tokenizer.json"),
        ("roberta-large-openai-detector", "https://huggingface.co/roberta-large/resolve/main/tokenizer.json")
        ]

main = do
  putStrLn "Done"
