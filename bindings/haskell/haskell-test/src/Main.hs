module Main where

import Lib

main :: IO ()
main = do
  tokenizer <- mkTokenizer "roberta-base-vocab.json" "roberta-base-merges.txt"
  print tokenizer
  tokenize "Hey there!" tokenizer
