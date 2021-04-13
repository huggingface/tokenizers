module Main where

import Tokenizers

test string tokenizer = do
  putStrLn $ "\n----\n" ++ string ++ ""
  encoding <- encode tokenizer string
  result <- getTokens encoding
  putStrLn "Haskell Token List:"
  print (cleanTokens <$> result)
  putStrLn "Haskell IDs:"
  result <- getIDs encoding -- not working yet
  print result

main :: IO ()
main = do
  tokenizer <- mkRobertaTokenizer "roberta-base-vocab.json" "roberta-base-merges.txt"
  print tokenizer
  test "Hey there!" tokenizer
  test "The quick brown fox jumped over the lazy dogs." tokenizer
  test "The walls in the mall are totally totally tall" tokenizer
  test "It was the best of times it was the blurst of times." tokenizer
  test "It is precisely the common features of all experience, such as characterise everything we encounter, which are the primary and most profound occasion for astonishment; indeed, one might almost say that it is the fact that anything is experienced and encountered at all." tokenizer
  test "What does 'organic' mean?-that is, in the wider sense here supposed, naturally excluding such simple answers as 'protei' or 'protoplasm'. Fixing our attention on a somewhat wider concept than this, we arrive at the criterion of metabolism. Thus Schopenhauer's line of demarcation may be regarded as highly suitable, when he says that in inorganic being 'the essential and permanent element, the basis of identity and integrity, is the material, the matter, the inessential and mutable element being the form. In organic being the reverse is true; for its life, that is, its existence as an organic being, consists precisely in a constant change of matter while the form persists.'" tokenizer
  test "hey hey there hey hey hey there hey" tokenizer
  test "hi hi hi hello hello" tokenizer
  test "hi" tokenizer
  test "hi hi hi hi hi hi hi hi" tokenizer
  test "hi there hi there hi there hi there hi there hi there hi there hi there" tokenizer
  test "hello world. Let's try tokenizing this. hi hi hi and hello hello" tokenizer
