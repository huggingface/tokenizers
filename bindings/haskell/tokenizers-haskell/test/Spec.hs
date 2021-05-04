{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import qualified Data.ByteString.Lazy as LBS (toStrict)
import Data.Hashable (hash)
import qualified Network.HTTP.Client as HTTP
import qualified Network.HTTP.Client.TLS as HTTP
import qualified Test.Tasty as T
import qualified Test.Tasty.HUnit as H
import Tokenizers (Tokenizer, addSpecialToken, cleanTokens, createTokenizerFromJSONConfig, decode, encode, freeTokenizer, getIDs, getTokens, mkRobertaTokenizer)

data TestItem
  = Group String [TestItem]
  | EncodeBart String [Int]
  | DecodeBart [Int] String
  | MassDecodeBart [Int] [Int]
  | IncrementalDecodeBart [Int] [Int]
  | EncodeRoberta String [Int]
  | DecodeRoberta [Int] String
  | EncodeT5 String [Int]
  | DecodeT5 [Int] String
  | MassDecodeT5 [Int] [Int]
  | IncrementalDecodeT5 [Int] [Int]
  | IncrementalDecodeT5Fail [Int] [Int]
  deriving stock (Eq, Show)

data TestTokenizers = TestTokenizers
  { bartTokenizer :: Tokenizer,
    robertaTokenizer :: Tokenizer,
    t5Tokenizer :: Tokenizer
  }

testDrive :: IO ()
testDrive = do
  tokenizer <- mkRobertaTokenizer "models/roberta-base-vocab.json" "models/roberta-base-merges.txt"
  mapM_ (addSpecialToken tokenizer) ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]
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
  freeTokenizer tokenizer
  where
    test :: String -> Tokenizer -> IO ()
    test string tokenizer = do
      putStrLn $ "\n----\n" ++ string ++ ""
      encoding <- encode tokenizer string
      result <- getTokens encoding
      putStrLn "Haskell Token List:"
      print (cleanTokens <$> result)
      putStrLn "Haskell IDs:"
      result <- getIDs encoding
      print result

bartTests :: [TestItem]
bartTests =
  [ EncodeBart "<s>Hello world!</s>" [0, 31414, 232, 328, 2],
    EncodeBart "<s>Hello <mask>!</s><pad>" [0, 31414, 50264, 328, 2, 1],
    EncodeBart "<s>   Hello   <mask>  !    </s>  <pad>" [0, 1437, 1437, 20920, 1437, 1437, 50264, 1437, 27785, 1437, 1437, 1437, 1437, 2, 1437, 1437, 1],
    DecodeBart [0, 31414, 50264, 328, 2, 1] "<s>Hello<mask>!</s><pad>",
    DecodeBart [0, 1437, 1437, 20920, 1437, 1437, 50264, 1437, 27785, 1437, 1437, 1437, 1437, 2, 1437, 1437, 1] "<s>   Hello  <mask>  !    </s>  <pad>",
    MassDecodeBart [0, 21959, 1721, 44664, 2103, 4, 42351, 11974, 2103] ([0 .. 50107] <> [50109 .. 100000]),
    IncrementalDecodeBart [0] [31414, 50264, 328, 2, 1],
    IncrementalDecodeBart [0, 31414] [50264, 328, 2, 1],
    IncrementalDecodeBart [0, 31414, 50264, 328] [2, 1],
    IncrementalDecodeBart [0, 1437, 1437] [20920, 1437, 1437, 50264, 1437, 27785, 1437, 1437, 1437, 1437, 2, 1437, 1437, 1],
    IncrementalDecodeBart [0, 1437, 1437, 20920] [1437, 1437, 50264, 1437, 27785, 1437, 1437, 1437, 1437, 2, 1437, 1437, 1],
    IncrementalDecodeBart [0, 1437, 1437, 20920, 1437, 1437, 50264] [1437, 27785, 1437, 1437, 1437, 1437, 2, 1437, 1437, 1]
  ]

robertaTests :: [TestItem]
robertaTests =
  [ EncodeRoberta "<s>Hello world!</s>" [0, 31414, 232, 328, 2],
    EncodeRoberta "<s>Hello <mask>!</s><pad>" [0, 31414, 50264, 328, 2, 1],
    EncodeRoberta "<s>   Hello   <mask>  !    </s>  <pad>" [0, 1437, 1437, 20920, 1437, 1437, 50264, 1437, 27785, 1437, 1437, 1437, 1437, 2, 1437, 1437, 1],
    DecodeRoberta [0, 1437, 1437, 20920, 1437, 1437, 50264, 1437, 27785, 1437, 1437, 1437, 1437, 2, 1437, 1437, 1] "<s>   Hello  <mask>  !    </s>  <pad>"
  ]

t5Tests :: [TestItem]
t5Tests =
  [ EncodeT5 "<pad>Hello world!</s>" [0, 8774, 296, 55, 1],
    EncodeT5 "<pad>Hello <extra_id_0>!</s><pad>" [0, 8774, 32099, 3, 55, 1, 0],
    EncodeT5 "<pad>    Hello   <extra_id_0>  !   </s>   <pad>" [0, 8774, 32099, 3, 55, 1, 0],
    DecodeT5 [0, 8774, 32099, 3, 55, 1, 0] "<pad> Hello<extra_id_0> !</s><pad>",
    MassDecodeT5 [0, 8774, 32099, 3, 55, 1] [0 .. 100000],
    IncrementalDecodeT5Fail [0] [8774, 32099, 3, 55, 1, 0],
    IncrementalDecodeT5 [0, 8774, 32099, 3, 55] [1, 0],
    IncrementalDecodeT5Fail [0, 4219, 834, 7, 9963, 1820, 1738] [3476]
  ]

testData :: TestItem
testData =
  Group
    "tests"
    [ Group "Bart" bartTests,
      Group "Roberta" robertaTests,
      Group "T5" t5Tests
    ]

testTree :: T.TestTree
testTree =
  T.withResource
    createTokenizers
    freeTokenizers
    (toTest testData)
  where
    createTokenizer url expectedHash = do
      manager <- HTTP.newTlsManagerWith HTTP.tlsManagerSettings
      request <- HTTP.parseRequest url
      response <- HTTP.httpLbs request manager
      let body = LBS.toStrict . HTTP.responseBody $ response
      H.assertEqual "Unexpected json hash" expectedHash (hash body)
      createTokenizerFromJSONConfig body
    createTokenizers = do
      bartTokenizer <-
        createTokenizer
          "https://huggingface.co/facebook/bart-base/resolve/main/tokenizer.json"
          (-5675567303366998911)
      robertaTokenizer <-
        createTokenizer
          "https://huggingface.co/roberta-base/resolve/main/tokenizer.json"
          (-5675567303366998911)
      t5Tokenizer <-
        createTokenizer
          "https://huggingface.co/t5-base/resolve/main/tokenizer.json"
          (-6144928463468424742)
      pure $ TestTokenizers {..}
    freeTokenizers TestTokenizers {..} = do
      freeTokenizer bartTokenizer
      freeTokenizer robertaTokenizer
      freeTokenizer t5Tokenizer
    toTest :: TestItem -> IO TestTokenizers -> T.TestTree
    toTest (Group name tests) mtokenizers =
      T.testGroup name $ toTest <$> tests <*> pure mtokenizers
    toTest (EncodeBart s expected) mtokenizers = H.testCase ("Encode " <> show s) $ do
      TestTokenizers {..} <- mtokenizers
      enc <- encode bartTokenizer s
      ids <- getIDs enc
      H.assertEqual "Unexpected encoding result" expected ids
    toTest (DecodeBart ids expected) mtokenizers = H.testCase ("Decode " <> show ids) $ do
      TestTokenizers {..} <- mtokenizers
      s <- decode bartTokenizer ids
      H.assertEqual "Unexpected decoding result" expected s
    toTest (MassDecodeBart ids tokens) mtokenizers = H.testCase ("MassDecode " <> show ids) $ do
      TestTokenizers {..} <- mtokenizers
      mapM_ (\token -> decode bartTokenizer (ids <> [token])) tokens
    toTest (IncrementalDecodeBart ids otherIds) mtokenizers = H.testCase ("Incrementally decode " <> show ids <> " " <> show otherIds) $ do
      TestTokenizers {..} <- mtokenizers
      s <- decode bartTokenizer ids
      s' <- decode bartTokenizer otherIds
      s'' <- decode bartTokenizer $ ids <> otherIds
      H.assertEqual "Unexpected decoding result" s'' (s <> s')
    toTest (EncodeRoberta s expected) mtokenizers = H.testCase ("Encode " <> show s) $ do
      TestTokenizers {..} <- mtokenizers
      enc <- encode robertaTokenizer s
      ids <- getIDs enc
      H.assertEqual "Unexpected encoding result" expected ids
    toTest (DecodeRoberta ids expected) mtokenizers = H.testCase ("Decode " <> show ids) $ do
      TestTokenizers {..} <- mtokenizers
      s <- decode robertaTokenizer ids
      H.assertEqual "Unexpected decoding result" expected s
    toTest (EncodeT5 s expected) mtokenizers = H.testCase ("Encode " <> show s) $ do
      TestTokenizers {..} <- mtokenizers
      enc <- encode t5Tokenizer s
      ids <- getIDs enc
      H.assertEqual "Unexpected encoding result" expected ids
    toTest (DecodeT5 ids expected) mtokenizers = H.testCase ("Decode " <> show ids) $ do
      TestTokenizers {..} <- mtokenizers
      s <- decode t5Tokenizer ids
      H.assertEqual "Unexpected decoding result" expected s
    toTest (MassDecodeT5 ids tokens) mtokenizers = H.testCase ("MassDecode " <> show ids) $ do
      TestTokenizers {..} <- mtokenizers
      mapM_ (\token -> decode t5Tokenizer (ids <> [token])) tokens
    toTest (IncrementalDecodeT5 ids otherIds) mtokenizers = H.testCase ("Incrementally decode " <> show ids <> " " <> show otherIds) $ do
      TestTokenizers {..} <- mtokenizers
      s <- decode t5Tokenizer ids
      s' <- decode t5Tokenizer otherIds
      s'' <- decode t5Tokenizer $ ids <> otherIds
      H.assertEqual "Unexpected decoding result" s'' (s <> s')
    toTest (IncrementalDecodeT5Fail ids otherIds) mtokenizers = H.testCase ("Incrementally decode " <> show ids <> " " <> show otherIds) $ do
      TestTokenizers {..} <- mtokenizers
      s <- decode t5Tokenizer ids
      s' <- decode t5Tokenizer otherIds
      s'' <- decode t5Tokenizer $ ids <> otherIds
      H.assertBool "Unexpected decoding result" (s'' /= (s <> s'))

-- | Run 'stack ghci --test' to get a REPL for the tests.
main :: IO ()
main = T.defaultMain testTree
