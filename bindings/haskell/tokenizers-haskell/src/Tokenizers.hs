{-# LANGUAGE ForeignFunctionInterface #-}

module Tokenizers where

import Foreign.C.String (CString, newCString, peekCString)
import Foreign.C.Types (CInt, CUInt)
import Foreign.Ptr (Ptr, castPtr)
import Foreign.Storable (Storable (peek, peekByteOff))

data CTokenizer

data CEncoding

data CTokens

data CIDs

data Tokenizer = Tokenizer
  { tok :: Ptr CTokenizer,
    vocab :: FilePath,
    merges :: Maybe FilePath
  }

newtype Encoding = Encoding
  { enc :: Ptr CEncoding
  }

instance Show Tokenizer where
  show (Tokenizer _ vocab merges) =
    "Huggingface Tokenizer Object\n"
      <> "  vocab: "
      <> vocab
      <> "\n"
      <> maybe mempty ("  merges: " <>) merges

foreign import ccall unsafe "mk_wordpiece_tokenizer"
  r_mk_wordpiece_tokenizer ::
    CString -> IO (Ptr CTokenizer)

mkWordPieceTokenizer :: FilePath -> IO Tokenizer
mkWordPieceTokenizer vocab = do
  cvocab <- newCString $ vocab ++ "\0"
  result <- r_mk_wordpiece_tokenizer cvocab
  pure (Tokenizer result vocab Nothing)

foreign import ccall unsafe "mk_bpe_tokenizer"
  r_mk_bpe_tokenizer ::
    CString -> CString -> IO (Ptr CTokenizer)

mkBPETokenizer :: FilePath -> FilePath -> IO Tokenizer
mkBPETokenizer vocab merges = do
  cvocab <- newCString $ vocab ++ "\0"
  cmerges <- newCString $ merges ++ "\0"
  result <- r_mk_bpe_tokenizer cvocab cmerges
  pure (Tokenizer result vocab (Just merges))

foreign import ccall unsafe "mk_roberta_tokenizer"
  r_mk_roberta_tokenizer ::
    CString -> CString -> IO (Ptr CTokenizer)

mkRobertaTokenizer :: FilePath -> FilePath -> IO Tokenizer
mkRobertaTokenizer vocab merges = do
  cvocab <- newCString $ vocab ++ "\0"
  cmerges <- newCString $ merges ++ "\0"
  result <- r_mk_roberta_tokenizer cvocab cmerges
  pure (Tokenizer result vocab (Just merges))

foreign import ccall unsafe "encode"
  r_encode ::
    CString -> Ptr CTokenizer -> IO (Ptr CEncoding)

encode :: Tokenizer -> String -> IO Encoding
encode (Tokenizer tokenizer _ _) text = do
  str <- newCString text
  encoding <- r_encode str tokenizer
  pure (Encoding encoding)

foreign import ccall unsafe "add_special_token"
  r_add_special_token ::
    Ptr CTokenizer -> CString -> IO ()

add_special_token :: Tokenizer -> String -> IO ()
add_special_token (Tokenizer tokenizer _ _) token = do
  str <- newCString text
  r_add_special_token tokenizer str

foreign import ccall unsafe "get_tokens"
  r_get_tokens ::
    Ptr CEncoding -> IO (Ptr CTokens)

getTokens :: Encoding -> IO [String]
getTokens (Encoding encoding) = do
  ptr <- r_get_tokens encoding
  -- 1st value of struct is the # of tokens
  sz <- fromIntegral <$> (peek (castPtr ptr) :: IO CInt) :: IO Int
  -- 2nd value of struct is the array of tokens
  tokens <- peekByteOff ptr 8 :: IO (Ptr CString)
  mapM
    (\idx -> peekByteOff tokens (step * idx) >>= peekCString)
    [0 .. sz -1]
  where
    step = 8

cleanTokens :: String -> String
cleanTokens xs = [x | x <- xs, x `notElem` "\288"]

foreign import ccall unsafe "get_ids" r_get_ids :: Ptr CEncoding -> IO (Ptr CIDs)

getIDs :: Encoding -> IO [Int]
getIDs (Encoding encoding) = do
  ptr <- r_get_ids encoding
  -- 1st value of struct is the # of tokens
  sz <- fromIntegral <$> (peek (castPtr ptr) :: IO CUInt) :: IO Int
  -- print $ "SIZE " ++ show sz
  -- 2nd value of struct is the array of tokens
  tokens <- peekByteOff ptr 8 :: IO (Ptr CUInt)
  mapM
    (\idx -> (peekByteOff tokens (step * idx) :: IO CUInt) >>= (pure . fromIntegral))
    [0 .. sz -1]
  where
    step = 4
