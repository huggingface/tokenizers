{-# LANGUAGE ForeignFunctionInterface #-}

module Lib where

import Foreign.Ptr
-- import Foreign.ForeignPtr
import Foreign.Storable
import Foreign.C.String ( CString, newCString, peekCString )
import Foreign.C.Types

data CTokenizer
data CEncoding

data Tokens
data Tokenizer = Tokenizer {
  tok :: Ptr CTokenizer,
  vocab :: String,
  merges :: String
}

data Encoding = Encoding {
  enc :: Ptr CEncoding
}

instance Show Tokenizer where
  show (Tokenizer _ vocab merges) = "Huggingface Tokenizer Object\n  vocab: " ++ vocab ++ "\n  merges: " ++ merges 

foreign import ccall unsafe "tokenize_test" r_tokenize_test :: CString -> IO ()

tokenizeTest x = do
    str <- newCString (x ++ "\0")
    r_tokenize_test str

foreign import ccall unsafe "mk_tokenizer" r_mk_tokenizer :: CString -> CString -> IO (Ptr CTokenizer)

mkTokenizer vocab merges = do
  cvocab <- newCString $ vocab ++ "\0"
  cmerges <- newCString $ merges ++ "\0"
  result <- r_mk_tokenizer cvocab cmerges
  pure (Tokenizer result vocab merges)

foreign import ccall unsafe "mk_roberta_tokenizer" r_mk_roberta_tokenizer :: CString -> CString -> IO (Ptr CTokenizer)
  
mkRobertaTokenizer vocab merges = do
  cvocab <- newCString $ vocab ++ "\0"
  cmerges <- newCString $ merges ++ "\0"
  result <- r_mk_roberta_tokenizer cvocab cmerges
  pure (Tokenizer result vocab merges)

foreign import ccall unsafe "tokenize" r_tokenize :: CString -> Ptr CTokenizer -> IO ()

tokenize text (Tokenizer tokenizer _ _) = do
  str <- newCString (text ++ "\0")
  r_tokenize str tokenizer

foreign import ccall unsafe "encode" r_encode :: CString -> Ptr CTokenizer -> IO (Ptr CEncoding)

encode (Tokenizer tokenizer _ _) text = do
  str <- newCString text
  encoding <- r_encode str tokenizer
  pure (Encoding encoding)

foreign import ccall unsafe "get_tokens" r_get_tokens :: Ptr CEncoding -> IO (Ptr Tokens)

getTokens :: Encoding -> IO [String]
getTokens (Encoding encoding) = do
  ptr <- r_get_tokens encoding
  -- 1st value of struct is the # of tokens
  sz <- fromIntegral <$> (peek (castPtr ptr) :: IO CInt) :: IO Int
  -- 2nd value of struct is the array of tokens
  tokens <- peekByteOff ptr 8 :: IO (Ptr CString) 
  mapM 
    (\idx -> peekByteOff tokens (8*idx) >>= peekCString) 
    [0 .. sz-1]

cleanTokens xs = [x | x <- xs, not (x `elem` "\288")]
