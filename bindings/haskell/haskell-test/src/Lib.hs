{-# LANGUAGE ForeignFunctionInterface #-}

module Lib where

import Foreign.Ptr
-- import Foreign.ForeignPtr
import Foreign.C.String ( CString, newCString )

data CTokenizer

data Tokenizer = Tokenizer { 
  tk :: Ptr CTokenizer,
  vocab :: String,
  merges :: String
  }

instance Show Tokenizer where
  show (Tokenizer _ vocab merges) = "Huggingface Tokenizer Object\n  vocab: " ++ vocab ++ "\n  merges: " ++ merges 

foreign import ccall unsafe "tokenize_test" r_tokenize_test :: CString -> IO ()
foreign import ccall unsafe "mk_tokenizer" r_mk_tokenizer :: CString -> CString -> IO (Ptr CTokenizer)
foreign import ccall unsafe "mk_roberta_tokenizer" r_mk_roberta_tokenizer :: CString -> CString -> IO (Ptr CTokenizer)
foreign import ccall unsafe "tokenize" r_tokenize :: CString -> Ptr CTokenizer -> IO ()

tokenizeTest x = do
    str <- newCString (x ++ "\0")
    r_tokenize_test str

mkTokenizer vocab merges = do
  cvocab <- newCString $ vocab ++ "\0"
  cmerges <- newCString $ merges ++ "\0"
  result <- r_mk_tokenizer cvocab cmerges
  pure (Tokenizer result vocab merges)


mkRobertaTokenizer vocab merges = do
  cvocab <- newCString $ vocab ++ "\0"
  cmerges <- newCString $ merges ++ "\0"
  result <- r_mk_roberta_tokenizer cvocab cmerges
  pure (Tokenizer result vocab merges)

tokenize text (Tokenizer tokenizer _ _) = do
  str <- newCString (text ++ "\0")
  r_tokenize str tokenizer
