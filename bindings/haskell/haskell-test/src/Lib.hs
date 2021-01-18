{-# LANGUAGE ForeignFunctionInterface #-}

module Lib where

import Foreign.Ptr
-- import Foreign.ForeignPtr
import Foreign.Storable
import Foreign.C.String ( CString, newCString, peekCString )

data CTokenizer
data CEncoding

data Tokenizer = Tokenizer { 
  tk :: Ptr CTokenizer,
  vocab :: String,
  merges :: String
  }

data Encoding = Encoding {
  enc :: Ptr CEncoding
}

instance Show Tokenizer where
  show (Tokenizer _ vocab merges) = "Huggingface Tokenizer Object\n  vocab: " ++ vocab ++ "\n  merges: " ++ merges 

foreign import ccall unsafe "tokenize_test" r_tokenize_test :: CString -> IO ()
foreign import ccall unsafe "mk_tokenizer" r_mk_tokenizer :: CString -> CString -> IO (Ptr CTokenizer)
foreign import ccall unsafe "tokenize" r_tokenize :: CString -> Ptr CTokenizer -> IO ()
foreign import ccall unsafe "mk_roberta_tokenizer" r_mk_roberta_tokenizer :: CString -> CString -> IO (Ptr CTokenizer)
foreign import ccall unsafe "encode" r_encode :: CString -> Ptr CTokenizer -> IO (Ptr CEncoding)
foreign import ccall unsafe "get_tokens" r_get_tokens :: Ptr CEncoding -> IO (Ptr CString)

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

encode (Tokenizer tokenizer _ _) text = do
  str <- newCString text
  encoding <- r_encode str tokenizer
  pure (Encoding encoding)

getTokens (Encoding encoding) = do
  tokens <- r_get_tokens encoding
  firstElem <- peek tokens
  value <- peekCString firstElem
  print value
  
