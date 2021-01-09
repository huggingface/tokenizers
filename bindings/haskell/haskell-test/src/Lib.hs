{-# LANGUAGE ForeignFunctionInterface #-}

module Lib where

import Foreign.C.Types
import Foreign.C.String

foreign import ccall unsafe "tokenize" r_tokenize :: CString -> IO ()

tokenize x = do
    str <- newCString (x ++ "\0")
    r_tokenize str