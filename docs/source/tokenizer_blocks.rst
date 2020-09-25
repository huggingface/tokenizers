
Models
======

.. _tokenizer_blocks:

Models are the core algorithms that serves for tokenizers.

.. list-table::
   :header-rows: 1

   * - Name
     - Description
   * - BPE
     - Works by looking at most frequent pairs in a dataset, and iteratively fusing them in new tokens
   * - Unigram
     - Works by building a suffix array and using an EM algorithm to find best suitable tokens
   * - WordPiece
     - ...
   * - WordLevel
     - ...


Normalizers
===========

A normalizer will take a unicode string input, and modify it to make it more uniform for the underlying algorithm.
Usually fixes some unicode quirks. The specificity of ``tokenizers`` is that we keep track of all offsets
to know how a string was normalizers, which is especially useful to debug a tokenizer.

.. list-table::
   :header-rows: 1

   * - Name
     - Desription
     - Example
   * - NFD
     - NFD unicode normalization
     - 
   * - NFKD
     - NFKD unicode normalization
     - 
   * - NFC
     - NFC unicode normalization
     - 
   * - NFKC
     - NFKC unicode normalization
     - 
   * - Lowercase
     - Replaces all uppercase to lowercase
     - "HELLO ὈΔΥΣΣΕΎΣ" -> "hello ὀδυσσεύς"
   * - Strip
     - Removes all spacelike characters on the sides of input
     - " hi " ->  "hi"
   * - StripAccents
     - Removes all accent symbols in unicode (to be used with NFD for consistency)
     - "é" -> "e"
   * - Nmt
     - Removes some control characters and zero-width characters
     - "\u200d" -> ""
   * - Replace
     - Replaces a custom string or regexp and changes it with given content
     - Replace("a", "e")("banana") -> "benene"
   * - Sequence
     - Composes multiple normalizers
     - Sequence([Nmt(), NFKC()])


Pre tokenizers
==============

A pre tokenizer splits an input string *before* it reaches the model, it's often used for efficiency.
It can also replace some characters.

.. list-table::
   :header-rows: 1

   * - Name
     - Description
     - Example
   * - ByteLevel
     - Splits on spaces but remaps all bytes into visible range (used in gpt-2)
     - "Hello my friend, how are you?" -> "Hello", "Ġmy", Ġfriend", ",", "Ġhow", "Ġare", "Ġyou", "?"
   * - Whitespace
     - Splits on word boundaries
     - "Hello there!" -> "Hello", "there", "!"
   * - WhitespaceSplit
     - Splits on spaces
     - "Hello there!" -> "Hello", "there!"
   * - Punctuation
     - Will isolate all punctuation characters
     - "Hello?" -> "Hello", "?"
   * - Metaspace
     - Splits on spaces an replaces it with a special char
     - Metaspace("_", false)("Hello there") -> "Hello", "_there"
   * - CharDelimiterSplit
     - Splits on a given char
     - CharDelimiterSplit("x")("Helloxthere") -> "Hello", "there"
   * - Sequence
     - Composes multiple pre_tokenizers
     - Sequence([Punctuation(), WhitespaceSplit()])


Decoders
========

As some normalizers and pre_tokenizers change some characters, we want to revert some changes to get back readable strings

.. list-table::
   :header-rows: 1

   * - Name
     - Description
   * - ByteLevel
     - Reverts ByteLevel Pre_tokenizer
   * - Metaspace
     - Reverts Metaspace Pre_tokenizer


PostProcessor
=============

After the whole pipeline, we sometimes want to insert some specific markers before feeding
a tokenized string into a model like "`<cls>` My horse is amazing `<eos>`".

.. list-table::
   :header-rows: 1

   * - Name
     - Description
     - Example
   * - TemplateProcessing
     - It should covert most needs. `seq_a` is a list of the outputs for single sentence, `seq_b` is used when encoding two sentences
     - TemplateProcessing(seq_a = ["<cls>", "$0", "<eos>"], seq_b = ["$1", "<eos>"]) ("I like this", "but not this") -> "<cls> I like this <eos> but not this <eos>"

