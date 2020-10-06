Tokenizers
====================================================================================================

Fast State-of-the-art tokenizers, optimized for both research and production

`🤗 Tokenizers`_ provides an implementation of today's most used tokenizers, with
a focus on performance and versatility. These tokenizers are also used in
`🤗 Transformers`_.

.. _🤗 Tokenizers: https://github.com/huggingface/tokenizers
.. _🤗 Transformers: https://github.com/huggingface/transformers

Main features:
----------------------------------------------------------------------------------------------------

 - Train new vocabularies and tokenize, using today's most used tokenizers.
 - Extremely fast (both training and tokenization), thanks to the Rust implementation. Takes
   less than 20 seconds to tokenize a GB of text on a server's CPU.
 - Easy to use, but also extremely versatile.
 - Designed for both research and production.
 - Full alignment tracking. Even with destructive normalization, it's always possible to get
   the part of the original sentence that corresponds to any token.
 - Does all the pre-processing: Truncation, Padding, add the special tokens your model needs.

Components:
----------------------------------------------------------------------------------------------------

.. toctree::
    :maxdepth: 2
    :caption: Getting Started

    quicktour
    installation
    pipeline
    components

.. toctree::
    :maxdepth: 3
    :caption: API Reference

    api/reference

Load an existing tokenizer:
----------------------------------------------------------------------------------------------------

Loading a previously saved tokenizer is extremely simple and requires a single line of code:

.. only:: rust

  .. literalinclude:: ../../tokenizers/tests/documentation.rs
     :language: rust
     :start-after: START load_tokenizer
     :end-before: END load_tokenizer
     :dedent: 4

.. only:: python

  .. literalinclude:: ../../bindings/python/tests/documentation/test_load.py
     :language: python
     :start-after: START load_tokenizer
     :end-before: END load_tokenizer
     :dedent: 4

.. only:: node

  .. literalinclude:: ../../bindings/node/examples/load.test.js
     :language: javascript
     :start-after: START load
     :end-before: END load
     :dedent: 4


Train a tokenizer:
----------------------------------------------------------------------------------------------------

.. only:: rust

  .. literalinclude:: ../../tokenizers/tests/documentation.rs
     :language: rust
     :start-after: START train_tokenizer
     :end-before: END train_tokenizer
     :dedent: 4

.. only:: python

  .. literalinclude:: ../../bindings/python/tests/documentation/test_train.py
     :language: python
     :start-after: START train_tokenizer
     :end-before: END train_tokenizer
     :dedent: 4

.. only:: node

  .. literalinclude:: ../../bindings/node/examples/train.test.js
     :language: javascript
     :start-after: START train_tokenizer
     :end-before: END train_tokenizer
     :dedent: 4
