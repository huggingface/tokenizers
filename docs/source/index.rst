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


.. toctree::
    :maxdepth: 2
    :caption: Getting Started

    quicktour
    installation/main
    pipeline
    components

.. toctree::
    :maxdepth: 3
    :caption: API Reference

    api/reference

.. include:: entities.inc
