.. tokenizers documentation master file, created by
   sphinx-quickstart on Fri Sep 25 14:32:54 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to tokenizers's documentation!
======================================

.. toctree::

    tokenizer_blocks

Getting started
==================


Provides an implementation of today's most used tokenizers, with a focus on performance and
versatility.

Main features:
--------------

 - Train new vocabularies and tokenize, using today's most used tokenizers.
 - Extremely fast (both training and tokenization), thanks to the Rust implementation. Takes
   less than 20 seconds to tokenize a GB of text on a server's CPU.
 - Easy to use, but also extremely versatile.
 - Designed for research and production.
 - Normalization comes with alignments tracking. It's always possible to get the part of the
   original sentence that corresponds to a given token.
 - Does all the pre-processing: Truncate, Pad, add the special tokens your model needs.
 - Bindings to Rust, Python and Node.

Load an existing tokenizer:
---------------------------


.. tabs::

   .. group-tab:: Rust

      .. literalinclude:: ../../tokenizers/examples/load.rs
         :language: rust
         :emphasize-lines: 4

   .. group-tab:: Python

      .. literalinclude:: ../../bindings/python/tests/examples/test_load.py
         :language: python
         :emphasize-lines: 4

   .. group-tab:: Node

      .. literalinclude:: ../../bindings/node/examples/load.test.ts
         :language: typescript
         :emphasize-lines: 11



Train a tokenizer:
------------------

Small guide of :ref:`how to create a Tokenizer options<tokenizer_blocks>`.

.. tabs::
   .. group-tab:: Rust

      .. literalinclude:: ../../tokenizers/examples/train.rs
         :language: rust

   .. group-tab:: Python

      .. literalinclude:: ../../bindings/python/tests/examples/test_train.py
         :language: python

   .. group-tab:: Node

      .. literalinclude:: ../../bindings/node/examples/train.test.ts
         :language: typescript

