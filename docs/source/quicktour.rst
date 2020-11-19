Quicktour
====================================================================================================

Let's have a quick look at the 🤗 Tokenizers library features. The library provides an
implementation of today's most used tokenizers that is both easy to use and blazing fast.

.. only:: python

    It can be used to instantiate a :ref:`pretrained tokenizer <pretrained>` but we will start our
    quicktour by building one from scratch and see how we can train it.


Build a tokenizer from scratch
----------------------------------------------------------------------------------------------------

To illustrate how fast the 🤗 Tokenizers library is, let's train a new tokenizer on `wikitext-103
<https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/>`__ (516M of
text) in just a few seconds. First things first, you will need to download this dataset and unzip it
with:

.. code-block:: bash

    wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
    unzip wikitext-103-raw-v1.zip

Training the tokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. entities:: python

    BpeTrainer
        :class:`~tokenizers.trainers.BpeTrainer`
    vocab_size
        :obj:`vocab_size`
    min_frequency
        :obj:`min_frequency`
    special_tokens
        :obj:`special_tokens`
    unk_token
        :obj:`unk_token`
    pad_token
        :obj:`pad_token`

.. entities:: rust

    BpeTrainer
        :rust:struct:`~tokenizers::models::bpe::BpeTrainer`
    vocab_size
        :obj:`vocab_size`
    min_frequency
        :obj:`min_frequency`
    special_tokens
        :obj:`special_tokens`
    unk_token
        :obj:`unk_token`
    pad_token
        :obj:`pad_token`

.. entities:: node

    BpeTrainer
        BpeTrainer
    vocab_size
        :obj:`vocabSize`
    min_frequency
        :obj:`minFrequency`
    special_tokens
        :obj:`specialTokens`
    unk_token
        :obj:`unkToken`
    pad_token
        :obj:`padToken`

In this tour, we will build and train a Byte-Pair Encoding (BPE) tokenizer. For more information
about the different type of tokenizers, check out this `guide
<https://huggingface.co/transformers/tokenizer_summary.html>`__ in the 🤗 Transformers
documentation. Here, training the tokenizer means it will learn merge rules by:

- Start with all the characters present in the training corpus as tokens.
- Identify the most common pair of tokens and merge it into one token.
- Repeat until the vocabulary (e.g., the number of tokens) has reached the size we want.

The main API of the library is the :entity:`class` :entity:`Tokenizer`, here is how we instantiate
one with a BPE model:

.. only:: python

    .. literalinclude:: ../../bindings/python/tests/documentation/test_quicktour.py
        :language: python
        :start-after: START init_tokenizer
        :end-before: END init_tokenizer
        :dedent: 8

.. only:: rust

    .. literalinclude:: ../../tokenizers/tests/documentation.rs
        :language: rust
        :start-after: START quicktour_init_tokenizer
        :end-before: END quicktour_init_tokenizer
        :dedent: 4

.. only:: node

    .. literalinclude:: ../../bindings/node/examples/documentation/quicktour.test.ts
        :language: javascript
        :start-after: START init_tokenizer
        :end-before: END init_tokenizer
        :dedent: 8

To train our tokenizer on the wikitext files, we will need to instantiate a `trainer`, in this case
a :entity:`BpeTrainer`

.. only:: python

    .. literalinclude:: ../../bindings/python/tests/documentation/test_quicktour.py
        :language: python
        :start-after: START init_trainer
        :end-before: END init_trainer
        :dedent: 8

.. only:: rust

    .. literalinclude:: ../../tokenizers/tests/documentation.rs
        :language: rust
        :start-after: START quicktour_init_trainer
        :end-before: END quicktour_init_trainer
        :dedent: 4

.. only:: node

    .. literalinclude:: ../../bindings/node/examples/documentation/quicktour.test.ts
        :language: javascript
        :start-after: START init_trainer
        :end-before: END init_trainer
        :dedent: 8

We can set the training arguments like :entity:`vocab_size` or :entity:`min_frequency` (here left at
their default values of 30,000 and 0) but the most important part is to give the
:entity:`special_tokens` we plan to use later on (they are not used at all during training) so that
they get inserted in the vocabulary.

.. note::

    The order in which you write the special tokens list matters: here :obj:`"[UNK]"` will get the
    ID 0, :obj:`"[CLS]"` will get the ID 1 and so forth.

We could train our tokenizer right now, but it wouldn't be optimal. Without a pre-tokenizer that
will split our inputs into words, we might get tokens that overlap several words: for instance we
could get an :obj:`"it is"` token since those two words often appear next to each other. Using a
pre-tokenizer will ensure no token is bigger than a word returned by the pre-tokenizer. Here we want
to train a subword BPE tokenizer, and we will use the easiest pre-tokenizer possible by splitting
on whitespace.

.. only:: python

    .. literalinclude:: ../../bindings/python/tests/documentation/test_quicktour.py
        :language: python
        :start-after: START init_pretok
        :end-before: END init_pretok
        :dedent: 8

.. only:: rust

    .. literalinclude:: ../../tokenizers/tests/documentation.rs
        :language: rust
        :start-after: START quicktour_init_pretok
        :end-before: END quicktour_init_pretok
        :dedent: 4

.. only:: node

    .. literalinclude:: ../../bindings/node/examples/documentation/quicktour.test.ts
        :language: javascript
        :start-after: START init_pretok
        :end-before: END init_pretok
        :dedent: 8

Now, we can just call the :entity:`Tokenizer.train` method with any list of files we want
to use:

.. only:: python

    .. literalinclude:: ../../bindings/python/tests/documentation/test_quicktour.py
        :language: python
        :start-after: START train
        :end-before: END train
        :dedent: 8

.. only:: rust

    .. literalinclude:: ../../tokenizers/tests/documentation.rs
        :language: rust
        :start-after: START quicktour_train
        :end-before: END quicktour_train
        :dedent: 4

.. only:: node

    .. literalinclude:: ../../bindings/node/examples/documentation/quicktour.test.ts
        :language: javascript
        :start-after: START train
        :end-before: END train
        :dedent: 8

This should only take a few seconds to train our tokenizer on the full wikitext dataset!
To save the tokenizer in one file that contains all its configuration and vocabulary, just use the
:entity:`Tokenizer.save` method:

.. only:: python

    .. literalinclude:: ../../bindings/python/tests/documentation/test_quicktour.py
        :language: python
        :start-after: START save
        :end-before: END save
        :dedent: 8

.. only:: rust

    .. literalinclude:: ../../tokenizers/tests/documentation.rs
        :language: rust
        :start-after: START quicktour_save
        :end-before: END quicktour_save
        :dedent: 4

.. only:: node

    .. literalinclude:: ../../bindings/node/examples/documentation/quicktour.test.ts
        :language: javascript
        :start-after: START save
        :end-before: END save
        :dedent: 8

and you can reload your tokenizer from that file with the :entity:`Tokenizer.from_file`
:entity:`classmethod`:

.. only:: python

    .. literalinclude:: ../../bindings/python/tests/documentation/test_quicktour.py
        :language: python
        :start-after: START reload_tokenizer
        :end-before: END reload_tokenizer
        :dedent: 12

.. only:: rust

    .. literalinclude:: ../../tokenizers/tests/documentation.rs
        :language: rust
        :start-after: START quicktour_reload_tokenizer
        :end-before: END quicktour_reload_tokenizer
        :dedent: 4

.. only:: node

    .. literalinclude:: ../../bindings/node/examples/documentation/quicktour.test.ts
        :language: javascript
        :start-after: START reload_tokenizer
        :end-before: END reload_tokenizer
        :dedent: 8

Using the tokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that we have trained a tokenizer, we can use it on any text we want with the
:entity:`Tokenizer.encode` method:

.. only:: python

    .. literalinclude:: ../../bindings/python/tests/documentation/test_quicktour.py
        :language: python
        :start-after: START encode
        :end-before: END encode
        :dedent: 8

.. only:: rust

    .. literalinclude:: ../../tokenizers/tests/documentation.rs
        :language: rust
        :start-after: START quicktour_encode
        :end-before: END quicktour_encode
        :dedent: 4

.. only:: node

    .. literalinclude:: ../../bindings/node/examples/documentation/quicktour.test.ts
        :language: javascript
        :start-after: START encode
        :end-before: END encode
        :dedent: 8

This applied the full pipeline of the tokenizer on the text, returning an
:entity:`Encoding` object. To learn more about this pipeline, and how to apply (or
customize) parts of it, check out :doc:`this page <pipeline>`.

This :entity:`Encoding` object then has all the attributes you need for your deep
learning model (or other). The :obj:`tokens` attribute contains the segmentation of your text in
tokens:

.. only:: python

    .. literalinclude:: ../../bindings/python/tests/documentation/test_quicktour.py
        :language: python
        :start-after: START print_tokens
        :end-before: END print_tokens
        :dedent: 8

.. only:: rust

    .. literalinclude:: ../../tokenizers/tests/documentation.rs
        :language: rust
        :start-after: START quicktour_print_tokens
        :end-before: END quicktour_print_tokens
        :dedent: 4

.. only:: node

    .. literalinclude:: ../../bindings/node/examples/documentation/quicktour.test.ts
        :language: javascript
        :start-after: START print_tokens
        :end-before: END print_tokens
        :dedent: 8

Similarly, the :obj:`ids` attribute will contain the index of each of those tokens in the
tokenizer's vocabulary:

.. only:: python

    .. literalinclude:: ../../bindings/python/tests/documentation/test_quicktour.py
        :language: python
        :start-after: START print_ids
        :end-before: END print_ids
        :dedent: 8

.. only:: rust

    .. literalinclude:: ../../tokenizers/tests/documentation.rs
        :language: rust
        :start-after: START quicktour_print_ids
        :end-before: END quicktour_print_ids
        :dedent: 4

.. only:: node

    .. literalinclude:: ../../bindings/node/examples/documentation/quicktour.test.ts
        :language: javascript
        :start-after: START print_ids
        :end-before: END print_ids
        :dedent: 8

An important feature of the 🤗 Tokenizers library is that it comes with full alignment tracking,
meaning you can always get the part of your original sentence that corresponds to a given token.
Those are stored in the :obj:`offsets` attribute of our :entity:`Encoding` object. For
instance, let's assume we would want to find back what caused the :obj:`"[UNK]"` token to appear,
which is the token at index 9 in the list, we can just ask for the offset at the index:

.. only:: python

    .. literalinclude:: ../../bindings/python/tests/documentation/test_quicktour.py
        :language: python
        :start-after: START print_offsets
        :end-before: END print_offsets
        :dedent: 8

.. only:: rust

    .. literalinclude:: ../../tokenizers/tests/documentation.rs
        :language: rust
        :start-after: START quicktour_print_offsets
        :end-before: END quicktour_print_offsets
        :dedent: 4

.. only:: node

    .. literalinclude:: ../../bindings/node/examples/documentation/quicktour.test.ts
        :language: javascript
        :start-after: START print_offsets
        :end-before: END print_offsets
        :dedent: 8

and those are the indices that correspond to the emoji in the original sentence:

.. only:: python

    .. literalinclude:: ../../bindings/python/tests/documentation/test_quicktour.py
        :language: python
        :start-after: START use_offsets
        :end-before: END use_offsets
        :dedent: 8

.. only:: rust

    .. literalinclude:: ../../tokenizers/tests/documentation.rs
        :language: rust
        :start-after: START quicktour_use_offsets
        :end-before: END quicktour_use_offsets
        :dedent: 4

.. only:: node

    .. literalinclude:: ../../bindings/node/examples/documentation/quicktour.test.ts
        :language: javascript
        :start-after: START use_offsets
        :end-before: END use_offsets
        :dedent: 8

Post-processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We might want our tokenizer to automatically add special tokens, like :obj:`"[CLS]"` or
:obj:`"[SEP]"`. To do this, we use a post-processor. :entity:`TemplateProcessing` is the
most commonly used, you just have to specify a template for the processing of single sentences and
pairs of sentences, along with the special tokens and their IDs.

When we built our tokenizer, we set :obj:`"[CLS]"` and :obj:`"[SEP]"` in positions 1 and 2 of our
list of special tokens, so this should be their IDs. To double-check, we can use the
:entity:`Tokenizer.token_to_id` method:

.. only:: python

    .. literalinclude:: ../../bindings/python/tests/documentation/test_quicktour.py
        :language: python
        :start-after: START check_sep
        :end-before: END check_sep
        :dedent: 8

.. only:: rust

    .. literalinclude:: ../../tokenizers/tests/documentation.rs
        :language: rust
        :start-after: START quicktour_check_sep
        :end-before: END quicktour_check_sep
        :dedent: 4

.. only:: node

    .. literalinclude:: ../../bindings/node/examples/documentation/quicktour.test.ts
        :language: javascript
        :start-after: START check_sep
        :end-before: END check_sep
        :dedent: 8

Here is how we can set the post-processing to give us the traditional BERT inputs:

.. only:: python

    .. literalinclude:: ../../bindings/python/tests/documentation/test_quicktour.py
        :language: python
        :start-after: START init_template_processing
        :end-before: END init_template_processing
        :dedent: 8

.. only:: rust

    .. literalinclude:: ../../tokenizers/tests/documentation.rs
        :language: rust
        :start-after: START quicktour_init_template_processing
        :end-before: END quicktour_init_template_processing
        :dedent: 4

.. only:: node

    .. literalinclude:: ../../bindings/node/examples/documentation/quicktour.test.ts
        :language: javascript
        :start-after: START init_template_processing
        :end-before: END init_template_processing
        :dedent: 8

Let's go over this snippet of code in more details. First we specify the template for single
sentences: those should have the form :obj:`"[CLS] $A [SEP]"` where :obj:`$A` represents our
sentence.

Then, we specify the template for sentence pairs, which should have the form
:obj:`"[CLS] $A [SEP] $B [SEP]"` where :obj:`$A` represents the first sentence and :obj:`$B` the
second one. The :obj:`:1` added in the template represent the `type IDs` we want for each part of
our input: it defaults to 0 for everything (which is why we don't have :obj:`$A:0`) and here we set
it to 1 for the tokens of the second sentence and the last :obj:`"[SEP]"` token.

Lastly, we specify the special tokens we used and their IDs in our tokenizer's vocabulary.

To check out this worked properly, let's try to encode the same sentence as before:

.. only:: python

    .. literalinclude:: ../../bindings/python/tests/documentation/test_quicktour.py
        :language: python
        :start-after: START print_special_tokens
        :end-before: END print_special_tokens
        :dedent: 8

.. only:: rust

    .. literalinclude:: ../../tokenizers/tests/documentation.rs
        :language: rust
        :start-after: START quicktour_print_special_tokens
        :end-before: END quicktour_print_special_tokens
        :dedent: 4

.. only:: node

    .. literalinclude:: ../../bindings/node/examples/documentation/quicktour.test.ts
        :language: javascript
        :start-after: START print_special_tokens
        :end-before: END print_special_tokens
        :dedent: 8

To check the results on a pair of sentences, we just pass the two sentences to
:entity:`Tokenizer.encode`:

.. only:: python

    .. literalinclude:: ../../bindings/python/tests/documentation/test_quicktour.py
        :language: python
        :start-after: START print_special_tokens_pair
        :end-before: END print_special_tokens_pair
        :dedent: 8

.. only:: rust

    .. literalinclude:: ../../tokenizers/tests/documentation.rs
        :language: rust
        :start-after: START quicktour_print_special_tokens_pair
        :end-before: END quicktour_print_special_tokens_pair
        :dedent: 4

.. only:: node

    .. literalinclude:: ../../bindings/node/examples/documentation/quicktour.test.ts
        :language: javascript
        :start-after: START print_special_tokens_pair
        :end-before: END print_special_tokens_pair
        :dedent: 8

You can then check the type IDs attributed to each token is correct with

.. only:: python

    .. literalinclude:: ../../bindings/python/tests/documentation/test_quicktour.py
        :language: python
        :start-after: START print_type_ids
        :end-before: END print_type_ids
        :dedent: 8

.. only:: rust

    .. literalinclude:: ../../tokenizers/tests/documentation.rs
        :language: rust
        :start-after: START quicktour_print_type_ids
        :end-before: END quicktour_print_type_ids
        :dedent: 4

.. only:: node

    .. literalinclude:: ../../bindings/node/examples/documentation/quicktour.test.ts
        :language: javascript
        :start-after: START print_type_ids
        :end-before: END print_type_ids
        :dedent: 8

If you save your tokenizer with :entity:`Tokenizer.save`, the post-processor will be saved along.

Encoding multiple sentences in a batch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To get the full speed of the 🤗 Tokenizers library, it's best to process your texts by batches by
using the :entity:`Tokenizer.encode_batch` method:

.. only:: python

    .. literalinclude:: ../../bindings/python/tests/documentation/test_quicktour.py
        :language: python
        :start-after: START encode_batch
        :end-before: END encode_batch
        :dedent: 8

.. only:: rust

    .. literalinclude:: ../../tokenizers/tests/documentation.rs
        :language: rust
        :start-after: START quicktour_encode_batch
        :end-before: END quicktour_encode_batch
        :dedent: 4

.. only:: node

    .. literalinclude:: ../../bindings/node/examples/documentation/quicktour.test.ts
        :language: javascript
        :start-after: START encode_batch
        :end-before: END encode_batch
        :dedent: 8

The output is then a list of :entity:`Encoding` objects like the ones we saw before. You
can process together as many texts as you like, as long as it fits in memory.

To process a batch of sentences pairs, pass two lists to the
:entity:`Tokenizer.encode_batch` method: the list of sentences A and the list of sentences
B:

.. only:: python

    .. literalinclude:: ../../bindings/python/tests/documentation/test_quicktour.py
        :language: python
        :start-after: START encode_batch_pair
        :end-before: END encode_batch_pair
        :dedent: 8

.. only:: rust

    .. literalinclude:: ../../tokenizers/tests/documentation.rs
        :language: rust
        :start-after: START quicktour_encode_batch_pair
        :end-before: END quicktour_encode_batch_pair
        :dedent: 4

.. only:: node

    .. literalinclude:: ../../bindings/node/examples/documentation/quicktour.test.ts
        :language: javascript
        :start-after: START encode_batch_pair
        :end-before: END encode_batch_pair
        :dedent: 8

When encoding multiple sentences, you can automatically pad the outputs to the longest sentence
present by using :entity:`Tokenizer.enable_padding`, with the :entity:`pad_token` and its ID
(which we can double-check the id for the padding token with
:entity:`Tokenizer.token_to_id` like before):

.. only:: python

    .. literalinclude:: ../../bindings/python/tests/documentation/test_quicktour.py
        :language: python
        :start-after: START enable_padding
        :end-before: END enable_padding
        :dedent: 8

.. only:: rust

    .. literalinclude:: ../../tokenizers/tests/documentation.rs
        :language: rust
        :start-after: START quicktour_enable_padding
        :end-before: END quicktour_enable_padding
        :dedent: 4

.. only:: node

    .. literalinclude:: ../../bindings/node/examples/documentation/quicktour.test.ts
        :language: javascript
        :start-after: START enable_padding
        :end-before: END enable_padding
        :dedent: 8

We can set the :obj:`direction` of the padding (defaults to the right) or a given :obj:`length` if
we want to pad every sample to that specific number (here we leave it unset to pad to the size of
the longest text).

.. only:: python

    .. literalinclude:: ../../bindings/python/tests/documentation/test_quicktour.py
        :language: python
        :start-after: START print_batch_tokens
        :end-before: END print_batch_tokens
        :dedent: 8

.. only:: rust

    .. literalinclude:: ../../tokenizers/tests/documentation.rs
        :language: rust
        :start-after: START quicktour_print_batch_tokens
        :end-before: END quicktour_print_batch_tokens
        :dedent: 4

.. only:: node

    .. literalinclude:: ../../bindings/node/examples/documentation/quicktour.test.ts
        :language: javascript
        :start-after: START print_batch_tokens
        :end-before: END print_batch_tokens
        :dedent: 8

In this case, the `attention mask` generated by the tokenizer takes the padding into account:

.. only:: python

    .. literalinclude:: ../../bindings/python/tests/documentation/test_quicktour.py
        :language: python
        :start-after: START print_attention_mask
        :end-before: END print_attention_mask
        :dedent: 8

.. only:: rust

    .. literalinclude:: ../../tokenizers/tests/documentation.rs
        :language: rust
        :start-after: START quicktour_print_attention_mask
        :end-before: END quicktour_print_attention_mask
        :dedent: 4

.. only:: node

    .. literalinclude:: ../../bindings/node/examples/documentation/quicktour.test.ts
        :language: javascript
        :start-after: START print_attention_mask
        :end-before: END print_attention_mask
        :dedent: 8

.. _pretrained:

.. only:: python

    Using a pretrained tokenizer
    ----------------------------------------------------------------------------------------------------

    You can also use a pretrained tokenizer directly in, as long as you have its vocabulary file. For
    instance, here is how to get the classic pretrained BERT tokenizer:

    .. code-block:: python

        from tokenizers import BertWordPieceTokenizer

        tokenizer = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True)

    as long as you have downloaded the file `bert-base-uncased-vocab.txt` with

    .. code-block:: bash

        wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt

    .. note::

        Better support for pretrained tokenizers is coming in a next release, so expect this API to
        change soon.
