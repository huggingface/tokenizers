The tokenization pipeline
====================================================================================================

When calling :entity:`Tokenizer.encode` or :entity:`Tokenizer.encode_batch`, the input text(s) go
through the following pipeline:

- :ref:`normalization`
- :ref:`pre-tokenization`
- :ref:`model`
- :ref:`post-processing`

We'll see in details what happens during each of those steps in detail, as well as when you want to
:ref:`decode <decoding>` some token ids, and how the 🤗 Tokenizers library allows you to customize
each of those steps to your needs. If you're already familiar with those steps and want to learn by
seeing some code, jump to :ref:`our BERT from scratch example <example>`.

For the examples that require a :entity:`Tokenizer`, we will use the tokenizer we trained
in the :doc:`quicktour`, which you can load with:

.. only:: python

    .. literalinclude:: ../../bindings/python/tests/documentation/test_pipeline.py
        :language: python
        :start-after: START reload_tokenizer
        :end-before: END reload_tokenizer
        :dedent: 8

.. only:: rust

    .. literalinclude:: ../../tokenizers/tests/documentation.rs
        :language: rust
        :start-after: START pipeline_reload_tokenizer
        :end-before: END pipeline_reload_tokenizer
        :dedent: 4

.. only:: node

    .. literalinclude:: ../../bindings/node/examples/documentation/pipeline.test.ts
        :language: javascript
        :start-after: START reload_tokenizer
        :end-before: END reload_tokenizer
        :dedent: 8


.. _normalization:

Normalization
----------------------------------------------------------------------------------------------------

Normalization is, in a nutshell, a set of operations you apply to a raw string to make it less
random or "cleaner". Common operations include stripping whitespace, removing accented characters
or lowercasing all text. If you're familiar with `Unicode normalization
<https://unicode.org/reports/tr15>`__, it is also a very common normalization operation applied
in most tokenizers.

Each normalization operation is represented in the 🤗 Tokenizers library by a
:entity:`Normalizer`, and you can combine several of those by using a
:entity:`normalizers.Sequence`. Here is a normalizer applying NFD Unicode normalization
and removing accents as an example:

.. only:: python

    .. literalinclude:: ../../bindings/python/tests/documentation/test_pipeline.py
        :language: python
        :start-after: START setup_normalizer
        :end-before: END setup_normalizer
        :dedent: 8

.. only:: rust

    .. literalinclude:: ../../tokenizers/tests/documentation.rs
        :language: rust
        :start-after: START pipeline_setup_normalizer
        :end-before: END pipeline_setup_normalizer
        :dedent: 4

.. only:: node

    .. literalinclude:: ../../bindings/node/examples/documentation/pipeline.test.ts
        :language: javascript
        :start-after: START setup_normalizer
        :end-before: END setup_normalizer
        :dedent: 8


You can manually test that normalizer by applying it to any string:

.. only:: python

    .. literalinclude:: ../../bindings/python/tests/documentation/test_pipeline.py
        :language: python
        :start-after: START test_normalizer
        :end-before: END test_normalizer
        :dedent: 8

.. only:: rust

    .. literalinclude:: ../../tokenizers/tests/documentation.rs
        :language: rust
        :start-after: START pipeline_test_normalizer
        :end-before: END pipeline_test_normalizer
        :dedent: 4

.. only:: node

    .. literalinclude:: ../../bindings/node/examples/documentation/pipeline.test.ts
        :language: javascript
        :start-after: START test_normalizer
        :end-before: END test_normalizer
        :dedent: 8


When building a :entity:`Tokenizer`, you can customize its normalizer by just changing
the corresponding attribute:

.. only:: python

    .. literalinclude:: ../../bindings/python/tests/documentation/test_pipeline.py
        :language: python
        :start-after: START replace_normalizer
        :end-before: END replace_normalizer
        :dedent: 8

.. only:: rust

    .. literalinclude:: ../../tokenizers/tests/documentation.rs
        :language: rust
        :start-after: START pipeline_replace_normalizer
        :end-before: END pipeline_replace_normalizer
        :dedent: 4

.. only:: node

    .. literalinclude:: ../../bindings/node/examples/documentation/pipeline.test.ts
        :language: javascript
        :start-after: START replace_normalizer
        :end-before: END replace_normalizer
        :dedent: 8

Of course, if you change the way a tokenizer applies normalization, you should probably retrain it
from scratch afterward.


.. _pre-tokenization:

Pre-Tokenization
----------------------------------------------------------------------------------------------------

Pre-tokenization is the act of splitting a text into smaller objects that give an upper bound to
what your tokens will be at the end of training. A good way to think of this is that the
pre-tokenizer will split your text into "words" and then, your final tokens will be parts of those
words.

An easy way to pre-tokenize inputs is to split on spaces and punctuations, which is done by the
:entity:`pre_tokenizers.Whitespace` pre-tokenizer:

.. only:: python

    .. literalinclude:: ../../bindings/python/tests/documentation/test_pipeline.py
        :language: python
        :start-after: START setup_pre_tokenizer
        :end-before: END setup_pre_tokenizer
        :dedent: 8

.. only:: rust

    .. literalinclude:: ../../tokenizers/tests/documentation.rs
        :language: rust
        :start-after: START pipeline_setup_pre_tokenizer
        :end-before: END pipeline_setup_pre_tokenizer
        :dedent: 4

.. only:: node

    .. literalinclude:: ../../bindings/node/examples/documentation/pipeline.test.ts
        :language: javascript
        :start-after: START setup_pre_tokenizer
        :end-before: END setup_pre_tokenizer
        :dedent: 8

The output is a list of tuples, with each tuple containing one word and its span in the original
sentence (which is used to determine the final :obj:`offsets` of our :entity:`Encoding`).
Note that splitting on punctuation will split contractions like :obj:`"I'm"` in this example.

You can combine together any :entity:`PreTokenizer` together. For
instance, here is a pre-tokenizer that will split on space, punctuation and digits, separating
numbers in their individual digits:

.. only:: python

    .. literalinclude:: ../../bindings/python/tests/documentation/test_pipeline.py
        :language: python
        :start-after: START combine_pre_tokenizer
        :end-before: END combine_pre_tokenizer
        :dedent: 8

.. only:: rust

    .. literalinclude:: ../../tokenizers/tests/documentation.rs
        :language: rust
        :start-after: START pipeline_combine_pre_tokenizer
        :end-before: END pipeline_combine_pre_tokenizer
        :dedent: 4

.. only:: node

    .. literalinclude:: ../../bindings/node/examples/documentation/pipeline.test.ts
        :language: javascript
        :start-after: START combine_pre_tokenizer
        :end-before: END combine_pre_tokenizer
        :dedent: 8

As we saw in the :doc:`quicktour`, you can customize the pre-tokenizer of a
:entity:`Tokenizer` by just changing the corresponding attribute:

.. only:: python

    .. literalinclude:: ../../bindings/python/tests/documentation/test_pipeline.py
        :language: python
        :start-after: START replace_pre_tokenizer
        :end-before: END replace_pre_tokenizer
        :dedent: 8

.. only:: rust

    .. literalinclude:: ../../tokenizers/tests/documentation.rs
        :language: rust
        :start-after: START pipeline_replace_pre_tokenizer
        :end-before: END pipeline_replace_pre_tokenizer
        :dedent: 4

.. only:: node

    .. literalinclude:: ../../bindings/node/examples/documentation/pipeline.test.ts
        :language: javascript
        :start-after: START replace_pre_tokenizer
        :end-before: END replace_pre_tokenizer
        :dedent: 8

Of course, if you change the way the pre-tokenizer, you should probably retrain your tokenizer from
scratch afterward.


.. _model:

The Model
----------------------------------------------------------------------------------------------------

Once the input texts are normalized and pre-tokenized, the :entity:`Tokenizer` applies the model on
the pre-tokens.  This is the part of the pipeline that needs training on your corpus (or that has
been trained if you are using a pretrained tokenizer).

The role of the model is to split your "words" into tokens, using the rules it has learned. It's
also responsible for mapping those tokens to their corresponding IDs in the vocabulary of the model.

This model is passed along when intializing the :entity:`Tokenizer` so you already know
how to customize this part. Currently, the 🤗 Tokenizers library supports:

- :entity:`models.BPE`
- :entity:`models.Unigram`
- :entity:`models.WordLevel`
- :entity:`models.WordPiece`

For more details about each model and its behavior, you can check `here <components.html#models>`__


.. _post-processing:

Post-Processing
----------------------------------------------------------------------------------------------------

Post-processing is the last step of the tokenization pipeline, to perform any additional
transformation to the :entity:`Encoding` before it's returned, like adding potential
special tokens.

As we saw in the quick tour, we can customize the post processor of a :entity:`Tokenizer`
by setting the corresponding attribute. For instance, here is how we can post-process to make the
inputs suitable for the BERT model:

.. only:: python

    .. literalinclude:: ../../bindings/python/tests/documentation/test_pipeline.py
        :language: python
        :start-after: START setup_processor
        :end-before: END setup_processor
        :dedent: 8

.. only:: rust

    .. literalinclude:: ../../tokenizers/tests/documentation.rs
        :language: rust
        :start-after: START pipeline_setup_processor
        :end-before: END pipeline_setup_processor
        :dedent: 4

.. only:: node

    .. literalinclude:: ../../bindings/node/examples/documentation/pipeline.test.ts
        :language: javascript
        :start-after: START setup_processor
        :end-before: END setup_processor
        :dedent: 8

Note that contrarily to the pre-tokenizer or the normalizer, you don't need to retrain a tokenizer
after changing its post-processor.

.. _example:

All together: a BERT tokenizer from scratch
----------------------------------------------------------------------------------------------------

Let's put all those pieces together to build a BERT tokenizer. First, BERT relies on WordPiece, so
we instantiate a new :entity:`Tokenizer` with this model:

.. only:: python

    .. literalinclude:: ../../bindings/python/tests/documentation/test_pipeline.py
        :language: python
        :start-after: START bert_setup_tokenizer
        :end-before: END bert_setup_tokenizer
        :dedent: 8

.. only:: rust

    .. literalinclude:: ../../tokenizers/tests/documentation.rs
        :language: rust
        :start-after: START bert_setup_tokenizer
        :end-before: END bert_setup_tokenizer
        :dedent: 4

.. only:: node

    .. literalinclude:: ../../bindings/node/examples/documentation/pipeline.test.ts
        :language: javascript
        :start-after: START bert_setup_tokenizer
        :end-before: END bert_setup_tokenizer
        :dedent: 8

Then we know that BERT preprocesses texts by removing accents and lowercasing. We also use a unicode
normalizer:

.. only:: python

    .. literalinclude:: ../../bindings/python/tests/documentation/test_pipeline.py
        :language: python
        :start-after: START bert_setup_normalizer
        :end-before: END bert_setup_normalizer
        :dedent: 8

.. only:: rust

    .. literalinclude:: ../../tokenizers/tests/documentation.rs
        :language: rust
        :start-after: START bert_setup_normalizer
        :end-before: END bert_setup_normalizer
        :dedent: 4

.. only:: node

    .. literalinclude:: ../../bindings/node/examples/documentation/pipeline.test.ts
        :language: javascript
        :start-after: START bert_setup_normalizer
        :end-before: END bert_setup_normalizer
        :dedent: 8

The pre-tokenizer is just splitting on whitespace and punctuation:

.. only:: python

    .. literalinclude:: ../../bindings/python/tests/documentation/test_pipeline.py
        :language: python
        :start-after: START bert_setup_pre_tokenizer
        :end-before: END bert_setup_pre_tokenizer
        :dedent: 8

.. only:: rust

    .. literalinclude:: ../../tokenizers/tests/documentation.rs
        :language: rust
        :start-after: START bert_setup_pre_tokenizer
        :end-before: END bert_setup_pre_tokenizer
        :dedent: 4

.. only:: node

    .. literalinclude:: ../../bindings/node/examples/documentation/pipeline.test.ts
        :language: javascript
        :start-after: START bert_setup_pre_tokenizer
        :end-before: END bert_setup_pre_tokenizer
        :dedent: 8

And the post-processing uses the template we saw in the previous section:

.. only:: python

    .. literalinclude:: ../../bindings/python/tests/documentation/test_pipeline.py
        :language: python
        :start-after: START bert_setup_processor
        :end-before: END bert_setup_processor
        :dedent: 8

.. only:: rust

    .. literalinclude:: ../../tokenizers/tests/documentation.rs
        :language: rust
        :start-after: START bert_setup_processor
        :end-before: END bert_setup_processor
        :dedent: 4

.. only:: node

    .. literalinclude:: ../../bindings/node/examples/documentation/pipeline.test.ts
        :language: javascript
        :start-after: START bert_setup_processor
        :end-before: END bert_setup_processor
        :dedent: 8

We can use this tokenizer and train on it on wikitext like in the :doc:`quicktour`:

.. only:: python

    .. literalinclude:: ../../bindings/python/tests/documentation/test_pipeline.py
        :language: python
        :start-after: START bert_train_tokenizer
        :end-before: END bert_train_tokenizer
        :dedent: 8

.. only:: rust

    .. literalinclude:: ../../tokenizers/tests/documentation.rs
        :language: rust
        :start-after: START bert_train_tokenizer
        :end-before: END bert_train_tokenizer
        :dedent: 4

.. only:: node

    .. literalinclude:: ../../bindings/node/examples/documentation/pipeline.test.ts
        :language: javascript
        :start-after: START bert_train_tokenizer
        :end-before: END bert_train_tokenizer
        :dedent: 8


.. _decoding:

Decoding
----------------------------------------------------------------------------------------------------

.. entities:: python

    bert_tokenizer
        :obj:`bert_tokenizer`

.. entities:: rust

    bert_tokenizer
        :obj:`bert_tokenizer`

.. entities:: node

    bert_tokenizer
        :obj:`bertTokenizer`


On top of encoding the input texts, a :entity:`Tokenizer` also has an API for decoding,
that is converting IDs generated by your model back to a text. This is done by the methods
:entity:`Tokenizer.decode` (for one predicted text) and :entity:`Tokenizer.decode_batch` (for a
batch of predictions).

The `decoder` will first convert the IDs back to tokens (using the tokenizer's vocabulary) and
remove all special tokens, then join those tokens with spaces:

.. only:: python

    .. literalinclude:: ../../bindings/python/tests/documentation/test_pipeline.py
        :language: python
        :start-after: START test_decoding
        :end-before: END test_decoding
        :dedent: 8

.. only:: rust

    .. literalinclude:: ../../tokenizers/tests/documentation.rs
        :language: rust
        :start-after: START pipeline_test_decoding
        :end-before: END pipeline_test_decoding
        :dedent: 4

.. only:: node

    .. literalinclude:: ../../bindings/node/examples/documentation/pipeline.test.ts
        :language: javascript
        :start-after: START test_decoding
        :end-before: END test_decoding
        :dedent: 8

If you used a model that added special characters to represent subtokens of a given "word" (like
the :obj:`"##"` in WordPiece) you will need to customize the `decoder` to treat them properly. If we
take our previous :entity:`bert_tokenizer` for instance the default decoing will give:

.. only:: python

    .. literalinclude:: ../../bindings/python/tests/documentation/test_pipeline.py
        :language: python
        :start-after: START bert_test_decoding
        :end-before: END bert_test_decoding
        :dedent: 8

.. only:: rust

    .. literalinclude:: ../../tokenizers/tests/documentation.rs
        :language: rust
        :start-after: START bert_test_decoding
        :end-before: END bert_test_decoding
        :dedent: 4

.. only:: node

    .. literalinclude:: ../../bindings/node/examples/documentation/pipeline.test.ts
        :language: javascript
        :start-after: START bert_test_decoding
        :end-before: END bert_test_decoding
        :dedent: 8

But by changing it to a proper decoder, we get:

.. only:: python

    .. literalinclude:: ../../bindings/python/tests/documentation/test_pipeline.py
        :language: python
        :start-after: START bert_proper_decoding
        :end-before: END bert_proper_decoding
        :dedent: 8

.. only:: rust

    .. literalinclude:: ../../tokenizers/tests/documentation.rs
        :language: rust
        :start-after: START bert_proper_decoding
        :end-before: END bert_proper_decoding
        :dedent: 4

.. only:: node

    .. literalinclude:: ../../bindings/node/examples/documentation/pipeline.test.ts
        :language: javascript
        :start-after: START bert_proper_decoding
        :end-before: END bert_proper_decoding
        :dedent: 8
