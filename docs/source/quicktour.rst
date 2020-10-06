Quicktour
====================================================================================================

Let's have a quick look at the 🤗 Tokenizers library features. The library provides an
implementation of today's most used tokenizers that is both easy to use and blazing fast.

Load and use a pretrained tokenizer
----------------------------------------------------------------------------------------------------

Preprocess one sentence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use a pretrained tokenizer, you will need to download its "vocab file". For our example, let's
use the tokenizer of the base BERT model:

.. code-block:: bash

    wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt

Once you have downloaded this file, you can instantiate the associated BERT tokenizer in juste one
line:

.. code-block:: python

    from tokenizers import BertWordPieceTokenizer
    tokenizer = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True)

To use this tokenizer object on some text, just call its :meth:`~tokenizers.Tokenizer.encode`
method:

.. code-block:: python

    output = tokenizer.encode("Hello, y'all! How are you 😁 ?")

This applied the full pipeline of the tokenizer on the text, returning an
:class:`~tokenizers.Encoding` object. To learn more about this pipeline, and how to apply (or
customize) parts of it, check out :doc:`this apge <pipeline>`.

This :class:`~tokenizers.Encoding` object then has all the attributes you need for your deep
learning model (or other). The :obj:`tokens` attribute contains the segmentation of your text in
tokens:

.. code-block:: python

    print(output.tokens)
    # ["[CLS]", "hello", ",", "y", "'", "all", "!", "how", "are", "you", "[UNK]", "?", "[SEP]"]

Note that the tokenizer automatically added the special tokens required by the model (here
``"[CLS]"`` and ``"[SEP]"``) and replaced the smiley by the unknown token (here ``"[UNK]"``).

Similarly, the :obj:`ids` attribute will contain the index of each of those tokens in the
tokenizer's vocabulary:

.. code-block:: python

    print(output.ids)
    # [101, 7592, 1010, 1061, 1005, 2035, 999, 2129, 2024, 2017, 100, 1029, 102]

An important feature of the 🤗 Tokenizers library is that it comes with full alignmenbt tracking,
meaning you can always get the part of your original sentence that corresponds to a given token.
Those are stored in the :obj:`offsets` attribute of our :class:`~tokenizers.Encoding` object. For
instance, let's assume we would want to find back what caused the :obj:`"[UNK]"` token to appear,
which is the token at index 10 in the list, we can just ask for the offset at the index:

.. code-block:: python

    print(output.offsets[10])
    # (25, 26)

and those are the indices that correspond to the smiler in the original sentence:

.. code-block::

    sentence = "Hello, y'all! How are you 😁 ?"
    sentence[26:27]
    # "😁"

Preprocess a pair of sentences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your problem requires preprocessing pairs of sentences together, you can still use the
:meth:`~tokenizers.Tokenizer.encode` method:

.. code-block:: python

    output = tokenizer.encode("Hello, y'all!", "How are you 😁 ?")

Like for one sentence, the tokenizer will add the special tokens between them automatically:

.. code-block:: python

    print(output.tokens)
    # ["[CLS]", "hello", ",", "y", "'", "all", "[SEP]", "!", "how", "are", "you", "[UNK]", "?", "[SEP]"]

You can then access the token type ids for each token (e.g., which tokens are in the first sentence
and which are in the second sentence) with the :obj:`type_ids` attribute:

.. code-block:: python

    print(output.type_ids)
    # [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]


