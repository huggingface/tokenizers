The tokenization pipeline
====================================================================================================

When calling :meth:`~tokenizers.Tokenizer.encode` or :meth:`~tokenizers.Tokenizer.encode_batch`, the
input text(s) go through the following pipeline:

- :ref:`normalization`
- :ref:`pre-tokenization`
- :ref:`tokenization`
- :ref:`post-processing`

We'll see in details what happens during each of those steps in detail, as well as when you want to
:ref:`decode <decoding>` some token ids, and how the 🤗 Tokenizers library allows you to customize
each of those steps to your needs. 

For the examples that require a :class:`~tokenizers.Tokenizer`, we will use the tokenizer we trained
in the :doc:`quicktour`, which you can load with:

.. code-block:: python

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file("pretrained/wiki.json")


.. _normalization:

Normalization
----------------------------------------------------------------------------------------------------

Normalization is, in a nutshell, a set of operations you apply to a raw string to make it less
random or "cleaner". Common operations include stripping whitespace, removing accented characters
or lowercasing all text. If you're familiar with `unicode normalization
<https://unicode.org/reports/tr15>`__, it is also a very common normalization operation applied
in most tokenizers.

Each normalization operation is represented in the 🤗 Tokenizers library by a
:class:`~tokenizers.normalizers.Normalizer`, and you can combine several of those by using a
:class:`~tokenizers.normalizers.Sequence`. Here is a normalizer applying NFD Unicode normalization
and removing accents as an example:

.. code-block::

    import tokenizers
    from tokenizers.normalizers import NFD, StripAccents

    normalizer = tokenizers.normalizers.Sequence([NFD(), StripAccents()])

You can apply that normalizer to any string with the
:meth:`~tokenizers.normalizers.Normalizer.normalize_str` method:

.. code-block::

    normalizer.normalize_str("Héllò hôw are ü?")
    # "Hello how are u?"

When building a :class:`~tokenizers.Tokenizer`, you can customize its normalizer by just changing
the corresponding attribute:

.. code-block::

    tokenizer.normalizer = normalizer

Of course, if you change the way a tokenizer applies normalization, you should probably retrain it
from scratch afterward.

.. _pre-tokenization:

Pre-Tokenization
----------------------------------------------------------------------------------------------------

Pre-tokenization is the act of splitting a text into smaller objects that give an upper bound to
what your tokens will be at the end of training. A good way to think of this is that the
pre-tokenizer will split your text into "words" and then, your final tokens will be parts of those
words.

.. _tokenization:

Tokenization
----------------------------------------------------------------------------------------------------


.. _post-processing:

Post-Processing
----------------------------------------------------------------------------------------------------


.. _decoding:

Decoding
----------------------------------------------------------------------------------------------------

