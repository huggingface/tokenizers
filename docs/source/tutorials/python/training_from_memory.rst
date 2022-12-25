Training from memory
----------------------------------------------------------------------------------------------------

In the `Quicktour <quicktour>`__, we saw how to build and train a tokenizer using text files,
but we can actually use any Python Iterator. In this section we'll see a few different ways of
training our tokenizer.

For all the examples listed below, we'll use the same :class:`~tokenizers.Tokenizer` and
:class:`~tokenizers.trainers.Trainer`, built as following:

.. literalinclude:: ../../../../bindings/python/tests/documentation/test_tutorial_train_from_iterators.py
    :language: python
    :start-after: START init_tokenizer_trainer
    :end-before: END init_tokenizer_trainer
    :dedent: 8

This tokenizer is based on the :class:`~tokenizers.models.Unigram` model. It takes care of
normalizing the input using the NFKC Unicode normalization method, and uses a
:class:`~tokenizers.pre_tokenizers.ByteLevel` pre-tokenizer with the corresponding decoder.

For more information on the components used here, you can check `here <components>`__

The most basic way
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As you probably guessed already, the easiest way to train our tokenizer is by using a :obj:`List`:

.. literalinclude:: ../../../../bindings/python/tests/documentation/test_tutorial_train_from_iterators.py
    :language: python
    :start-after: START train_basic
    :end-before: END train_basic
    :dedent: 8

Easy, right? You can use anything working as an iterator here, be it a :obj:`List`, :obj:`Tuple`,
or a :obj:`np.Array`. Anything works as long as it provides strings.

Using the 🤗 Datasets library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An awesome way to access one of the many datasets that exist out there is by using the 🤗 Datasets
library. For more information about it, you should check
`the official documentation here <https://huggingface.co/docs/datasets/>`__.

Let's start by loading our dataset:

.. literalinclude:: ../../../../bindings/python/tests/documentation/test_tutorial_train_from_iterators.py
    :language: python
    :start-after: START load_dataset
    :end-before: END load_dataset
    :dedent: 8

The next step is to build an iterator over this dataset. The easiest way to do this is probably by
using a generator:

.. literalinclude:: ../../../../bindings/python/tests/documentation/test_tutorial_train_from_iterators.py
    :language: python
    :start-after: START def_batch_iterator
    :end-before: END def_batch_iterator
    :dedent: 8

As you can see here, for improved efficiency we can actually provide a batch of examples used
to train, instead of iterating over them one by one. By doing so, we can expect performances very
similar to those we got while training directly from files.

With our iterator ready, we just need to launch the training. In order to improve the look of our
progress bars, we can specify the total length of the dataset:

.. literalinclude:: ../../../../bindings/python/tests/documentation/test_tutorial_train_from_iterators.py
    :language: python
    :start-after: START train_datasets
    :end-before: END train_datasets
    :dedent: 8

And that's it!

Using gzip files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since gzip files in Python can be used as iterators, it is extremely simple to train on such files:

.. literalinclude:: ../../../../bindings/python/tests/documentation/test_tutorial_train_from_iterators.py
    :language: python
    :start-after: START single_gzip
    :end-before: END single_gzip
    :dedent: 8

Now if we wanted to train from multiple gzip files, it wouldn't be much harder:

.. literalinclude:: ../../../../bindings/python/tests/documentation/test_tutorial_train_from_iterators.py
    :language: python
    :start-after: START multi_gzip
    :end-before: END multi_gzip
    :dedent: 8

And voilà!
