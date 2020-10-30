Quicktour
====================================================================================================

Let's have a quick look at the 🤗 Tokenizers library features. The library provides an
implementation of today's most used tokenizers that is both easy to use and blazing fast.

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

In this tour, we will build and train a Byte-Pair Encoding (BPE) tokenzier. For more information
about the different type of tokenizers, check out this `guide
<https://huggingface.co/transformers/tokenizer_summary.html>`__ in the 🤗 Transformers
documentation. Here, training the tokenizer means it will learn merge rules by:

- Start with all the characters present in the training corpus as tokens.
- Identify the most common pair of tokens and merge it into one token.
- Repeat until the vocabulary (e.g., the number of tokens) has reached the size we want.

The main API of the library is the class :class:`~tokenizers.Tokenizer`, here is how we instantiate
one with a BPE model:

.. code-block:: python

    from tokenizers import Tokenizer
    from tokenizers.models import BPE

    tokenizer = Tokenizer(BPE())

To train our tokenizer on the wikitext files, we will need to instantiate a `trainer`, in this case
a :class:`~tokenizers.BpeTrainer`:

.. code-block:: python

    from tokenizers.trainers import BpeTrainer

    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

We can set the training arguments like :obj:`vocab_size` or :obj:`min_frequency` (here left at their
default values of 30,000 and 0) but the most important part is to give the :obj:`special_tokens` we
plan to use later on (they are not used at all during training) so that they get inserted in the
vocabulary.

.. note::

    The order in which you write the special tokens list matters: here :obj:`"[UNK]"` will get the
    ID 0, :obj:`"[CLS]"` will get the ID 1 and so forth.

We could train our tokenizer right now, but it wouldn't be optimal. Without a pre-tokenizer that
will split our inputs into words, we might get tokens that overlap several words: for instance we
could get an :obj:`"it is"` token since those two words often appear next to each other. Using a
pre-tokenizer will ensure no token is bigger than a word returned by the pre-tokenizer. Here we want
to train a subword BPE tokenizer, and we will use the easiest pre-tokenizer possible by splitting
on whitespace.

.. code-block:: python

    from tokenizers.pre_tokenizers import Whitespace

    tokenizer.pre_tokenizer = Whitespace()

Now, we can just call the :meth:`~tokenizers.Tokenizer.train` method with any list of files we want
to use:

.. code-block:: python

    files = [f"wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]
    tokenizer.train(trainer, files)

This should only take a few seconds to train our tokenizer on the full wikitext dataset! Once this
is done, we need to save the model and reinstantiate it with the unkown token, or this token won't
be used. This will be simplified in a further release, to let you set the :obj:`unk_token` when
first instantiating the model.

.. code-block:: python

    files = tokenizer.model.save("pretrained", "wiki")
    tokenizer.model = BPE(*files, unk_token="[UNK]")

To save the tokenizer in one file that contains all its configuration and vocabulary, just use the
:meth:`~tokenizers.Tokenizer.save` method:

.. code-block:: python

    tokenizer.save("pretrained/wiki.json")

and you can reload your tokenzier from that file with the :meth:`~tokenizers.Tokenizer.from_file`
class method:

.. code-block:: python

    tokenizer = Tokenizer.from_file("tst-tokenizer/wiki-trained.json")

Using the tokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that we have trained a tokenizer, we can use it on any text we want with the
:meth:`~tokenizers.Tokenizer.encode` method:

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
    # ["Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?"]

Similarly, the :obj:`ids` attribute will contain the index of each of those tokens in the
tokenizer's vocabulary:

.. code-block:: python

    print(output.ids)
    # [27194, 16, 93, 11, 5068, 5, 7928, 5083, 6190, 0, 35]

An important feature of the 🤗 Tokenizers library is that it comes with full alignmenbt tracking,
meaning you can always get the part of your original sentence that corresponds to a given token.
Those are stored in the :obj:`offsets` attribute of our :class:`~tokenizers.Encoding` object. For
instance, let's assume we would want to find back what caused the :obj:`"[UNK]"` token to appear,
which is the token at index 9 in the list, we can just ask for the offset at the index:

.. code-block:: python

    print(output.offsets[9])
    # (26, 27)

and those are the indices that correspond to the smiler in the original sentence:

.. code-block:: python

    sentence = "Hello, y'all! How are you 😁 ?"
    sentence[26:27]
    # "😁"

Post-processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We might want our tokenizer to automatically add special tokens, like :obj:`"[CLS]"` or
:obj:`"[SEP]"`. To do this, we use a post-processor. :class:`~tokenizers.TemplateProcessing` is the
most commonly used, you just have so specify a template for the processing of single sentences and
pairs of sentences, along with the special tokens and their IDs.

When we built our tokenizer, we set :obj:`"[CLS]"` and :obj:`"[SEP]"` in positions 1 and 2 of our
list of special tokens, so this should be their IDs. To double-check, we can use the
:meth:`~tokenizers.Tokenizer.token_to_id` method:

.. code-block:: python

    tokenizer.token_to_id("[SEP]")
    # 2

Here is how we can set the post-processing to give us the traditional BERT inputs:

.. code-block:: python

    from tokenizers.processors import TemplateProcessing

    tokenizer.post_processor = TemplateProcessing
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
    )

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

.. code-block:: python

    output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
    print(output.tokens)
    # ["[CLS]", "Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?", "[SEP]"]

To check the results on a pair of sentences, we just pass the two sentences to
:meth:`~tokenizers.Tokenizer.encode`:

.. code-block:: python

    output = tokenizer.encode("Hello, y'all!", "How are you 😁 ?")
    print(output.tokens)
    # ["[CLS]", "Hello", ",", "y", "'", "all", "!", "[SEP]", "How", "are", "you", "[UNK]", "?", "[SEP]"]

You can then check the type IDs attributed to each token is correct with

.. code-block:: python

    print(output.type_ids)
    # [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

If you save your tokenizer with :meth:`~tokenizers.Tokenizer.save`, the post-processor will be saved
along.

Encoding multiple sentences in a batch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To get the full speed of the 🤗 Tokenizers library, it's best to process your texts by batches by
using the :meth:`~tokenizers.Tokenizer.encode_batch` method:

.. code-block:: python

    output = tokenizer.encode_batch(["Hello, y'all!", "How are you 😁 ?"])

The output is then a list of :class:`~tokenizers.Encoding` objects like the ones we saw before. You
can process together as many texts as you like, as long as it fits in memory.

To process a batch of sentences pairs, pass two lists to the
:meth:`~tokenizers.Tokenizer.encode_batch` method: the list of sentences A and the list of sentences
B:

.. code-block:: python

    output = tokenizer.encode_batch(
        ["Hello, y'all!", "How are you 😁 ?"],
        ["Hello to you too!", "I'm fine, thank you!"]
    )

When encoding multiple sentences, you can automatically pad the outputs to the longest sentence
present by using :meth:`~tokenizers.Tokenizer.enable_padding`, with the :obj:`pad_token` and its ID
(which we can double-check the id for the padding token with
:meth:`~tokenizers.Tokenizer.token_to_id` like before):

.. code-block:: python

    tokenizer.enable_padding(pad_id=3, pad_token="[PAD]")

We can set the :obj:`direction` of the padding (defaults to the right) or a given :obj:`length` if
we want to pad every sample to that specific number (here we leave it unset to pad to the size of
the longest text).

.. code-block:: python

    output = tokenizer.encode_batch(["Hello, y'all!", "How are you 😁 ?"])
    print(output[1].tokens)
    # ["[CLS]", "How", "are", "you", "[UNK]", "?", "[SEP]", "[PAD]"]

In this case, the `attention mask` generated by the tokenizer takes the padding into account:

.. code-block:: python

    print(output[1].attention_mask)
    [1, 1, 1, 1, 1, 1, 1, 0]

.. _pretrained:

Using a pretrained tokenizer
----------------------------------------------------------------------------------------------------

You can also use a pretrained tokenizer directly in, as long as you have its vocabulary file. For
instance, here is how to get the classic pretrained BERT tokenizer:

.. code-block:: python

    from tokenizers import ByteLevelBPETokenizer

    tokenizer = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True)

as long as you have downloaded the file `bert-base-uncased-vocab.txt` with

.. code-block:: bash

    wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt

.. note::

    Better support for pretrained tokenziers is coming in a next release, so expect this API to
    change soon.
