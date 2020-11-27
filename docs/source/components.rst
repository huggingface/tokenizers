Components
====================================================================================================

When building a Tokenizer, you can attach various types of components to this Tokenizer in order
to customize its behavior. This page lists most provided components.

.. _normalizers:


.. entities:: python

    BertNormalizer.clean_text
        clean_text
    BertNormalizer.handle_chinese_chars
        handle_chinese_chars
    BertNormalizer.strip_accents
        strip_accents
    BertNormalizer.lowercase
        lowercase
    Normalizer.Sequence
        ``Sequence([NFKC(), Lowercase()])``
    PreTokenizer.Sequence
        ``Sequence([Punctuation(), WhitespaceSplit()])``
    SplitDelimiterBehavior.removed
        :obj:`removed`
    SplitDelimiterBehavior.isolated
        :obj:`isolated`
    SplitDelimiterBehavior.merged_with_previous
        :obj:`merged_with_previous`
    SplitDelimiterBehavior.merged_with_next
        :obj:`merged_with_next`
    SplitDelimiterBehavior.contiguous
        :obj:`contiguous`

.. entities:: rust

    BertNormalizer.clean_text
        clean_text
    BertNormalizer.handle_chinese_chars
        handle_chinese_chars
    BertNormalizer.strip_accents
        strip_accents
    BertNormalizer.lowercase
        lowercase
    Normalizer.Sequence
        ``Sequence::new(vec![NFKC, Lowercase])``
    PreTokenizer.Sequence
        ``Sequence::new(vec![Punctuation, WhitespaceSplit])``
    SplitDelimiterBehavior.removed
        :obj:`Removed`
    SplitDelimiterBehavior.isolated
        :obj:`Isolated`
    SplitDelimiterBehavior.merged_with_previous
        :obj:`MergedWithPrevious`
    SplitDelimiterBehavior.merged_with_next
        :obj:`MergedWithNext`
    SplitDelimiterBehavior.contiguous
        :obj:`Contiguous`

.. entities:: node

    BertNormalizer.clean_text
        cleanText
    BertNormalizer.handle_chinese_chars
        handleChineseChars
    BertNormalizer.strip_accents
        stripAccents
    BertNormalizer.lowercase
        lowercase
    Normalizer.Sequence
        ..
    PreTokenizer.Sequence
        ..
    SplitDelimiterBehavior.removed
        :obj:`removed`
    SplitDelimiterBehavior.isolated
        :obj:`isolated`
    SplitDelimiterBehavior.merged_with_previous
        :obj:`mergedWithPrevious`
    SplitDelimiterBehavior.merged_with_next
        :obj:`mergedWithNext`
    SplitDelimiterBehavior.contiguous
        :obj:`contiguous`

Normalizers
----------------------------------------------------------------------------------------------------

A ``Normalizer`` is in charge of pre-processing the input string in order to normalize it as
relevant for a given use case. Some common examples of normalization are the Unicode normalization
algorithms (NFD, NFKD, NFC & NFKC), lowercasing etc...
The specificity of ``tokenizers`` is that we keep track of the alignment while normalizing. This
is essential to allow mapping from the generated tokens back to the input text.

The ``Normalizer`` is optional.

.. list-table::
   :header-rows: 1

   * - Name
     - Description
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
     - Input: ``HELLO ὈΔΥΣΣΕΎΣ``

       Output: ``hello ὀδυσσεύς``

   * - Strip
     - Removes all whitespace characters on the specified sides (left, right or both) of the input
     - Input: ``" hi "``

       Output: ``"hi"``

   * - StripAccents
     - Removes all accent symbols in unicode (to be used with NFD for consistency)
     - Input: ``é``

       Ouput: ``e``

   * - Replace
     - Replaces a custom string or regexp and changes it with given content
     - ``Replace("a", "e")`` will behave like this:

       Input: ``"banana"``
       Ouput: ``"benene"``

   * - BertNormalizer
     - Provides an implementation of the Normalizer used in the original BERT. Options
       that can be set are:

            - :entity:`BertNormalizer.clean_text`
            - :entity:`BertNormalizer.handle_chinese_chars`
            - :entity:`BertNormalizer.strip_accents`
            - :entity:`BertNormalizer.lowercase`

     -

   * - Sequence
     - Composes multiple normalizers that will run in the provided order
     - :entity:`Normalizer.Sequence`


.. _pre-tokenizers:

Pre tokenizers
----------------------------------------------------------------------------------------------------

The ``PreTokenizer`` takes care of splitting the input according to a set of rules. This
pre-processing lets you ensure that the underlying ``Model`` does not build tokens across multiple
"splits".
For example if you don't want to have whitespaces inside a token, then you can have a
``PreTokenizer`` that splits on these whitespaces.

You can easily combine multiple ``PreTokenizer`` together using a ``Sequence`` (see below).
The ``PreTokenizer`` is also allowed to modify the string, just like a ``Normalizer`` does. This
is necessary to allow some complicated algorithms that require to split before normalizing (e.g.
the ByteLevel)

.. list-table::
   :header-rows: 1

   * - Name
     - Description
     - Example

   * - ByteLevel
     - Splits on whitespaces while remapping all the bytes to a set of visible characters. This
       technique as been introduced by OpenAI with GPT-2 and has some more or less nice properties:

        - Since it maps on bytes, a tokenizer using this only requires **256** characters as initial
          alphabet (the number of values a byte can have), as opposed to the 130,000+ Unicode
          characters.
        - A consequence of the previous point is that it is absolutely unnecessary to have an
          unknown token using this since we can represent anything with 256 tokens (Youhou!! 🎉🎉)
        - For non ascii characters, it gets completely unreadable, but it works nonetheless!

     - Input: ``"Hello my friend, how are you?"``

       Ouput: ``"Hello", "Ġmy", Ġfriend", ",", "Ġhow", "Ġare", "Ġyou", "?"``

   * - Whitespace
     - Splits on word boundaries (using the following regular expression: ``\w+|[^\w\s]+``
     - Input: ``"Hello there!"``

       Output: ``"Hello", "there", "!"``

   * - WhitespaceSplit
     - Splits on any whitespace character
     - Input: ``"Hello there!"``

       Output: ``"Hello", "there!"``

   * - Punctuation
     - Will isolate all punctuation characters
     - Input: ``"Hello?"``

       Ouput: ``"Hello", "?"``

   * - Metaspace
     - Splits on whitespaces and replaces them with a special char "▁" (U+2581)
     - Input: ``"Hello there"``

       Ouput: ``"Hello", "▁there"``

   * - CharDelimiterSplit
     - Splits on a given character
     - Example with ``x``:

       Input: ``"Helloxthere"``

       Ouput: ``"Hello", "there"``

   * - Digits
     - Splits the numbers from any other characters.
     - Input: ``"Hello123there"``

       Output: ```"Hello", "123", "there"```

   * - Split
     - Versatile pre-tokenizer that splits on provided pattern and according to provided behavior.
       The pattern can be inverted if necessary.

         - pattern should be either a custom string or regexp.
         - behavior should be one of:

            * :entity:`SplitDelimiterBehavior.removed`
            * :entity:`SplitDelimiterBehavior.isolated`
            * :entity:`SplitDelimiterBehavior.merged_with_previous`
            * :entity:`SplitDelimiterBehavior.merged_with_next`
            * :entity:`SplitDelimiterBehavior.contiguous`

         - invert should be a boolean flag.

     - Example with `pattern` = :obj:`" "`, `behavior` = :obj:`"isolated"`, `invert` = :obj:`False`:

        Input: ``"Hello, how are you?"``

        Output: ```"Hello,", " ", "how", " ", "are", " ", "you?"```

   * - Sequence
     - Lets you compose multiple ``PreTokenizer`` that will be run in the given order
     - :entity:`PreTokenizer.Sequence`


.. _models:

Models
----------------------------------------------------------------------------------------------------

Models are the core algorithms used to actually tokenize, and therefore, they are the only mandatory
component of a Tokenizer.

.. list-table::
   :header-rows: 1

   * - Name
     - Description

   * - WordLevel
     - This is the "classic" tokenization algorithm. It let's you simply map words to IDs
       without anything fancy. This has the advantage of being really simple to use and
       understand, but it requires extremely large vocabularies for a good coverage.


       *Using this* ``Model`` *requires the use of a* ``PreTokenizer``. *No choice will be made by
       this model directly, it simply maps input tokens to IDs*

   * - BPE
     - One of the most popular subword tokenization algorithm. The Byte-Pair-Encoding works by
       starting with characters, while merging those that are the most frequently seen together,
       thus creating new tokens. It then works iteratively to build new tokens out of the most
       frequent pairs it sees in a corpus.

       BPE is able to build words it has never seen by using multiple subword tokens, and thus
       requires smaller vocabularies, with less chances of having "unk" (unknown) tokens.

   * - WordPiece
     - This is a subword tokenization algorithm quite similar to BPE, used mainly by Google in
       models like BERT. It uses a greedy algorithm, that tries to build long words first, splitting
       in multiple tokens when entire words don't exist in the vocabulary. This is different from
       BPE that starts from characters, building bigger tokens as possible.

       It uses the famous ``##`` prefix to identify tokens that are part of a word (ie not starting
       a word).

   * - Unigram
     - Unigram is also a subword tokenization algorithm, and works by trying to identify the best
       set of subword tokens to maximize the probability for a given sentence. This is different
       from BPE in the way that this is not deterministic based on a set of rules applied
       sequentially. Instead Unigram will be able to compute multiple ways of tokenizing, while
       choosing the most probable one.


.. _post-processors:

PostProcessor
----------------------------------------------------------------------------------------------------

After the whole pipeline, we sometimes want to insert some special tokens before feed
a tokenized string into a model like "[CLS] My horse is amazing [SEP]". The ``PostProcessor``
is the component doing just that.

.. list-table::
   :header-rows: 1

   * - Name
     - Description
     - Example
   * - TemplateProcessing
     - Let's you easily template the post processing, adding special tokens, and specifying
       the ``type_id`` for each sequence/special token. The template is given two strings
       representing the single sequence and the pair of sequences, as well as a set of
       special tokens to use.
     - Example, when specifying a template with these values:

            - single: ``"[CLS] $A [SEP]"``
            - pair: ``"[CLS] $A [SEP] $B [SEP]"``
            - special tokens:

                - ``"[CLS]"``
                - ``"[SEP]"``

       Input: ``("I like this", "but not this")``

       Output: ``"[CLS] I like this [SEP] but not this [SEP]"``


.. _decoders:

Decoders
----------------------------------------------------------------------------------------------------

The Decoder knows how to go from the IDs used by the Tokenizer, back to a readable piece of text.
Some ``Normalizer`` and ``PreTokenizer`` use special characters or identifiers that need to be
reverted for example.

.. list-table::
   :header-rows: 1

   * - Name
     - Description
   * - ByteLevel
     - Reverts the ByteLevel PreTokenizer. This PreTokenizer encodes at the byte-level, using
       a set of visible Unicode characters to represent each byte, so we need a Decoder to
       revert this process and get something readable again.
   * - Metaspace
     - Reverts the Metaspace PreTokenizer. This PreTokenizer uses a special identifer ``▁`` to
       identify whitespaces, and so this Decoder helps with decoding these.
   * - WordPiece
     - Reverts the WordPiece Model. This model uses a special identifier ``##`` for continuing
       subwords, and so this Decoder helps with decoding these.


