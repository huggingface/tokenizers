# tk-train

Training half of the 🤗 Tokenizers library.

This crate builds on top of [`tk_encode`] and provides everything related to
*training* a tokenizer: the [`Trainer`] trait, every concrete `*Trainer`, the
[`TrainerWrapper`] enum, the [`Trainable`] extension (the `get_trainer`
association that used to live on `tk_encode::Model`), and [`train_tokenizer`].

[`train_tokenizer`] trains a fresh model through an existing pipeline's
normalization and pre-tokenization and returns a new pipeline retaining its
other stages. It is fresh training, not incremental training: the old model
vocabulary and old added tokens are not used to split the corpus or copied
into the result; only the trainer's special tokens are registered, so include
any specials retained settings need. Unigram byte fallback is rejected, and a
sequence longer than `u32::MAX` bytes after normalization is rejected before
pre-tokenization.

To train a model directly without a pipeline, drive a trainer itself instead:
`feed`, then `train`.

License: Apache-2.0
