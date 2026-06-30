# tk-train

Training half of the 🤗 Tokenizers library.

This crate builds on top of [`tk_encode`] and provides everything related to
*training* a tokenizer: the [`Trainer`] trait, every concrete `*Trainer`, the
[`TrainerWrapper`] enum, the [`Trainable`] extension (the `get_trainer`
association that used to live on `tk_encode::Model`), and the
[`TokenizerTrainExt`] extension that adds `train` / `train_from_files` back
onto `tk_encode`'s `Tokenizer`.

License: Apache-2.0
