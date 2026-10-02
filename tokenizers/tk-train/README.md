# tk-train

Training half of the 🤗 Tokenizers library.

This crate builds on top of [`tk_encode`] and provides everything related to
*training* a tokenizer: [`TokenizerTrainer`], the entry point, the [`Trainer`]
trait and every concrete `*Trainer`, the [`TrainerWrapper`] enum, the [`Trainable`]
extension (the `get_trainer` association that used to live on `tk_encode::Model`),
and [`TokenizerBlueprint`], the tokenizer a trainer trains into.

[`TokenizerTrainer`] holds the blueprint and a trainer, checks that they agree,
runs the text through the pipeline, trains, and returns the finished tokenizer.

License: Apache-2.0
