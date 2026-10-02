# tk-train

Training half of the 🤗 Tokenizers library.

This crate builds on top of [`tk_encode`] and provides everything related to
*training* a tokenizer: `TokenizerTrainer`, the entry point, the `ModelTrainer`
trait and every concrete `*Trainer`, and the `TrainerWrapper` enum that dispatches
over them.

`TokenizerTrainer` holds the training pipeline and a trainer, runs the text through
the pipeline, trains, and returns the finished tokenizer.

License: Apache-2.0
