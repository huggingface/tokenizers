#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_favicon_url = "https://huggingface.co/favicon.ico")]
#![doc(html_logo_url = "https://huggingface.co/landing/assets/huggingface_logo.svg")]

//! The 🤗 Tokenizers library.
//!
//! Starting with `0.23`, the implementation is split across two crates:
//!
//! - [`tk_encode`] — inference: the model engines, the full pipeline components
//!   ([`Normalizer`], [`PreTokenizer`], [`Model`], [`PostProcessor`],
//!   [`Decoder`]) and the [`Tokenizer`] orchestration (encode / decode).
//! - [`tk_train`] — training: the [`Trainer`] trait, every concrete `*Trainer`,
//!   and the [`TokenizerTrainExt`] extension that adds `train` /
//!   `train_from_files` onto a [`Tokenizer`].
//!
//! This `tokenizers` crate is a thin umbrella that re-exports both so existing
//! `tokenizers::…` paths keep working. Training lives behind the (default-on)
//! `train` feature; disable default features for an inference-only build.
//!
//! ## Deserialization and tokenization example
//!
//! ```no_run
//! use tokenizers::tokenizer::{Result, Tokenizer, EncodeInput};
//! use tokenizers::models::bpe::BPE;
//!
//! fn main() -> Result<()> {
//!     let bpe_builder = BPE::from_file("./path/to/vocab.json", "./path/to/merges.txt");
//!     let bpe = bpe_builder
//!         .dropout(0.1)
//!         .unk_token("[UNK]".into())
//!         .build()?;
//!
//!     let mut tokenizer = Tokenizer::new(bpe);
//!
//!     let encoding = tokenizer.encode("Hey there!", false)?;
//!     println!("{:?}", encoding.get_tokens());
//!
//!     Ok(())
//! }
//! ```

// ---------------------------------------------------------------------------
// Inference (always available) — re-exported from `tk-encode`.
// ---------------------------------------------------------------------------
pub use tk_encode::{decoders, normalizers, pre_tokenizers, processors, tokenizer, utils};

// Mirror the v1 top-level re-exports (`pub use tokenizer::*;` etc.).
pub use tk_encode::tokenizer::*;
pub use tk_encode::utils::parallelism;
pub use tk_encode::utils::ProgressFormat;

#[cfg(feature = "http")]
pub use tk_encode::FromPretrainedParameters;

// ---------------------------------------------------------------------------
// Models — inference engines, augmented with their trainers when `train` is on.
// ---------------------------------------------------------------------------
pub mod models {
    pub use tk_encode::models::*;

    #[cfg(feature = "train")]
    pub use tk_train::TrainerWrapper;

    pub mod bpe {
        pub use tk_encode::models::bpe::*;
        #[cfg(feature = "train")]
        pub use tk_train::trainers::bpe::*;
        /// Legacy module path: `tokenizers::models::bpe::trainer::BpeTrainer`.
        #[cfg(feature = "train")]
        pub mod trainer {
            pub use tk_train::trainers::bpe::*;
        }
    }

    pub mod unigram {
        pub use tk_encode::models::unigram::*;
        #[cfg(feature = "train")]
        pub use tk_train::trainers::unigram::*;
        /// Legacy module path: `tokenizers::models::unigram::trainer::UnigramTrainer`.
        #[cfg(feature = "train")]
        pub mod trainer {
            pub use tk_train::trainers::unigram::*;
        }
    }

    pub mod wordlevel {
        pub use tk_encode::models::wordlevel::*;
        #[cfg(feature = "train")]
        pub use tk_train::trainers::wordlevel::*;
        /// Legacy module path: `tokenizers::models::wordlevel::trainer::WordLevelTrainer`.
        #[cfg(feature = "train")]
        pub mod trainer {
            pub use tk_train::trainers::wordlevel::*;
        }
    }

    pub mod wordpiece {
        pub use tk_encode::models::wordpiece::*;
        #[cfg(feature = "train")]
        pub use tk_train::trainers::wordpiece::*;
        /// Legacy module path: `tokenizers::models::wordpiece::trainer::WordPieceTrainer`.
        #[cfg(feature = "train")]
        pub mod trainer {
            pub use tk_train::trainers::wordpiece::*;
        }
    }
}

// ---------------------------------------------------------------------------
// Training surface — only with the `train` feature.
// ---------------------------------------------------------------------------
#[cfg(feature = "train")]
pub use tk_train::{TokenizerTrainExt, Trainable, Trainer, TrainerWrapper};
