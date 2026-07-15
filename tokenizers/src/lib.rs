#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_favicon_url = "https://huggingface.co/favicon.ico")]
#![doc(html_logo_url = "https://huggingface.co/landing/assets/huggingface_logo.svg")]

//! The 🤗 Tokenizers library.
//!
//! Starting with `0.23`, the implementation is split across two crates:
//!
//! - [`tk_encode`] — inference: the model engines, the pipeline components
//!   ([`Normalizer`], [`PreTokenizer`], [`Model`]) and the [`Tokenizer`] loader
//!   that feeds the fast `PipelineTokenizer` encode path.
//! - [`tk_train`] — training: the [`Trainer`] trait, every concrete `*Trainer`,
//!   and the [`TokenizerTrainExt`] extension that adds `train` /
//!   `train_from_files` onto a [`Tokenizer`].
//!
//! This `tokenizers` crate is a thin umbrella that re-exports both so existing
//! `tokenizers::…` paths keep working. Training lives behind the (default-on)
//! `train` feature; disable default features for an inference-only build.
//!
//! ## Load and encode (pipeline)
//!
//! ```no_run
//! use std::convert::TryFrom;
//! use tokenizers::tokenizer::{Result, Tokenizer};
//! use tokenizers::tokenizer::pipeline::PipelineTokenizer;
//!
//! fn main() -> Result<()> {
//!     let tok = Tokenizer::from_file("tokenizer.json")?;
//!     let pipeline = PipelineTokenizer::try_from(&tok)?;
//!     let ids = pipeline.encode("Hey there!", false)?;
//!     println!("{:?}", ids.len());
//!     Ok(())
//! }
//! ```

// ---------------------------------------------------------------------------
// Inference (always available) — re-exported from `tk-encode`.
// ---------------------------------------------------------------------------
pub use tk_encode::{normalizers, pre_tokenizers, tokenizer, utils};

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
