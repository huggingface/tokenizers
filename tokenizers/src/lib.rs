#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_favicon_url = "https://huggingface.co/favicon.ico")]
#![doc(html_logo_url = "https://huggingface.co/landing/assets/huggingface_logo.svg")]

//! The 🤗 Tokenizers library.
//!
//! Starting with `0.23`, the implementation is split across two public crates (each built on internal
//! engines — `tk_encode` on the `atomsplit` SIMD pre-tokenizer, and the shared `bitmap_gen` tables):
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
pub use tk_encode::{pipeline, utils, vocab};

// Mirror the v1 top-level re-exports (`pub use tokenizer::*;` etc.).
pub use tk_encode::tokenizer::*;
pub use tk_encode::utils::ProgressFormat;
pub use tk_encode::utils::parallelism;

// ...and the config layer's half of the same surface. Both halves are glob-re-exported here so that
// `tokenizers::Tokenizer`, `tokenizers::AddedToken`, `tokenizers::ModelWrapper` and friends keep
// resolving at their historical paths after the `tk-encode` / `tk-convert` split.
pub use tk_convert::tokenizer::*;
pub use tk_convert::{
    DecoderWrapper, ModelWrapper, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper,
};

// The serde mirrors for `tk-encode`'s own types. `tk-encode` links no serde, so anything that needs
// to write one of its types out -- `PyEncoding`'s pickling, most visibly -- goes through here.
pub use tk_convert::mirror;

#[cfg(feature = "http")]
pub use tk_encode::FromPretrainedParameters;

// ---------------------------------------------------------------------------
// Components — the runtime structs from `tk-encode`, the wrapper enums and the
// wrapper-parameterised `Sequence`s from `tk-convert`. Each of these modules is a merge of
// the two halves, so every `tokenizers::<module>::…` path predates the split.
// ---------------------------------------------------------------------------
pub mod tokenizer {
    pub use tk_convert::tokenizer::*;
    pub use tk_convert::{
        DecoderWrapper, ModelWrapper, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper,
    };
    pub use tk_encode::tokenizer::*;
}

pub mod normalizers {
    pub use tk_convert::normalizers::{NormalizerWrapper, Sequence};
    pub use tk_encode::normalizers::*;

    pub mod utils {
        pub use tk_convert::normalizers::utils::Sequence;
        pub use tk_encode::normalizers::utils::*;
    }
}

pub mod pre_tokenizers {
    pub use tk_convert::pre_tokenizers::PreTokenizerWrapper;
    pub use tk_encode::pre_tokenizers::*;

    pub mod sequence {
        pub use tk_convert::pre_tokenizers::sequence::Sequence;
        pub use tk_encode::pre_tokenizers::sequence::*;
    }
}

pub mod processors {
    pub use tk_convert::processors::PostProcessorWrapper;
    pub use tk_encode::processors::*;

    pub mod sequence {
        pub use tk_convert::processors::sequence::Sequence;
    }
}

pub mod decoders {
    pub use tk_convert::decoders::DecoderWrapper;
    pub use tk_encode::decoders::*;

    pub mod sequence {
        pub use tk_convert::decoders::sequence::Sequence;
    }
}

// ---------------------------------------------------------------------------
// Models — inference engines, augmented with their trainers when `train` is on.
// ---------------------------------------------------------------------------
pub mod models {
    // `tk_encode::models` exports only the per-model submodules, and each is re-declared below with
    // its trainer merged in, so there is nothing left for a glob to bring over.
    pub use tk_convert::ModelWrapper;

    #[cfg(feature = "train")]
    pub use tk_train::TrainerWrapper;

    pub mod bpe {
        // Two halves now: `PipelineBPE` and the shared types are the runtime's, the config-shaped
        // `BPE` with its builder, its file loaders and the `Word` machinery is `tk-convert`'s. The
        // path `tokenizers::models::bpe::BPE` resolves to the same name it always did.
        pub use tk_convert::models::bpe::*;
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
        // `load` moved to `tk-convert` with the rest of the serde: it reads a whole model out of a
        // `unigram.json`, which the runtime crate has no parser for.
        pub use tk_convert::models::unigram::load;
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
        // `read_file` / `from_file` moved to `tk-convert`: a `vocab.json` needs a JSON parser.
        pub use tk_convert::models::wordlevel::{from_file, read_file};
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
        // The file loaders moved to `tk-convert`, with every other way of getting a model out of a
        // file. They are free functions there, because the model itself stayed in the runtime and an
        // inherent `impl` has to live with its type.
        pub use tk_convert::models::wordpiece::{
            from_bpe, from_bytes, from_file, read_bytes, read_file,
        };
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
