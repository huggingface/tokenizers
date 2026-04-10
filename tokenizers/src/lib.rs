#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(clippy::all)]
#![allow(clippy::upper_case_acronyms)]
#![doc(html_favicon_url = "https://huggingface.co/favicon.ico")]
#![doc(html_logo_url = "https://huggingface.co/landing/assets/huggingface_logo.svg")]

//! The core of `tokenizers`, written in Rust.
//! Provides an implementation of today's most used tokenizers, with a focus on performance and
//! versatility.
//!
//! # What is a Tokenizer
//!
//! A Tokenizer works as a pipeline, it processes some raw text as input and outputs an `Encoding`.
//! The various steps of the pipeline are:
//!
//! 1. The `Normalizer`: in charge of normalizing the text. Common examples of normalization are
//!    the [unicode normalization standards](https://unicode.org/reports/tr15/#Norm_Forms), such as `NFD` or `NFKC`.
//!    More details about how to use the `Normalizers` are available on the
//!    [Hugging Face blog](https://huggingface.co/docs/tokenizers/components#normalizers)
//! 2. The `PreTokenizer`: in charge of creating initial words splits in the text. The most common way of
//!    splitting text is simply on whitespace.
//! 3. The `Model`: in charge of doing the actual tokenization. An example of a `Model` would be
//!    `BPE` or `WordPiece`.
//! 4. The `PostProcessor`: in charge of post-processing the `Encoding` to add anything relevant
//!    that, for example, a language model would need, such as special tokens.
//!
//! ## Loading a pretrained tokenizer from the Hub
//! ```
//! use tokenizers::tokenizer::{Result, Tokenizer};
//!
//! fn main() -> Result<()> {
//!     # #[cfg(feature = "http")]
//!     # {
//!     // needs http feature enabled
//!     let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None)?;
//!
//!     let encoding = tokenizer.encode("Hey there!", false)?;
//!     println!("{:?}", encoding.get_tokens());
//!     # }
//!     Ok(())
//! }
//! ```
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
//!
//! ## Training and serialization example
//!
//! ```no_run
//! use tokenizers::decoders::DecoderWrapper;
//! use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
//! use tokenizers::normalizers::{strip::Strip, unicode::NFC, utils::Sequence, NormalizerWrapper};
//! use tokenizers::pre_tokenizers::byte_level::ByteLevel;
//! use tokenizers::pre_tokenizers::PreTokenizerWrapper;
//! use tokenizers::processors::PostProcessorWrapper;
//! use tokenizers::{AddedToken, Model, Result, TokenizerBuilder};
//!
//! use std::path::Path;
//!
//! fn main() -> Result<()> {
//!     let vocab_size: usize = 100;
//!
//!     let mut trainer = BpeTrainerBuilder::new()
//!         .show_progress(true)
//!         .vocab_size(vocab_size)
//!         .min_frequency(0)
//!         .special_tokens(vec![
//!             AddedToken::from(String::from("<s>"), true),
//!             AddedToken::from(String::from("<pad>"), true),
//!             AddedToken::from(String::from("</s>"), true),
//!             AddedToken::from(String::from("<unk>"), true),
//!             AddedToken::from(String::from("<mask>"), true),
//!         ])
//!         .build();
//!
//!     let mut tokenizer = TokenizerBuilder::new()
//!         .with_model(BPE::default())
//!         .with_normalizer(Some(Sequence::new(vec![
//!             Strip::new(true, true).into(),
//!             NFC.into(),
//!         ])))
//!         .with_pre_tokenizer(Some(ByteLevel::default()))
//!         .with_post_processor(Some(ByteLevel::default()))
//!         .with_decoder(Some(ByteLevel::default()))
//!         .build()?;
//!
//!     let pretty = false;
//!     tokenizer
//!         .train_from_files(
//!             &mut trainer,
//!             vec!["path/to/vocab.txt".to_string()],
//!         )?
//!         .save("tokenizer.json", pretty)?;
//!
//!     Ok(())
//! }
//! ```
//!
//! # Additional information
//!
//! - tokenizers is designed to leverage CPU parallelism when possible. The level of parallelism is determined
//!   by the total number of core/threads your CPU provides but this can be tuned by setting the `RAYON_RS_NUM_THREADS`
//!   environment variable. As an example setting `RAYON_RS_NUM_THREADS=4` will allocate a maximum of 4 threads.
//!   **_Please note this behavior may evolve in the future_**
//!
//! # Features
//!
//! All features are **enabled by default** for backward compatibility. Disable them for on-device/embedded use.
//!
//! | Feature | Default | Description | Deps saved |
//! |---------|---------|-------------|------------|
//! | `training` | on | Tokenizer training (trainers, `train()` method) | rand, esaxx-rs, compact_str |
//! | `parallel` | on | Multi-threaded encoding via rayon | rayon, rayon-cond, crossbeam |
//! | `spm` | on | SentencePiece precompiled normalizer (T5, mBART) | spm_precompiled, nom, unicode-segmentation |
//! | `unicode-normalization` | on | NFC/NFD/NFKC/NFKD normalizers | unicode-normalization-alignments |
//! | `progressbar` | on | Progress bars during training | indicatif |
//! | `onig` | on | Oniguruma regex engine (C binding) | onig, onig_sys |
//! | `http` | off | Download tokenizers from Hugging Face Hub | hf-hub, ureq |
//! | `unstable_wasm` | off | WASM target support (uses fancy-regex) | fancy-regex |
//!
//! ## On-device / embedded configuration
//!
//! ```toml
//! # Minimal inference-only (with Oniguruma regex):
//! tokenizers = { version = "0.22", default-features = false, features = ["onig"] }
//!
//! # WASM (pure Rust, no C dependencies):
//! tokenizers = { version = "0.22", default-features = false, features = ["unstable_wasm"] }
//! ```
//!
//! # Bundle size
//!
//! The deployed library size depends on how you link it. Here are measured sizes on macOS arm64:
//!
//! | Configuration | .dylib (shared) | .a (static) | After final link |
//! |---------------|----------------|-------------|-----------------|
//! | Default (all features) | 2.5 MB | 9.2 MB | ~2.5 MB |
//! | Inference-only (`onig`) | 2.0 MB | 8.0 MB | ~2.0 MB |
//!
//! > **Note**: `.a` (static archive) files contain all object code including unused functions.
//! > The linker strips dead code at final link time, so the actual contribution to your app
//! > binary is close to the `.dylib` size. The `.a` size is NOT what ships to users.
//!
//! ## Comparison with Meta pytorch/tokenizers (C++)
//!
//! | | Meta (C++) | HuggingFace (Rust) |
//! |---|---|---|
//! | Stripped binary (all tokenizer types) | **0.8 MB** | **2.0 MB** |
//! | Static .a (pre-link, all deps) | 5.5 MB | 8.0 MB |
//! | Features | SP, Tiktoken, Llama2c | BPE, WordPiece, Unigram, WordLevel + normalizers, pre-tokenizers, decoders, added vocab |
//!
//! HuggingFace is ~2.5x larger because it includes full `tokenizer.json` parsing (serde), Unicode-aware
//! regex, all normalizer/pre-tokenizer/decoder types, and added vocabulary matching — features Meta's
//! library doesn't have.
//!
//! ## How to measure bundle size
//!
//! **1. Measure the linked shared library (what ships to users):**
//!
//! ```bash
//! # Create a test crate that links tokenizers as a cdylib
//! cargo new --lib measure-size && cd measure-size
//! cat >> Cargo.toml << 'EOF'
//! [lib]
//! crate-type = ["cdylib"]
//!
//! [dependencies]
//! tokenizers = { path = "../tokenizers", default-features = false, features = ["onig"] }
//!
//! [profile.release]
//! lto = "fat"
//! opt-level = "s"
//! strip = true
//! EOF
//!
//! echo 'use tokenizers::Tokenizer;
//! #[no_mangle]
//! pub extern "C" fn tokenize() { let _ = Tokenizer::from_file("t.json"); }' > src/lib.rs
//!
//! cargo build --release
//! ls -lh target/release/*.dylib  # macOS
//! ls -lh target/release/*.so     # Linux
//! ```
//!
//! **2. Measure per-crate contribution with cargo-bloat:**
//!
//! ```bash
//! cargo install cargo-bloat
//! cargo bloat --release --crates -n 30
//! ```
//!
//! **3. Measure dependency rlib sizes (compile-time cost):**
//!
//! ```bash
//! # Total rlib for runtime deps only
//! cargo tree --edges=normal --prefix none -f '{p}' | awk '{print $1}' | sort -u | sed 's/-/_/g' > /tmp/deps.txt
//!
//! for f in target/release/deps/*.rlib; do
//!   sz=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null)
//!   name=$(basename "$f" | sed 's/-[a-f0-9]*\.rlib//' | sed 's/^lib//')
//!   echo "$sz $name"
//! done | sort -t' ' -k2 | awk '!seen[$2]++ {print}' | sort -k2 > /tmp/rlibs.txt
//!
//! join -1 2 -2 1 /tmp/rlibs.txt /tmp/deps.txt | awk '{
//!   total+=$2
//!   printf "%8.1f KB  %s\n", $2/1024, $1
//! } END {
//!   printf "\nTOTAL: %.1f MB\n", total/1048576
//! }' | sort -rn
//! ```
//!
//! **4. Track size in CI (regression test):**
//!
//! ```bash
//! #!/bin/bash
//! # scripts/check-bundle-size.sh
//! set -e
//!
//! MAX_DYLIB_KB=2500  # 2.5 MB threshold
//!
//! cargo build --release --no-default-features --features "onig" \
//!   --target-dir /tmp/size-check
//!
//! SIZE=$(stat -f%z /tmp/size-check/release/libtokenizers.rlib 2>/dev/null \
//!     || stat -c%s /tmp/size-check/release/libtokenizers.rlib)
//! SIZE_KB=$((SIZE / 1024))
//!
//! echo "libtokenizers.rlib: ${SIZE_KB} KB"
//!
//! # For the actual linked size, build a cdylib test crate
//! # (see step 1 above) and check the .dylib/.so size
//!
//! if [ "$SIZE_KB" -gt "$MAX_DYLIB_KB" ]; then
//!   echo "FAIL: bundle size ${SIZE_KB} KB exceeds threshold ${MAX_DYLIB_KB} KB"
//!   exit 1
//! fi
//! echo "PASS: bundle size OK"
//! ```

#[macro_use]
extern crate log;

#[macro_use]
pub mod utils;
pub mod decoders;
pub mod models;
pub mod normalizers;
pub mod pre_tokenizers;
pub mod processors;
pub mod tokenizer;

// Re-export from tokenizer
pub use tokenizer::*;

// Re-export also parallelism utils
pub use utils::parallelism;

// Re-export ProgressFormat for trainer configuration
pub use utils::ProgressFormat;

// Re-export for from_pretrained
#[cfg(feature = "http")]
pub use utils::from_pretrained::FromPretrainedParameters;
