//! Write a [`PipelineTokenizer`] back out as a canonical `tokenizer.json`, with no serde anywhere.
//!
//! The mirror of [`crate::from_json`], and deliberately its shape: one module per component, the
//! same order, the same names. What it is *not* is the inverse of it, because the pipeline is not
//! the config -- it is what the config was lowered into, and lowering threw things away.
//!
//! ## The pipeline is a lossy source, and this is what that costs
//!
//! Worth reading before touching anything here, because most of the code below is dealing with one
//! of these rather than transcribing a field:
//!
//! - **A BPE model has no vocabulary or merge list.** The merges were consumed into a perfect-hash
//!   map plus a dense grid, and a byte-level vocabulary was decoded to raw bytes.
//!   [`tk_encode::models::bpe::PipelineBPE::to_config`] runs that construction backwards. It is
//!   exact, and it is exact only because the grid encodes each pair in its slot index and the hash
//!   map stores each key beside its value -- a future table that drops either would break this
//!   silently, which is why that method carries the warning it does.
//! - **A `Metaspace` pre-tokenizer is a normalizer plus a `Split`**, and a **`ByteLevel`** is a
//!   `Split` on the GPT-2 regex or nothing at all. Both are recognised and folded back up in
//!   [`pre_tokenizers`], which is an inverse copy of the reader's rules -- the one real duplication
//!   in here.
//! - **A `Precompiled` normalizer's charsmap is only in the pipeline because it was put there for
//!   this.** `spm_precompiled` keeps the blob private and publishes it only through serde, so
//!   [`tk_encode::normalizers::precompiled::PrecompiledNormalizer`] carries the bytes alongside.
//! - **A post-processor's special tokens are ids, not names.** [`post_processors`] reconstructs
//!   names from the ids, which rebuilds the same template under a possibly different spelling.
//! - **Fields that cannot change an id are not recovered**: `trim_offsets` on a `ByteLevel`
//!   pre-tokenizer, `dropout`, and `unk_token`/`fuse_unk`/`byte_fallback` on a byte-level BPE, all
//!   of which the load path drops because nothing reads them.
//!
//! Ids are the contract, in other words, not bytes. Writing then reading gives back a pipeline that
//! encodes identically; it does not give back the file that was read, and `tests` asserts the first
//! rather than the second.
//!
//! ## Canonical only
//!
//! Every component gets an explicit `"type"`, merges are `["a", "b"]` pairs, and a `Metaspace` is
//! spelled with `prepend_scheme` and `split`. None of the legacy shapes [`crate::from_json`] still
//! tolerates are ever produced -- there is no reason to write a form that only exists to be read.
//!
//! ```no_run
//! let tokenizer = tk_serialize::from_json_file("./tokenizer.json")?;
//! let text = tk_serialize::to_json(&tokenizer)?;
//! # Ok::<(), tk_encode::Error>(())
//! ```

mod added_tokens;
mod decoders;
mod model;
mod normalizers;
mod post_processors;
mod pre_tokenizers;
mod writer;

// The tests read a config before writing it, so they need the reader as well.
#[cfg(all(test, feature = "deserialize"))]
mod tests;

use self::added_tokens::write_added_tokens;
use self::decoders::write_decoder;
use self::model::write_model;
use self::normalizers::write_normalizer;
use self::post_processors::write_post_processor;
use self::pre_tokenizers::write_pre_tokenizer;
use self::writer::Out;
use tk_encode::normalizers::metaspace::MetaspaceNormalizer;
use tk_encode::pipeline::{PipelineModel, PipelineNormalizer, PipelineTokenizer};
use tk_encode::tokenizer::Result;

// Free functions rather than inherent methods, for the same reason the reader's are: a
// `PipelineTokenizer` is `tk-encode`'s type, and knowing what a `tokenizer.json` looks like is this
// crate's job.
/// Write a `tokenizer.json` as a string.
pub fn to_json(tokenizer: &PipelineTokenizer) -> Result<String> {
    let mut out = Out::new();
    let model = tokenizer.get_model();
    // The `Metaspace` normalizer is half of a pre-tokenizer, so it leaves the chain here and is
    // written over there.
    let (normalizers, metaspace) = split_off_metaspace(tokenizer.get_normalizers());
    // Only the model knows whether a `ByteLevel` pre-tokenizer was in play: it is the flag that
    // made the model byte-level, and nothing else records it.
    let byte_level = match model {
        PipelineModel::BPE(bpe) => bpe.is_byte_level(),
        #[cfg(feature = "unigram")]
        PipelineModel::Unigram(_) => false,
        #[cfg(feature = "wordpiece")]
        PipelineModel::WordPiece(_) => false,
        #[cfg(feature = "wordlevel")]
        PipelineModel::WordLevel(_) => false,
    };

    // A post-processor template names its special tokens by id. The added vocabulary is asked first
    // because that is where a special token with anything unusual in it lives, and a byte-level
    // model would answer for the same id with its decoded bytes.
    let added_tokens = tokenizer.get_added_vocabulary().get_added_tokens_decoder();
    let name_of = |id: u32| -> Option<String> {
        added_tokens
            .get(&id)
            .map(|token| token.content.clone())
            .or_else(|| model.id_to_token(id))
    };

    // Key order follows what `Tokenizer::save` has always written, so a diff against a file from
    // the Hub lines up field by field.
    out.obj_open();
    out.field_str("version", "1.0");
    // Neither is part of a pipeline: both are encode-time settings the reader does not read and the
    // runtime does not keep. Spelled as `null` rather than omitted, which is how a tokenizer with
    // neither has always been written.
    out.field_null("truncation");
    out.field_null("padding");
    out.key("added_tokens");
    write_added_tokens(&mut out, tokenizer.get_added_vocabulary());
    out.key("normalizer");
    write_normalizer(&mut out, normalizers)?;
    out.key("pre_tokenizer");
    write_pre_tokenizer(
        &mut out,
        tokenizer.get_pre_tokenizer(),
        metaspace,
        byte_level,
    )?;
    out.key("post_processor");
    write_post_processor(&mut out, tokenizer.get_post_processor(), &name_of)?;
    out.key("decoder");
    write_decoder(&mut out, tokenizer.get_decoder())?;
    out.key("model");
    write_model(&mut out, model)?;
    out.obj_close();
    Ok(out.finish())
}

/// The chain without its trailing `Metaspace`, and that `Metaspace` if there was one.
///
/// It can only be the last member: the reader builds the declared `normalizer` first and the
/// pre-tokenizer appends to what it built, so a `Metaspace` normalizer is always the one on the end.
/// Anywhere else it did not come from a pre-tokenizer, and the caller reports that rather than
/// quietly writing it as a `normalizer` -- which would read back as a normalizer, and move ids.
fn split_off_metaspace(
    chain: &[PipelineNormalizer],
) -> (&[PipelineNormalizer], Option<&MetaspaceNormalizer>) {
    match chain.split_last() {
        Some((PipelineNormalizer::Metaspace(metaspace), rest)) => (rest, Some(metaspace)),
        _ => (chain, None),
    }
}
