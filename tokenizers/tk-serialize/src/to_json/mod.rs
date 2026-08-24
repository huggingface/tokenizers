//! Write a [`PipelineTokenizer`] back out as a `tokenizer.json`.
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

#[cfg(all(test, feature = "deserialize"))]
mod tests;

use self::added_tokens::write_added_tokens;
use self::decoders::write_decoder;
use self::model::write_model;
use self::normalizers::write_normalizer;
use self::post_processors::write_post_processor;
use self::pre_tokenizers::write_pre_tokenizer;
use self::writer::Out;
use tk_encode::pipeline::{PipelineModel, PipelineTokenizer};
use tk_encode::tokenizer::Result;

/// Write a `tokenizer.json` as a string.
pub fn to_json(tokenizer: &PipelineTokenizer) -> Result<String> {
    let mut out = Out::new();
    let model = tokenizer.get_model();
    let normalizers = tokenizer.get_normalizers();
    let byte_level = match model {
        PipelineModel::BPE(bpe) => bpe.is_byte_level(),
        #[cfg(feature = "unigram")]
        PipelineModel::Unigram(_) => false,
        #[cfg(feature = "wordpiece")]
        PipelineModel::WordPiece(_) => false,
        #[cfg(feature = "wordlevel")]
        PipelineModel::WordLevel(_) => false,
    };

    out.obj_open();
    out.field_str("version", "2.0");
    // TODO: these are REQUIRED for v1
    out.field_null("truncation");
    out.field_null("padding");
    out.key("added_tokens");
    write_added_tokens(&mut out, tokenizer.get_added_vocabulary());
    out.key("normalizer");
    write_normalizer(&mut out, normalizers)?;
    out.key("pre_tokenizer");
    write_pre_tokenizer(&mut out, tokenizer.get_pre_tokenizer(), byte_level)?;
    out.key("post_processor");
    write_post_processor(&mut out, tokenizer.get_post_processor())?;
    out.key("decoder");
    write_decoder(&mut out, tokenizer.get_decoder())?;
    out.key("model");
    write_model(&mut out, model)?;
    out.obj_close();
    Ok(out.finish())
}
