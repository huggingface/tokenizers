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
mod padding;
mod post_processors;
mod pre_tokenizers;
mod writer;

#[cfg(all(test, feature = "deserialize"))]
mod tests;

use self::added_tokens::write_added_tokens;
use self::decoders::write_decoder;
use self::model::write_model;
use self::normalizers::write_normalizer;
use self::padding::write_padding;
use self::post_processors::write_post_processor;
use self::pre_tokenizers::write_pre_tokenizer;
use self::writer::Out;
use tk_encode::pipeline::PipelineTokenizer;
use tk_encode::tokenizer::Result;

/// Write a `tokenizer.json` as a string.
pub fn to_json(tokenizer: &PipelineTokenizer) -> Result<String> {
    let mut out = Out::new();
    let model = tokenizer.get_model();
    let normalizers = tokenizer.get_normalizers();
    out.obj_open();
    out.field_str("version", "2.0");
    // TODO: this is REQUIRED for v1
    out.field_null("truncation");
    out.key("padding");
    write_padding(&mut out, tokenizer.get_padding());
    // Which token plays which role, so this file can stand in for a `tokenizer_config.json`.
    out.key("role_to_token");
    let roles = tokenizer.get_role_to_token();
    if roles.is_empty() {
        out.null();
    } else {
        out.obj_open();
        for (role, token) in roles {
            out.field_str(role, token);
        }
        out.obj_close();
    }
    out.key("added_tokens");
    write_added_tokens(&mut out, tokenizer.get_added_vocabulary());
    out.key("normalizer");
    write_normalizer(&mut out, normalizers)?;
    out.key("pre_tokenizer");
    write_pre_tokenizer(&mut out, tokenizer.get_pre_tokenizer())?;
    out.key("post_processor");
    write_post_processor(&mut out, tokenizer.get_post_processor())?;
    out.key("decoder");
    write_decoder(&mut out, tokenizer.get_decoder())?;
    out.key("model");
    write_model(&mut out, model)?;
    out.obj_close();
    Ok(out.finish())
}
