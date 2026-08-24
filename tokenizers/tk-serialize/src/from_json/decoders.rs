//! The `decoder` object, built straight into a `DecoderRuntime`.

use super::normalizers::read_replace_fields;
use super::pre_tokenizers::read_char;
use super::unsupported;
use crate::json::Json;
use tk_encode::decoders::DecoderRuntime;
use tk_encode::decoders::bpe::BPEDecoder;
use tk_encode::decoders::byte_fallback::ByteFallback;
use tk_encode::decoders::byte_level::ByteLevelDecoder;
use tk_encode::decoders::ctc::CTC;
use tk_encode::decoders::fuse::Fuse;
use tk_encode::decoders::metaspace::MetaspaceDecoder;
use tk_encode::decoders::metaspace::PrependScheme;
use tk_encode::decoders::replace::ReplaceDecoder;
use tk_encode::decoders::strip::Strip as StripDecoder;
use tk_encode::decoders::wordpiece::WordPiece as WordPieceDecoder;
use tk_encode::tokenizer::Result;

pub(super) fn read_replace_decoder(cfg: &Json<'_>) -> Result<ReplaceDecoder> {
    let (pattern, content) = read_replace_fields(cfg)?;
    ReplaceDecoder::new(pattern, content)
}

/// The decoder, as a [`DecoderRuntime`].
pub(super) fn read_decoder(cfg: Option<&Json<'_>>) -> Result<Option<DecoderRuntime>> {
    match cfg {
        Some(cfg) => Ok(Some(read_one_decoder(cfg)?)),
        None => Ok(None),
    }
}

/// The `Metaspace` decoder's `prepend_scheme`. Spelled out, always: `add_prefix_space` is the
/// legacy spelling of it and tk-convert rewrites that before this reader ever sees the file.
fn read_prepend_scheme(cfg: &Json<'_>) -> Result<PrependScheme> {
    match cfg.need("the `Metaspace` decoder", "prepend_scheme", Json::as_str)? {
        "always" => Ok(PrependScheme::Always),
        "first" => Ok(PrependScheme::First),
        "never" => Ok(PrependScheme::Never),
        other => Err(format!("unknown metaspace prepend_scheme {other:?}").into()),
    }
}

pub(super) fn read_one_decoder(cfg: &Json<'_>) -> Result<DecoderRuntime> {
    let kind = cfg
        .type_tag()
        .ok_or_else(|| unsupported("a decoder with no `type`"))?;
    let owner = format!("the `{kind}` decoder");

    Ok(match kind {
        "ByteLevel" => DecoderRuntime::ByteLevel(ByteLevelDecoder::new()),
        "Replace" => DecoderRuntime::Replace(read_replace_decoder(cfg)?),
        "ByteFallback" => DecoderRuntime::ByteFallback(ByteFallback::new()),
        "Fuse" => DecoderRuntime::Fuse(Fuse::new()),
        "Strip" => DecoderRuntime::Strip(StripDecoder::new(
            read_char(cfg, "content")?,
            cfg.need(&owner, "start", Json::as_usize)?,
            cfg.need(&owner, "stop", Json::as_usize)?,
        )),
        // Spelled `BPEDecoder` in the file, unlike every other tag, which matches its type name.
        "BPEDecoder" => DecoderRuntime::BPE(BPEDecoder::new(
            cfg.need(&owner, "suffix", Json::as_str)?.to_string(),
        )),
        "WordPiece" => DecoderRuntime::WordPiece(WordPieceDecoder::new(
            cfg.need(&owner, "prefix", Json::as_str)?.to_string(),
            cfg.need(&owner, "cleanup", Json::as_bool)?,
        )),
        // `split` is ignored: it says how the *pre-tokenizer* cut the text, and decoding never
        // looks at it.
        "Metaspace" => DecoderRuntime::Metaspace(MetaspaceDecoder::new(
            read_char(cfg, "replacement")?,
            read_prepend_scheme(cfg)?,
        )),
        "CTC" => DecoderRuntime::CTC(CTC::new(
            cfg.need(&owner, "pad_token", Json::as_str)?.to_string(),
            cfg.need(&owner, "word_delimiter_token", Json::as_str)?
                .to_string(),
            cfg.need(&owner, "cleanup", Json::as_bool)?,
        )),
        "Sequence" => {
            let members = cfg
                .field("decoders")
                .and_then(Json::as_array)
                .unwrap_or(&[]);
            DecoderRuntime::Sequence(
                members
                    .iter()
                    .map(read_one_decoder)
                    .collect::<Result<Vec<_>>>()?,
            )
        }
        other => return Err(unsupported(&format!("the `{other}` decoder"))),
    })
}
