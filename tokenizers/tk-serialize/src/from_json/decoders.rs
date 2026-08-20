//! The `decoder` object, built straight into a `DecoderRuntime`.

use super::normalizers::read_replace_fields;
use super::pre_tokenizers::{read_char, read_prepend_scheme};
use super::unsupported;
use crate::json::{Json, JsonExt};
use tk_encode::decoders::DecoderRuntime;
use tk_encode::decoders::bpe::BPEDecoder;
use tk_encode::decoders::byte_fallback::ByteFallback;
use tk_encode::decoders::byte_level::ByteLevelDecoder;
use tk_encode::decoders::ctc::CTC;
use tk_encode::decoders::fuse::Fuse;
use tk_encode::decoders::metaspace::MetaspaceDecoder;
use tk_encode::decoders::replace::ReplaceDecoder;
use tk_encode::decoders::strip::Strip as StripDecoder;
use tk_encode::decoders::wordpiece::WordPiece as WordPieceDecoder;
use tk_encode::tokenizer::Result;

pub(super) fn read_replace_decoder(cfg: &Json<'_>) -> Result<ReplaceDecoder> {
    let (pattern, content) = read_replace_fields(cfg)?;
    ReplaceDecoder::new(pattern, content)
}

/// The decoder, as a [`DecoderRuntime`].
///
/// The wrapper is fine here where a `ModelWrapper` would not be: only its `Deserialize` impl pulls
/// serde in, and building the enum by hand does not name it. Decode is not on the encode hot path
/// and every variant is small, so there is nothing to gain from a second, pipeline-native enum.
pub(super) fn read_decoder(cfg: Option<&Json<'_>>) -> Result<Option<DecoderRuntime>> {
    match cfg {
        Some(cfg) => Ok(Some(read_one_decoder(cfg)?)),
        None => Ok(None),
    }
}

pub(super) fn read_one_decoder(cfg: &Json<'_>) -> Result<DecoderRuntime> {
    let kind = cfg
        .type_tag()
        .ok_or_else(|| unsupported("a decoder with no `type`"))?;
    let flag = |name: &str| -> Result<bool> {
        cfg.get_some(name)
            .and_then(Json::as_bool)
            .ok_or_else(|| format!("the `{kind}` decoder has no `{name}`").into())
    };
    let text = |name: &str| -> Result<String> {
        cfg.get_some(name)
            .and_then(Json::as_str)
            .map(str::to_string)
            .ok_or_else(|| format!("the `{kind}` decoder has no `{name}`").into())
    };
    let count = |name: &str| -> Result<usize> {
        cfg.get_some(name)
            .and_then(Json::as_usize)
            .ok_or_else(|| format!("the `{kind}` decoder has no `{name}`").into())
    };

    Ok(match kind {
        "ByteLevel" => DecoderRuntime::ByteLevel(ByteLevelDecoder::new(
            flag("add_prefix_space")?,
            flag("trim_offsets")?,
            // The one decoder field with a serde default, and it is `true`.
            cfg.get_some("use_regex")
                .and_then(Json::as_bool)
                .unwrap_or(true),
        )),
        "Replace" => DecoderRuntime::Replace(read_replace_decoder(cfg)?),
        "ByteFallback" => DecoderRuntime::ByteFallback(ByteFallback::new()),
        "Fuse" => DecoderRuntime::Fuse(Fuse::new()),
        "Strip" => DecoderRuntime::Strip(StripDecoder::new(
            read_char(cfg, "content")?,
            count("start")?,
            count("stop")?,
        )),
        // Spelled `BPEDecoder` in the file, unlike every other tag, which matches its type name.
        "BPEDecoder" => DecoderRuntime::BPE(BPEDecoder::new(text("suffix")?)),
        "WordPiece" => {
            DecoderRuntime::WordPiece(WordPieceDecoder::new(text("prefix")?, flag("cleanup")?))
        }
        "Metaspace" => DecoderRuntime::Metaspace(MetaspaceDecoder::new(
            read_char(cfg, "replacement")?,
            read_prepend_scheme(cfg)?,
            cfg.get_some("split")
                .and_then(Json::as_bool)
                .unwrap_or(true),
        )),
        "CTC" => DecoderRuntime::CTC(CTC::new(
            text("pad_token")?,
            text("word_delimiter_token")?,
            flag("cleanup")?,
        )),
        "Sequence" => {
            let members = cfg
                .get_some("decoders")
                .and_then(Json::as_arr)
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
