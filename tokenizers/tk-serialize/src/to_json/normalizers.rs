//! The `normalizer` object, rebuilt from the pipeline's flattened `Vec<Normalizer>`.
use super::writer::Out;
#[cfg(feature = "normalizers")]
use base64::Engine as _;
use tk_encode::normalizers::metaspace::PrependScheme;
use tk_encode::normalizers::replace::{Replace, ReplacePattern};
use tk_encode::pipeline::Normalizer;
use tk_encode::tokenizer::Result;

pub(super) fn prepend_scheme_name(scheme: PrependScheme) -> Result<&'static str> {
    match scheme {
        PrependScheme::Always => Ok("always"),
        PrependScheme::First => Ok("first"),
        PrependScheme::Never => Ok("never"),
        other => Err(format!("to_json cannot write the prepend_scheme {other:?}").into()),
    }
}

/// The `normalizer` value for a chain.
pub(super) fn write_normalizer(out: &mut Out, chain: &[Normalizer]) -> Result<()> {
    match chain {
        // No normalizer at all. The reader reads a missing or null `normalizer` as an empty chain,
        // so this is exact rather than an approximation.
        [] => out.null(),
        [one] => write_one(out, one)?,
        many => {
            out.obj_open();
            out.type_tag("Sequence");
            out.key("normalizers");
            out.arr_open();
            for member in many {
                write_one(out, member)?;
            }
            out.arr_close();
            out.obj_close();
        }
    }
    Ok(())
}

fn write_one(out: &mut Out, normalizer: &Normalizer) -> Result<()> {
    /// A field-less normalizer: the tag is the whole of it.
    fn bare(out: &mut Out, tag: &str) {
        out.obj_open();
        out.type_tag(tag);
        out.obj_close();
    }

    match normalizer {
        Normalizer::Metaspace(metaspace) => {
            out.obj_open();
            out.type_tag("MetaspaceNormalizer");
            out.field_str(
                "replacement",
                metaspace.delimiter().encode_utf8(&mut [0; 4]),
            );
            out.field_str("prepend", prepend_scheme_name(metaspace.prepend())?);
            out.field_bool("drop_whitespace", metaspace.drop_whitespace());
            out.obj_close();
        }
        Normalizer::Replace(replace) => write_replace(out, replace)?,
        Normalizer::Prepend(prepend) => {
            out.obj_open();
            out.type_tag("Prepend");
            out.field_str("prepend", &prepend.prepend);
            out.obj_close();
        }
        Normalizer::Strip(strip) => {
            out.obj_open();
            out.type_tag("Strip");
            out.field_bool("strip_left", strip.strip_left);
            out.field_bool("strip_right", strip.strip_right);
            out.obj_close();
        }
        Normalizer::Lowercase(_) => bare(out, "Lowercase"),
        Normalizer::ByteLevel(_) => bare(out, "ByteLevel"),
        #[cfg(feature = "normalizers")]
        Normalizer::Bert(bert) => {
            out.obj_open();
            out.type_tag("BertNormalizer");
            out.field_bool("clean_text", bert.clean_text);
            out.field_bool("handle_chinese_chars", bert.handle_chinese_chars);
            // Required even when it is `null`: `null` is what means "decide from `lowercase`", and
            // the reader demands the key be present for exactly that reason.
            match bert.strip_accents {
                Some(value) => out.field_bool("strip_accents", value),
                None => out.field_null("strip_accents"),
            }
            out.field_bool("lowercase", bert.lowercase);
            out.obj_close();
        }
        #[cfg(feature = "normalizers")]
        Normalizer::StripAccents(_) => bare(out, "StripAccents"),
        #[cfg(feature = "normalizers")]
        Normalizer::NFC(_) => bare(out, "NFC"),
        #[cfg(feature = "normalizers")]
        Normalizer::NFD(_) => bare(out, "NFD"),
        #[cfg(feature = "normalizers")]
        Normalizer::NFKC(_) => bare(out, "NFKC"),
        #[cfg(feature = "normalizers")]
        Normalizer::NFKD(_) => bare(out, "NFKD"),
        #[cfg(feature = "normalizers")]
        Normalizer::Nmt(_) => bare(out, "Nmt"),
        #[cfg(feature = "normalizers")]
        Normalizer::Precompiled(precompiled) => {
            // TODO: this will be fixed for v1!
            // The one normalizer whose configuration is a binary blob, and the one place the
            // pipeline had to be taught to remember something purely so it could be written back:
            // `spm_precompiled` keeps the charsmap private and publishes it only through serde.
            let charsmap = precompiled.charsmap().ok_or_else(|| -> tk_encode::Error {
                "this `Precompiled` normalizer was built from an already-parsed value, so the \
                 `precompiled_charsmap` bytes it came from are gone and cannot be written back"
                    .into()
            })?;
            out.obj_open();
            out.type_tag("Precompiled");
            out.field_str("precompiled_charsmap", &crate::BASE64.encode(charsmap));
            out.obj_close();
        }
        _ => return Err("to_json cannot write this normalizer".into()),
    }
    Ok(())
}

/// A `Replace`, whose JSON shape the decoder of the same name shares.
pub(super) fn write_replace(out: &mut Out, replace: &Replace) -> Result<()> {
    out.obj_open();
    out.type_tag("Replace");
    write_replace_pattern(out, replace.pattern())?;
    out.field_str("content", &replace.content);
    out.obj_close();
    Ok(())
}

/// `{"String": "..."}` or `{"Regex": "..."}`: externally tagged, as the reader expects.
pub(super) fn write_replace_pattern(out: &mut Out, pattern: &ReplacePattern) -> Result<()> {
    out.key("pattern");
    out.obj_open();
    match pattern {
        ReplacePattern::String(s) => out.field_str("String", s),
        ReplacePattern::Regex(r) => out.field_str("Regex", r),
        other => return Err(format!("to_json cannot write the Replace pattern {other:?}").into()),
    }
    out.obj_close();
    Ok(())
}
