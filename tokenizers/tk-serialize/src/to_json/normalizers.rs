//! The `normalizer` object, rebuilt from the pipeline's flattened `Vec<PipelineNormalizer>`.
//!
//! The chain went in as a tree and comes out as a list, so it goes back out as one `Sequence` --
//! never the original nesting. The reader flattens a `Sequence` again on the way in, so both forms
//! build the same pipeline; only the file's shape differs, and the flat one is the honest
//! description of what actually runs.

use super::writer::Out;
#[cfg(feature = "normalizers")]
use base64::Engine as _;
use tk_encode::normalizers::replace::{Replace, ReplacePattern};
use tk_encode::pipeline::PipelineNormalizer;
use tk_encode::tokenizer::Result;

/// The `normalizer` value for a chain.
///
/// A `Metaspace` normalizer is *not* one of these: it is half of a `Metaspace` pre-tokenizer, and
/// [`super::pre_tokenizers`] writes it there. The caller splits it off before calling this, so
/// meeting one here is a bug rather than a config this writer cannot handle.
pub(super) fn write_normalizer(out: &mut Out, chain: &[PipelineNormalizer]) -> Result<()> {
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

fn write_one(out: &mut Out, normalizer: &PipelineNormalizer) -> Result<()> {
    /// A field-less normalizer: the tag is the whole of it.
    fn bare(out: &mut Out, tag: &str) {
        out.obj_open();
        out.type_tag(tag);
        out.obj_close();
    }

    match normalizer {
        PipelineNormalizer::Metaspace(_) => {
            return Err(
                "a `Metaspace` normalizer belongs to the pre-tokenizer that produced it, \
                        and cannot be written as a `normalizer`"
                    .into(),
            );
        }
        PipelineNormalizer::Replace(replace) => write_replace(out, replace),
        PipelineNormalizer::Prepend(prepend) => {
            out.obj_open();
            out.type_tag("Prepend");
            out.field_str("prepend", &prepend.prepend);
            out.obj_close();
        }
        PipelineNormalizer::Strip(strip) => {
            out.obj_open();
            out.type_tag("Strip");
            out.field_bool("strip_left", strip.strip_left);
            out.field_bool("strip_right", strip.strip_right);
            out.obj_close();
        }
        PipelineNormalizer::Lowercase(_) => bare(out, "Lowercase"),
        PipelineNormalizer::ByteLevel(_) => bare(out, "ByteLevel"),
        #[cfg(feature = "normalizers")]
        PipelineNormalizer::Bert(bert) => {
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
        PipelineNormalizer::StripAccents(_) => bare(out, "StripAccents"),
        #[cfg(feature = "normalizers")]
        PipelineNormalizer::NFC(_) => bare(out, "NFC"),
        #[cfg(feature = "normalizers")]
        PipelineNormalizer::NFD(_) => bare(out, "NFD"),
        #[cfg(feature = "normalizers")]
        PipelineNormalizer::NFKC(_) => bare(out, "NFKC"),
        #[cfg(feature = "normalizers")]
        PipelineNormalizer::NFKD(_) => bare(out, "NFKD"),
        #[cfg(feature = "normalizers")]
        PipelineNormalizer::Nmt(_) => bare(out, "Nmt"),
        #[cfg(feature = "normalizers")]
        PipelineNormalizer::Precompiled(precompiled) => {
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
    }
    Ok(())
}

/// A `Replace`, whose JSON shape the decoder of the same name shares.
pub(super) fn write_replace(out: &mut Out, replace: &Replace) {
    out.obj_open();
    out.type_tag("Replace");
    write_replace_pattern(out, replace.pattern());
    out.field_str("content", &replace.content);
    out.obj_close();
}

/// `{"String": "..."}` or `{"Regex": "..."}`: externally tagged, as the reader expects.
pub(super) fn write_replace_pattern(out: &mut Out, pattern: &ReplacePattern) {
    out.key("pattern");
    out.obj_open();
    match pattern {
        ReplacePattern::String(s) => out.field_str("String", s),
        ReplacePattern::Regex(r) => out.field_str("Regex", r),
    }
    out.obj_close();
}
