//! The `pre_tokenizer` object, reassembled from the shapes the reader lowered it into.
//!
//! This is the component the pipeline describes least directly, because two of the pre-tokenizers a
//! `tokenizer.json` can name do not survive as themselves:
//!
//! * a **`Metaspace`** became a [`MetaspaceNormalizer`] plus a `Split` on the delimiter, so it has
//!   to be recognised from the normalizer chain and written back with the `Split` dropped;
//! * a **`ByteLevel`** became a `Split` on the GPT-2 regex (`use_regex: true`) or nothing at all
//!   (`use_regex: false`), so which one it was is read off the model -- a byte-level `PipelineBPE`
//!   is the only thing that flag produces.
//!
//! Both inversions are decidable, which is the only reason this works: the reader puts the
//! `Metaspace` normalizer last and requires a `ByteLevel` to be the last member of a `Sequence`, so
//! there is exactly one place to look in each case.
//!
//! What is *not* recovered is `trim_offsets`, on either `ByteLevel`. The pipeline tracks no offsets,
//! so the flag never reached it; the canonical default is written instead. It cannot move an id.

use super::writer::Out;
use tk_encode::normalizers::metaspace::MetaspaceNormalizer;
use tk_encode::pipeline::PipelinePreTokenizer;
use tk_encode::pre_tokenizers::split::{Split as SplitPretok, SplitPattern};
use tk_encode::tokenizer::{Result, SplitDelimiterBehavior};

/// The `pre_tokenizer` value.
///
/// `metaspace` is the normalizer [`super::split_off_metaspace`] took off the end of the chain, and
/// `byte_level` is whether the model applies the byte-level map. Between them they say which of the
/// two rewritten forms `pretok` is, so this never has to guess.
pub(super) fn write_pre_tokenizer(
    out: &mut Out,
    pretok: &PipelinePreTokenizer,
    metaspace: Option<&MetaspaceNormalizer>,
    byte_level: bool,
) -> Result<()> {
    // A `Metaspace` outranks everything: the pre-tokenizer slot holds the `Split` it produced, and
    // that `Split` is not a component of its own.
    if let Some(metaspace) = metaspace {
        if byte_level {
            return Err(
                "a `Metaspace` pre-tokenizer and a byte-level model cannot both be written: the \
                 reader accepts a `ByteLevel` only with BPE and never alongside a `Metaspace`"
                    .into(),
            );
        }
        return write_metaspace(out, metaspace);
    }

    if byte_level {
        return write_byte_level_form(out, pretok);
    }

    match pretok {
        // No pre-tokenizer. A missing `pre_tokenizer` reads back as exactly this.
        PipelinePreTokenizer::None => {
            out.null();
            Ok(())
        }
        other => write_one(out, other),
    }
}

/// The `Metaspace` family: on its own, or behind the `WhitespaceSplit` that `drop_whitespace` means.
fn write_metaspace(out: &mut Out, metaspace: &MetaspaceNormalizer) -> Result<()> {
    let metaspace_obj = |out: &mut Out| {
        out.obj_open();
        out.type_tag("Metaspace");
        out.field_str(
            "replacement",
            metaspace.delimiter().encode_utf8(&mut [0; 4]),
        );
        // Canonical spelling, never the `add_prefix_space` the old files use. `first` cannot occur:
        // the reader refuses it, so nothing in a pipeline came from one.
        out.field_str(
            "prepend_scheme",
            if metaspace.prepend() {
                "always"
            } else {
                "never"
            },
        );
        // Always `true`: `split: false` is refused on the way in, because a rewrite with no cut
        // cannot be expressed as a pre-tokenizer at all.
        out.field_bool("split", true);
        out.obj_close();
    };

    if metaspace.drop_whitespace() {
        // The pair t5 and albert ship. The reader recognises exactly this shape and collapses it
        // again, which is why it has to go back out as a `Sequence` rather than a lone `Metaspace`.
        out.obj_open();
        out.type_tag("Sequence");
        out.key("pretokenizers");
        out.arr_open();
        out.obj_open();
        out.type_tag("WhitespaceSplit");
        out.obj_close();
        metaspace_obj(out);
        out.arr_close();
        out.obj_close();
    } else {
        metaspace_obj(out);
    }
    Ok(())
}

/// A `ByteLevel` in the position the reader found it: on its own, or last in a `Sequence`.
fn write_byte_level_form(out: &mut Out, pretok: &PipelinePreTokenizer) -> Result<()> {
    match pretok {
        PipelinePreTokenizer::Sequence(sequence) => {
            let members = sequence.pre_tokenizers();
            let (last, rest) = members.split_last().ok_or_else(|| -> tk_encode::Error {
                "a byte-level model with an empty pre-tokenizer `Sequence`".into()
            })?;
            out.obj_open();
            out.type_tag("Sequence");
            out.key("pretokenizers");
            out.arr_open();
            for member in rest {
                write_one(out, member)?;
            }
            write_byte_level(out, last)?;
            out.arr_close();
            out.obj_close();
        }
        lone => write_byte_level(out, lone)?,
    }
    Ok(())
}

/// The `ByteLevel` object, from whichever of its two lowerings is in hand.
fn write_byte_level(out: &mut Out, lowered: &PipelinePreTokenizer) -> Result<()> {
    let use_regex = match lowered {
        // `use_regex: false` asks only for the byte map, which the model half already applies, so
        // the splitting step became the identity.
        PipelinePreTokenizer::None => false,
        // `use_regex: true` splits on the GPT-2 regex, which is what the `Split` carries.
        PipelinePreTokenizer::Split(split) if is_gpt2_regex(split) => true,
        other => {
            return Err(format!(
                "a byte-level model's last pre-tokenizer is {other:?}, which is neither of the two \
                 forms a `ByteLevel` lowers to (a `Split` on the GPT-2 regex, or nothing)"
            )
            .into());
        }
    };
    out.obj_open();
    out.type_tag("ByteLevel");
    // Never `true`: the reader refuses `add_prefix_space: true` outright, so no pipeline it built
    // came from one.
    out.field_bool("add_prefix_space", false);
    // Not represented in the pipeline -- offsets are not tracked -- so the canonical default.
    out.field_bool("trim_offsets", true);
    out.field_bool("use_regex", use_regex);
    out.obj_close();
    Ok(())
}

/// Whether this `Split` is the one a `use_regex` `ByteLevel` lowers to.
fn is_gpt2_regex(split: &SplitPretok) -> bool {
    matches!(&split.pattern, SplitPattern::Regex(r) if r == atomsplit::regexes::GPT2)
        && split.behavior == SplitDelimiterBehavior::Isolated
        && !split.invert
}

fn write_one(out: &mut Out, pretok: &PipelinePreTokenizer) -> Result<()> {
    /// A field-less pre-tokenizer.
    fn bare(out: &mut Out, tag: &str) {
        out.obj_open();
        out.type_tag(tag);
        out.obj_close();
    }

    match pretok {
        // Only a `ByteLevel` with `use_regex: false` lowers to nothing, and that is handled by
        // `write_byte_level_form` -- reaching here means the model does not think it is byte-level.
        PipelinePreTokenizer::None => {
            return Err(
                "a pre-tokenizer lowered to nothing, which only a `ByteLevel` with \
                        `use_regex: false` does -- but the model is not byte-level, so there is no \
                        `ByteLevel` to write"
                    .into(),
            );
        }
        PipelinePreTokenizer::Split(split) => {
            out.obj_open();
            out.type_tag("Split");
            out.key("pattern");
            out.obj_open();
            match &split.pattern {
                SplitPattern::String(s) => out.field_str("String", s),
                SplitPattern::Regex(r) => out.field_str("Regex", r),
            }
            out.obj_close();
            // `Display` rather than a second table of names; `display_matches_serde` in `tk-encode`
            // pins it against the serialized spelling.
            out.field_str("behavior", &split.behavior.to_string());
            out.field_bool("invert", split.invert);
            out.obj_close();
        }
        PipelinePreTokenizer::Whitespace(_) => bare(out, "Whitespace"),
        PipelinePreTokenizer::WhitespaceSplit(_) => bare(out, "WhitespaceSplit"),
        PipelinePreTokenizer::Bert(_) => bare(out, "BertPreTokenizer"),
        #[cfg(feature = "unicode-scripts")]
        PipelinePreTokenizer::UnicodeScripts(_) => bare(out, "UnicodeScripts"),
        PipelinePreTokenizer::Digits(digits) => {
            out.obj_open();
            out.type_tag("Digits");
            out.field_bool("individual_digits", digits.individual_digits);
            out.obj_close();
        }
        PipelinePreTokenizer::Punctuation(punctuation) => {
            out.obj_open();
            out.type_tag("Punctuation");
            out.field_str("behavior", &punctuation.behavior.to_string());
            out.obj_close();
        }
        PipelinePreTokenizer::Delimiter(delimiter) => {
            out.obj_open();
            out.type_tag("CharDelimiterSplit");
            out.field_str("delimiter", delimiter.delimiter.encode_utf8(&mut [0; 4]));
            out.obj_close();
        }
        PipelinePreTokenizer::FixedLength(fixed) => {
            out.obj_open();
            out.type_tag("FixedLength");
            out.field_usize("length", fixed.length);
            out.obj_close();
        }
        PipelinePreTokenizer::Sequence(sequence) => {
            out.obj_open();
            out.type_tag("Sequence");
            out.key("pretokenizers");
            out.arr_open();
            for member in sequence.pre_tokenizers() {
                write_one(out, member)?;
            }
            out.arr_close();
            out.obj_close();
        }
    }
    Ok(())
}
