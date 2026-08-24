//! The `pre_tokenizer` object, reassembled from the shapes the reader lowered it into.

use super::writer::Out;
use tk_encode::pipeline::PipelinePreTokenizer;
use tk_encode::pre_tokenizers::split::SplitPattern;
use tk_encode::tokenizer::Result;

/// The `pre_tokenizer` value.
pub(super) fn write_pre_tokenizer(out: &mut Out, pretok: &PipelinePreTokenizer) -> Result<()> {
    match pretok {
        // No pre-tokenizer. A missing `pre_tokenizer` reads back as exactly this.
        PipelinePreTokenizer::None => {
            out.null();
            Ok(())
        }
        other => write_one(out, other),
    }
}

fn write_one(out: &mut Out, pretok: &PipelinePreTokenizer) -> Result<()> {
    /// A field-less pre-tokenizer.
    fn bare(out: &mut Out, tag: &str) {
        out.obj_open();
        out.type_tag(tag);
        out.obj_close();
    }

    match pretok {
        // `None` is the whole slot being empty, which the caller writes as `null`. A `Sequence`
        // member cannot be one: nothing lowers to nothing inside a sequence.
        PipelinePreTokenizer::None => {
            return Err("a `Sequence` pre-tokenizer has an empty member".into());
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
