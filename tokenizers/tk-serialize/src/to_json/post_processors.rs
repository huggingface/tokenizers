//! The `post_processor` object, written back out of the two `Template`s it was lowered into.
use super::writer::Out;
use tk_encode::pipeline::{PipelinePostProcessor, Seq, Slice, Template};
use tk_encode::tokenizer::Result;

pub(super) fn write_post_processor(
    out: &mut Out,
    post_processor: &PipelinePostProcessor,
) -> Result<()> {
    let (single, pair) = post_processor.templates();
    if is_default(single, pair) {
        out.null();
        return Ok(());
    }

    out.obj_open();
    out.type_tag("TemplateProcessing");
    write_pieces(out, "single", single);
    write_pieces(out, "pair", pair);
    out.obj_close();
    Ok(())
}

/// Whether these two templates are the frame that does nothing: `$A`, and `$A $B` with the default
/// type ids. That is [`PipelinePostProcessor::default`], which is what a missing `post_processor`
/// and a `ByteLevel` both produce.
fn is_default(single: &Template, pair: &Template) -> bool {
    matches!(
        single.slices(),
        [Slice::Sequence {
            seq: Seq::A,
            type_id: 0
        }]
    ) && matches!(
        pair.slices(),
        [
            Slice::Sequence {
                seq: Seq::A,
                type_id: 0
            },
            Slice::Sequence {
                seq: Seq::B,
                type_id: 1
            }
        ]
    )
}

/// One template as an array of pieces: `{"Sequence": {...}}` or `{"SpecialToken": {...}}`.
fn write_pieces(out: &mut Out, key: &str, template: &Template) {
    out.key(key);
    out.arr_open();
    for slice in template.slices() {
        out.obj_open();
        match slice {
            Slice::Sequence { seq, type_id } => {
                out.key("Sequence");
                out.obj_open();
                out.field_str("id", if matches!(seq, Seq::A) { "A" } else { "B" });
                out.field_u32("type_id", u32::from(*type_id));
                out.obj_close();
            }
            Slice::Specials { tokens, type_id } => {
                out.key("SpecialToken");
                out.obj_open();
                // The ids themselves. A name would only be a key into a table of these, and the
                // strings behind them are in the vocabulary already.
                out.key("ids");
                out.arr_open();
                for token in tokens.iter() {
                    out.u32(token.id());
                }
                out.arr_close();
                out.field_u32("type_id", u32::from(*type_id));
                out.obj_close();
            }
        }
        out.obj_close();
    }
    out.arr_close();
}
