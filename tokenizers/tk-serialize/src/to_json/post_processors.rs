//! The `post_processor` object, written back out of the two `Template`s it was lowered into.
use super::writer::Out;
use tk_encode::pipeline::{PipelinePostProcessor, PipelineToken, Template};
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
    single.n_special() == 0
        && single.a_type_id == 0
        && single.b_type_id.is_none()
        && pair.n_special() == 0
        && pair.a_type_id == 0
        && pair.b_type_id == Some(1)
}

/// One template as an array of pieces: `{"seq": "A"}` for a sequence, `{"ids": [...]}` for a run of
/// special tokens, and `type_id` only when it is not the `0` a piece defaults to.
fn write_pieces(out: &mut Out, key: &str, template: &Template) {
    out.key(key);
    out.arr_open();
    write_specials(out, &template.prefix);
    write_seq(out, "A", template.a_type_id);
    write_specials(out, &template.infix);
    if let Some(type_id) = template.b_type_id {
        write_seq(out, "B", type_id);
    }
    write_specials(out, &template.suffix);
    out.arr_close();
}

fn write_seq(out: &mut Out, seq: &str, type_id: u8) {
    out.obj_open();
    out.field_str("seq", seq);
    write_type_id(out, type_id);
    out.obj_close();
}

/// Consecutive specials sharing a `type_id` go back out as the one `{"ids": [...]}` piece they
/// most likely came in as.
///
/// The ids are all the pipeline kept. The strings behind them are in the vocabulary, so a name
/// here would only repeat it -- and a name needs a table.
fn write_specials(out: &mut Out, specials: &[(PipelineToken, u8)]) {
    let mut rest = specials;
    while let Some(&(_, type_id)) = rest.first() {
        let n = rest.iter().take_while(|&&(_, id)| id == type_id).count();
        out.obj_open();
        out.key("ids");
        out.arr_open();
        for &(token, _) in &rest[..n] {
            out.u32(token.id());
        }
        out.arr_close();
        write_type_id(out, type_id);
        out.obj_close();
        rest = &rest[n..];
    }
}

fn write_type_id(out: &mut Out, type_id: u8) {
    if type_id != 0 {
        out.field_u32("type_id", u32::from(type_id));
    }
}
