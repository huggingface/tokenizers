//! The `post_processor` object, written back out of the two `Template`s it was lowered into.
use super::writer::Out;
use tk_encode::pipeline::{PipelinePostProcessor, Seq, Slice, Template};
use tk_encode::tokenizer::Result;

pub(super) fn write_post_processor(
    out: &mut Out,
    post_processor: &PipelinePostProcessor,
    name_of: &dyn Fn(u32) -> Option<String>,
) -> Result<()> {
    let (single, pair) = post_processor.templates();
    if is_default(single, pair) {
        out.null();
        return Ok(());
    }

    let mut specials = Specials::default();
    let single_names = specials.name_all(single, name_of);
    let pair_names = specials.name_all(pair, name_of);

    out.obj_open();
    out.type_tag("TemplateProcessing");
    write_pieces(out, "single", single, &single_names);
    write_pieces(out, "pair", pair, &pair_names);
    out.key("special_tokens");
    out.obj_open();
    for entry in &specials.entries {
        out.key(&entry.name);
        out.obj_open();
        out.key("ids");
        out.arr_open();
        for id in &entry.ids {
            out.u32(*id);
        }
        out.arr_close();
        out.obj_close();
    }
    out.obj_close();
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

/// One entry of the `special_tokens` table.
struct SpecialEntry {
    name: String,
    ids: Vec<u32>,
}

/// The `special_tokens` table under construction, which is also what assigns the placeholder names.
#[derive(Default)]
struct Specials {
    entries: Vec<SpecialEntry>,
}

impl Specials {
    /// The placeholder name for each `Specials` slice of `template`, in order, interleaved with
    /// `None` for the sequence slices so the indices line up with the slices.
    fn name_all(
        &mut self,
        template: &Template,
        name_of: &dyn Fn(u32) -> Option<String>,
    ) -> Vec<Option<String>> {
        template
            .slices()
            .iter()
            .map(|slice| match slice {
                Slice::Sequence { .. } => None,
                Slice::Specials { tokens, .. } => {
                    let ids: Vec<u32> = tokens.iter().map(|token| token.id()).collect();
                    Some(self.name_for(&ids, name_of))
                }
            })
            .collect()
    }

    /// The name for a run of ids, reusing the entry when this exact run already has one.
    fn name_for(&mut self, ids: &[u32], name_of: &dyn Fn(u32) -> Option<String>) -> String {
        if let Some(entry) = self.entries.iter().find(|entry| entry.ids == ids) {
            return entry.name.clone();
        }
        // The tokens themselves, and the name is them joined -- which for the single-id case that
        // covers every real config is just the token. An id with no token left (a vocabulary that
        // does not contain it) is named after the number, which still resolves to the same ids.
        let tokens: Vec<String> = ids
            .iter()
            .map(|&id| name_of(id).unwrap_or_else(|| id.to_string()))
            .collect();
        let mut name = tokens.join(" ");
        // Two different runs must not share a name, or the second would resolve to the first's ids.
        // Only reachable when a token string is repeated across runs, which no real config does.
        if self.entries.iter().any(|entry| entry.name == name) {
            let mut suffix = 2;
            while self
                .entries
                .iter()
                .any(|entry| entry.name == format!("{name}#{suffix}"))
            {
                suffix += 1;
            }
            name = format!("{name}#{suffix}");
        }
        self.entries.push(SpecialEntry {
            name: name.clone(),
            ids: ids.to_vec(),
        });
        name
    }
}

/// One template as an array of pieces: `{"Sequence": {...}}` or `{"SpecialToken": {...}}`.
fn write_pieces(out: &mut Out, key: &str, template: &Template, names: &[Option<String>]) {
    out.key(key);
    out.arr_open();
    for (slice, name) in template.slices().iter().zip(names) {
        out.obj_open();
        match slice {
            Slice::Sequence { seq, type_id } => {
                out.key("Sequence");
                out.obj_open();
                out.field_str("id", if matches!(seq, Seq::A) { "A" } else { "B" });
                out.field_u32("type_id", u32::from(*type_id));
                out.obj_close();
            }
            Slice::Specials { type_id, .. } => {
                out.key("SpecialToken");
                out.obj_open();
                // `name_all` produced one name per `Specials` slice, in this order.
                out.field_str("id", name.as_deref().unwrap_or_default());
                out.field_u32("type_id", u32::from(*type_id));
                out.obj_close();
            }
        }
        out.obj_close();
    }
    out.arr_close();
}
