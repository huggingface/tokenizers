//! The `padding` object, written back from the `PaddingParams` the tokenizer pads with.

use super::writer::Out;
use tk_encode::tokenizer::{PaddingDirection, PaddingParams, PaddingStrategy};

pub(super) fn write_padding(out: &mut Out, padding: Option<&PaddingParams>) {
    // A missing `padding` reads back as `None`, so absent is exact.
    let Some(padding) = padding else {
        out.null();
        return;
    };
    out.obj_open();
    out.key("strategy");
    match padding.strategy {
        PaddingStrategy::BatchLongest => out.str("BatchLongest"),
        PaddingStrategy::Fixed(size) => {
            out.obj_open();
            out.field_usize("Fixed", size);
            out.obj_close();
        }
    }
    // Not `PaddingDirection`'s `AsRef<str>`: that one is lowercase, and the file spells the
    // variant.
    out.field_str(
        "direction",
        match padding.direction {
            PaddingDirection::Left => "Left",
            PaddingDirection::Right => "Right",
        },
    );
    out.key("pad_to_multiple_of");
    match padding.pad_to_multiple_of {
        Some(multiple) => out.usize(multiple),
        None => out.null(),
    }
    out.field_u32("pad_id", padding.pad_id);
    out.field_u32("pad_type_id", padding.pad_type_id);
    out.field_str("pad_token", &padding.pad_token);
    out.obj_close();
}
