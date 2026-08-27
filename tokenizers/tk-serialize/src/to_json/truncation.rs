//! The `truncation` object, written back from the `TruncationParams` the tokenizer cuts with.

use super::writer::Out;
use tk_encode::tokenizer::{TruncationDirection, TruncationParams, TruncationStrategy};

pub(super) fn write_truncation(out: &mut Out, truncation: Option<&TruncationParams>) {
    // A missing `truncation` reads back as `None`, so absent is exact.
    let Some(truncation) = truncation else {
        out.null();
        return;
    };
    out.obj_open();
    // Not `TruncationDirection`'s `AsRef<str>`: that one is lowercase, and the file spells the
    // variant.
    out.field_str(
        "direction",
        match truncation.direction {
            TruncationDirection::Left => "Left",
            TruncationDirection::Right => "Right",
        },
    );
    out.field_usize("max_length", truncation.max_length);
    out.field_str(
        "strategy",
        match truncation.strategy {
            TruncationStrategy::LongestFirst => "LongestFirst",
            TruncationStrategy::OnlyFirst => "OnlyFirst",
            TruncationStrategy::OnlySecond => "OnlySecond",
        },
    );
    // Carried through even though the encode path keeps only the first window, so a config that
    // declares one reads back unchanged.
    out.field_usize("stride", truncation.stride);
    out.obj_close();
}
