use crate::tokenizer::Result;
use crate::tokenizer::pipeline::{Encoding, PipelineToken};

/// The various possible padding directions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PaddingDirection {
    Left,
    Right,
}

impl std::convert::AsRef<str> for PaddingDirection {
    fn as_ref(&self) -> &str {
        match self {
            PaddingDirection::Left => "left",
            PaddingDirection::Right => "right",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PaddingParams {
    pub strategy: PaddingStrategy,
    pub direction: PaddingDirection,
    pub pad_to_multiple_of: Option<usize>,
    pub pad_id: u32,
    pub pad_type_id: u32,
    pub pad_token: String,
}

impl Default for PaddingParams {
    fn default() -> Self {
        Self {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id: 0,
            pad_type_id: 0,
            pad_token: String::from("[PAD]"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PaddingStrategy {
    BatchLongest,
    Fixed(usize),
}

/// Pads every encoding in `encodings`, and every overflowing window inside each, to the same
/// length. `BatchLongest` measures the encodings themselves, not their windows: a window is never
/// longer than the encoding it was cut from.
pub fn pad_encodings(encodings: &mut [Encoding], params: &PaddingParams) -> Result<()> {
    if encodings.is_empty() {
        return Ok(());
    }
    let pad_type_id = pad_type_id(params)?;

    let pad_length = round_up(
        match params.strategy {
            PaddingStrategy::Fixed(size) => size,
            PaddingStrategy::BatchLongest => encodings.iter().map(Encoding::len).max().unwrap(),
        },
        params,
    );

    encodings
        .iter_mut()
        .for_each(|encoding| pad_one(encoding, pad_length, params, pad_type_id));

    Ok(())
}

/// Pads one encoding (and its windows) to `params`' fixed length. `BatchLongest` needs the whole
/// batch, so it is not a length here and the encoding is returned untouched.
pub fn pad_encoding(encoding: &mut Encoding, params: &PaddingParams) -> Result<()> {
    let PaddingStrategy::Fixed(size) = params.strategy else {
        return Ok(());
    };
    let pad_type_id = pad_type_id(params)?;
    pad_one(encoding, round_up(size, params), params, pad_type_id);
    Ok(())
}

/// Pads the ids appended at `out[start..]`, treating them as a batch of one: `BatchLongest` is
/// their own length, so it only ever adds what `pad_to_multiple_of` asks for. For a caller that
/// fills a bare id buffer and has no `Encoding` to pad.
pub fn pad_ids(out: &mut Vec<PipelineToken>, start: usize, params: &PaddingParams) {
    let len = out.len() - start;
    let target = round_up(
        match params.strategy {
            PaddingStrategy::Fixed(size) => size,
            PaddingStrategy::BatchLongest => len,
        },
        params,
    );
    if len >= target {
        return;
    }
    let pad_id = PipelineToken::from(params.pad_id);
    match params.direction {
        PaddingDirection::Left => {
            out.splice(start..start, std::iter::repeat_n(pad_id, target - len));
        }
        PaddingDirection::Right => out.resize(start + target, pad_id),
    }
}

/// `length` rounded up to `pad_to_multiple_of`, when there is one.
fn round_up(length: usize, params: &PaddingParams) -> usize {
    match params.pad_to_multiple_of {
        Some(multiple) if multiple > 0 && !length.is_multiple_of(multiple) => {
            length + multiple - length % multiple
        }
        _ => length,
    }
}

/// Type ids are stored as bytes, so a `pad_type_id` past 255 cannot be written; refuse it
/// rather than wrap it to a different type id.
fn pad_type_id(params: &PaddingParams) -> Result<u8> {
    u8::try_from(params.pad_type_id).map_err(|_| {
        format!(
            "padding: `pad_type_id` {} does not fit a type id (the largest is {})",
            params.pad_type_id,
            u8::MAX
        )
        .into()
    })
}

fn pad_one(encoding: &mut Encoding, target_length: usize, params: &PaddingParams, pad_type_id: u8) {
    for window in &mut encoding.overflowing {
        pad_one(window, target_length, params, pad_type_id);
    }
    let original_len = encoding.ids.len();
    if original_len >= target_length {
        return;
    }
    let pad_length = target_length - original_len;
    let pad_id = PipelineToken::from(params.pad_id);

    let type_ids = encoding
        .type_ids
        .get_or_insert_with(|| vec![0; original_len]);
    let attention_mask = encoding
        .attention_mask
        .get_or_insert_with(|| vec![1; original_len]);

    match params.direction {
        PaddingDirection::Left => {
            // `splice` at the front is one reserve plus one memmove per buffer, where rebuilding
            // each buffer through `chain().collect()` was an allocation and a copy.
            encoding
                .ids
                .splice(0..0, std::iter::repeat_n(pad_id, pad_length));
            type_ids.splice(0..0, std::iter::repeat_n(pad_type_id, pad_length));
            attention_mask.splice(0..0, std::iter::repeat_n(0, pad_length));
            encoding.layout.pad_left += pad_length as u32;
        }
        PaddingDirection::Right => {
            encoding.ids.resize(target_length, pad_id);
            type_ids.resize(target_length, pad_type_id);
            attention_mask.resize(target_length, 0);
            encoding.layout.pad_right += pad_length as u32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::pipeline::Layout;

    fn make_tokens(ids: impl IntoIterator<Item = u32>) -> Vec<PipelineToken> {
        ids.into_iter().map(PipelineToken::from).collect()
    }

    /// A bare single-sequence encoding: every id belongs to sequence A.
    fn make_encoding(ids: impl IntoIterator<Item = u32>) -> Encoding {
        let ids = make_tokens(ids);
        let layout = Layout {
            a: ids.len() as u32,
            n_sequences: 1,
            ..Layout::default()
        };
        Encoding {
            ids,
            type_ids: None,
            attention_mask: None,
            layout,
            overflowing: Vec::new(),
        }
    }

    fn with_type_ids(ids: impl IntoIterator<Item = u32>, type_ids: Vec<u8>) -> Encoding {
        Encoding {
            type_ids: Some(type_ids),
            ..make_encoding(ids)
        }
    }

    #[test]
    fn test_multiple_of() {
        fn get_encodings() -> [Encoding; 2] {
            [make_encoding(0..5), make_encoding(0..3)]
        }

        // Test fixed
        let mut encodings = get_encodings();
        let mut params = PaddingParams {
            strategy: PaddingStrategy::Fixed(7),
            direction: PaddingDirection::Right,
            pad_to_multiple_of: Some(8),
            pad_id: 0,
            pad_type_id: 0,
            pad_token: String::from("[PAD]"),
        };
        pad_encodings(&mut encodings, &params).unwrap();
        assert!(encodings.iter().all(|e| e.len() == 8));

        // Test batch
        let mut encodings = get_encodings();
        params.strategy = PaddingStrategy::BatchLongest;
        params.pad_to_multiple_of = Some(6);
        pad_encodings(&mut encodings, &params).unwrap();
        assert!(encodings.iter().all(|e| e.len() == 6));

        // Do not crash with 0
        params.pad_to_multiple_of = Some(0);
        pad_encodings(&mut encodings, &params).unwrap();
    }

    #[test]
    fn test_noop() {
        let mut encodings = [make_encoding(0..5)];
        let params = PaddingParams {
            strategy: PaddingStrategy::Fixed(5),
            ..PaddingParams::default()
        };

        pad_encodings(&mut encodings, &params).unwrap();

        assert_eq!(encodings[0].ids(), make_tokens(0..5));
        assert!(encodings[0].attention_mask().is_none());
    }

    #[test]
    fn test_pad_right() {
        let mut encodings = [make_encoding(0..3)];
        let params = PaddingParams {
            strategy: PaddingStrategy::Fixed(5),
            direction: PaddingDirection::Right,
            pad_id: 99,
            ..PaddingParams::default()
        };

        pad_encodings(&mut encodings, &params).unwrap();

        assert_eq!(encodings[0].ids(), make_tokens([0, 1, 2, 99, 99]));
        assert_eq!(encodings[0].attention_mask().unwrap(), [1, 1, 1, 0, 0]);
    }

    #[test]
    fn test_pad_left() {
        let mut encodings = [make_encoding(0..3)];
        let params = PaddingParams {
            strategy: PaddingStrategy::Fixed(5),
            direction: PaddingDirection::Left,
            pad_id: 99,
            ..PaddingParams::default()
        };

        pad_encodings(&mut encodings, &params).unwrap();

        assert_eq!(encodings[0].ids(), make_tokens([99, 99, 0, 1, 2]));
        assert_eq!(encodings[0].attention_mask().unwrap(), [0, 0, 1, 1, 1]);
    }

    #[test]
    fn test_type_ids_right() {
        let mut encodings = [with_type_ids(0..3, vec![1, 1, 1])];
        let params = PaddingParams {
            strategy: PaddingStrategy::Fixed(5),
            direction: PaddingDirection::Right,
            pad_type_id: 7,
            ..PaddingParams::default()
        };

        pad_encodings(&mut encodings, &params).unwrap();

        assert_eq!(encodings[0].type_ids().unwrap(), [1, 1, 1, 7, 7]);
    }

    #[test]
    fn test_type_ids_left() {
        let mut encodings = [with_type_ids(0..3, vec![1, 1, 1])];
        let params = PaddingParams {
            strategy: PaddingStrategy::Fixed(5),
            direction: PaddingDirection::Left,
            pad_type_id: 7,
            ..PaddingParams::default()
        };

        pad_encodings(&mut encodings, &params).unwrap();

        assert_eq!(encodings[0].type_ids().unwrap(), [7, 7, 1, 1, 1]);
    }

    #[test]
    fn test_type_ids_missing() {
        let mut encodings = [make_encoding(0..3)];
        let params = PaddingParams {
            strategy: PaddingStrategy::Fixed(5),
            direction: PaddingDirection::Right,
            pad_type_id: 7,
            ..PaddingParams::default()
        };

        pad_encodings(&mut encodings, &params).unwrap();

        assert_eq!(encodings[0].type_ids().unwrap(), [0, 0, 0, 7, 7]);
    }

    // Padding is bookkept in the layout, so the sequence still resolves to its tokens after the
    // ids moved, on either side. The overflowing windows are padded like the encoding itself.
    #[test]
    fn test_layout_and_windows_follow_the_padding() {
        let mut encoding = make_encoding(0..3);
        encoding.overflowing = vec![make_encoding(10..12)];
        let params = PaddingParams {
            strategy: PaddingStrategy::Fixed(5),
            direction: PaddingDirection::Left,
            pad_id: 99,
            ..PaddingParams::default()
        };
        pad_encoding(&mut encoding, &params).unwrap();

        assert_eq!(encoding.sequence_range(0), Some(2..5));
        assert_eq!(encoding.special_tokens_mask(), [1, 1, 0, 0, 0]);
        let window = &encoding.overflowing()[0];
        assert_eq!(window.ids(), make_tokens([99, 99, 99, 10, 11]));
        assert_eq!(window.attention_mask().unwrap(), [0, 0, 0, 1, 1]);
        assert_eq!(window.sequence_range(0), Some(3..5));

        // `pad_encoding` cannot size `BatchLongest` on its own: untouched, not padded to 0.
        let mut encoding = make_encoding(0..3);
        pad_encoding(&mut encoding, &PaddingParams::default()).unwrap();
        assert_eq!(encoding.len(), 3);
        assert!(encoding.attention_mask().is_none());
    }

    // A `pad_type_id` that does not fit the byte-sized type ids is refused, not wrapped.
    #[test]
    fn test_pad_type_id_must_fit_a_byte() {
        let mut encodings = [make_encoding(0..3)];
        let params = PaddingParams {
            strategy: PaddingStrategy::Fixed(5),
            pad_type_id: 256,
            ..PaddingParams::default()
        };
        let err = pad_encodings(&mut encodings, &params).unwrap_err();
        assert!(err.to_string().contains("pad_type_id"));
        assert_eq!(encodings[0].len(), 3, "refused before touching anything");
    }
}
