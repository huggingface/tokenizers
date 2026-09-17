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

/// The width rows pad to: the strategy's target, rounded up to `pad_to_multiple_of`.
fn pad_length(longest: usize, params: &PaddingParams) -> usize {
    let mut length = match params.strategy {
        PaddingStrategy::Fixed(size) => size,
        PaddingStrategy::BatchLongest => longest,
    };
    if let Some(multiple) = params.pad_to_multiple_of
        && multiple > 0
        && length % multiple > 0
    {
        length += multiple - length % multiple;
    }
    length
}

pub fn pad_encodings(encodings: &mut [Encoding], params: &PaddingParams) -> Result<()> {
    let Some(longest) = encodings.iter().map(Encoding::len).max() else {
        return Ok(());
    };
    let length = pad_length(longest, params);
    encodings
        .iter_mut()
        .for_each(|encoding| pad_one(encoding, length, params));
    Ok(())
}

/// Pad a flat batch to one uniform row width, in place.
///
/// The offsets already say how long every row is, so this is a fill with the pad id and a copy of
/// each row into its slot: no reallocation per document, and the result is a dense
/// `(rows, stride)` buffer that reads as a 2D array. The stride is never shorter than the longest
/// row, because padding does not truncate.
pub fn pad_flat(batch: &mut Encoding, params: &PaddingParams) -> Result<()> {
    let rows = batch.rows();
    let Some(longest) = (0..rows).map(|i| batch.row_len(i)).max() else {
        return Ok(());
    };
    let stride = pad_length(longest, params).max(longest);
    if (0..rows).all(|i| batch.row_len(i) == stride) {
        return Ok(());
    }

    let mut ids = vec![PipelineToken::from(params.pad_id); rows * stride];
    let mut mask = vec![0u8; rows * stride];
    // Only worth a buffer when there is something other than zeros to say.
    let mut type_ids = (batch.type_ids.is_some() || params.pad_type_id != 0)
        .then(|| vec![params.pad_type_id as u8; rows * stride]);

    for i in 0..rows {
        let src = batch.row_range(i).expect("[BUG] row out of range");
        // Left padding puts the row at the end of its slot, right padding at the start.
        let at = i * stride
            + match params.direction {
                PaddingDirection::Left => stride - src.len(),
                PaddingDirection::Right => 0,
            };
        let to = at..at + src.len();
        ids[to.clone()].copy_from_slice(&batch.ids[src.clone()]);
        mask[to.clone()].fill(1);
        match (&mut type_ids, &batch.type_ids) {
            (Some(dst), Some(source)) => dst[to].copy_from_slice(&source[src]),
            (Some(dst), None) => dst[to].fill(0),
            _ => {}
        }
    }

    batch.ids = ids;
    batch.attention_mask = Some(mask);
    batch.type_ids = type_ids;
    batch.offsets = Some((0..=rows).map(|i| (i * stride) as u32).collect());
    Ok(())
}

fn pad_one(encoding: &mut Encoding, target_length: usize, params: &PaddingParams) {
    let original_len = encoding.ids.len();
    if original_len >= target_length {
        return;
    }
    let pad_length = target_length - original_len;
    let pad_id = PipelineToken::from(params.pad_id);
    let pad_type_id = params.pad_type_id as u8;

    let mut ids = std::mem::take(&mut encoding.ids);
    let mut type_ids =
        std::mem::take(&mut encoding.type_ids).unwrap_or_else(|| vec![0; original_len]);
    let mut attention_mask =
        std::mem::take(&mut encoding.attention_mask).unwrap_or_else(|| vec![1; original_len]);

    match params.direction {
        PaddingDirection::Left => {
            ids = (0..pad_length).map(|_| pad_id).chain(ids).collect();
            type_ids = (0..pad_length)
                .map(|_| pad_type_id)
                .chain(type_ids)
                .collect();
            attention_mask = (0..pad_length).map(|_| 0).chain(attention_mask).collect();
        }
        PaddingDirection::Right => {
            ids.extend((0..pad_length).map(|_| pad_id));
            type_ids.extend((0..pad_length).map(|_| pad_type_id));
            attention_mask.extend((0..pad_length).map(|_| 0));
        }
    }

    *encoding = Encoding {
        ids,
        type_ids: Some(type_ids),
        attention_mask: Some(attention_mask),
        offsets: None,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tokens(ids: impl IntoIterator<Item = u32>) -> Vec<PipelineToken> {
        ids.into_iter().map(PipelineToken::from).collect()
    }

    fn make_encoding(ids: impl IntoIterator<Item = u32>) -> Encoding {
        Encoding {
            ids: make_tokens(ids),
            type_ids: None,
            attention_mask: None,
            offsets: None,
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
        let mut encodings = [Encoding {
            ids: make_tokens(0..3),
            type_ids: Some(vec![1, 1, 1]),
            attention_mask: None,
            offsets: None,
        }];
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
        let mut encodings = [Encoding {
            ids: make_tokens(0..3),
            type_ids: Some(vec![1, 1, 1]),
            attention_mask: None,
            offsets: None,
        }];
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
}
