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

/// Pad every encoding of a batch to one length, its overflowing windows included: a window is a
/// row of its own and shares the target of the encoding it came from. `BatchLongest` measures
/// every row, windows included and recursively, so the batch comes out rectangular whatever the
/// truncation left behind: a window longer than every encoding -- the single window of a
/// truncation with no room -- is what sets the length then. `Fixed` names the length instead,
/// and a row already longer than it is left alone.
pub fn pad_encodings(encodings: &mut [Encoding], params: &PaddingParams) -> Result<()> {
    if encodings.is_empty() {
        return Ok(());
    }

    let mut pad_length = match params.strategy {
        PaddingStrategy::Fixed(size) => size,
        PaddingStrategy::BatchLongest => encodings.iter().map(longest_row).max().unwrap(),
    };

    if let Some(multiple) = params.pad_to_multiple_of
        && multiple > 0
        && pad_length % multiple > 0
    {
        pad_length += multiple - pad_length % multiple;
    }

    encodings
        .iter_mut()
        .for_each(|encoding| pad_one(encoding, pad_length, params));

    Ok(())
}

/// The length of the longest row of an encoding: the encoding itself or any window it carries,
/// windows inside windows included.
fn longest_row(encoding: &Encoding) -> usize {
    encoding.len().max(
        encoding
            .overflowing
            .iter()
            .map(longest_row)
            .max()
            .unwrap_or(0),
    )
}

fn pad_one(encoding: &mut Encoding, target_length: usize, params: &PaddingParams) {
    // A window is a row of its own and gets the same target length as the encoding it came from.
    // One that is already longer -- the single window of a truncation with no room at all -- is
    // left alone: padding lengthens, it does not shorten.
    let mut overflowing = std::mem::take(&mut encoding.overflowing);
    for window in &mut overflowing {
        pad_one(window, target_length, params);
    }

    let original_len = encoding.ids.len();
    if original_len >= target_length {
        encoding.overflowing = overflowing;
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
        overflowing,
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
            overflowing: Vec::new(),
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
            overflowing: Vec::new(),
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
            overflowing: Vec::new(),
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

    // An overflowing window is a row of its own and is padded together with the encoding it came
    // from: same target length, same direction, padding at the same end.
    #[test]
    fn test_windows_are_padded_with_their_encoding() {
        let mut encoding = make_encoding(0..3);
        encoding.overflowing = vec![make_encoding(0..2)];
        let params = PaddingParams {
            strategy: PaddingStrategy::Fixed(5),
            direction: PaddingDirection::Right,
            pad_id: 99,
            ..PaddingParams::default()
        };

        pad_encodings(std::slice::from_mut(&mut encoding), &params).unwrap();

        assert_eq!(encoding.ids(), make_tokens([0, 1, 2, 99, 99]));
        let window = &encoding.overflowing[0];
        assert_eq!(window.ids(), make_tokens([0, 1, 99, 99, 99]));
        assert_eq!(window.attention_mask().unwrap(), [1, 1, 0, 0, 0]);
    }

    // A `Fixed` target names the length, so a window that is already longer -- the single window
    // of a truncation with no room at all -- stays as it is: padding lengthens, it does not
    // shorten. `BatchLongest` measures every row instead and would reach that window's length.
    #[test]
    fn test_a_window_longer_than_the_target_is_left_alone() {
        let mut encoding = make_encoding(0..2);
        encoding.overflowing = vec![make_encoding(0..7)];
        let params = PaddingParams {
            strategy: PaddingStrategy::Fixed(5),
            ..PaddingParams::default()
        };

        pad_encodings(std::slice::from_mut(&mut encoding), &params).unwrap();

        assert_eq!(encoding.ids().len(), 5);
        assert_eq!(encoding.overflowing[0].ids().len(), 7);
    }

    // In a batch every row of every encoding reaches the same length, so the rows a caller
    // flattens stay rectangular.
    #[test]
    fn test_batch_windows_reach_the_batch_length() {
        let mut first = make_encoding(0..3);
        first.overflowing = vec![make_encoding(0..1)];
        let mut encodings = [first, make_encoding(0..5)];
        let params = PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..PaddingParams::default()
        };

        pad_encodings(&mut encodings, &params).unwrap();

        assert_eq!(encodings[0].ids().len(), 5);
        assert_eq!(encodings[0].overflowing[0].ids().len(), 5);
        assert_eq!(encodings[1].ids().len(), 5);
    }

    // A truncation with no room leaves the sequence as one window longer than its encoding, and
    // an unpickled `Encoding` can carry windows inside windows. `BatchLongest` has to measure
    // every row, deepest first, or the batch comes out ragged and cannot become an array.
    #[test]
    fn test_batch_longest_measures_the_windows_recursively() {
        let mut nested = make_encoding(0..7);
        nested.overflowing = vec![make_encoding(0..9)];
        let mut with_window = make_encoding(0..2);
        with_window.overflowing = vec![nested];
        let mut encodings = [with_window, make_encoding(0..3)];
        let params = PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..PaddingParams::default()
        };

        pad_encodings(&mut encodings, &params).unwrap();

        assert_eq!(encodings[0].len(), 9);
        assert_eq!(encodings[0].overflowing[0].len(), 9);
        assert_eq!(encodings[0].overflowing[0].overflowing[0].len(), 9);
        assert_eq!(encodings[1].len(), 9);
    }
}
