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

pub fn pad_encodings(encodings: &mut [Encoding], params: &PaddingParams) -> Result<()> {
    if encodings.is_empty() {
        return Ok(());
    }

    let mut pad_length = match params.strategy {
        PaddingStrategy::Fixed(size) => size,
        PaddingStrategy::BatchLongest => encodings.iter().map(Encoding::len).max().unwrap(),
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

fn pad_one(encoding: &mut Encoding, target_length: usize, params: &PaddingParams) {
    let original_len = encoding.ids.len();
    if original_len >= target_length {
        return;
    }
    let pad_length = target_length - original_len;
    let pad_id = PipelineToken::from(params.pad_id);

    let mut ids = std::mem::take(&mut encoding.ids);
    let mut type_ids = std::mem::take(&mut encoding.type_ids);
    let mut attention_mask =
        std::mem::take(&mut encoding.attention_mask).unwrap_or_else(|| vec![1; original_len]);

    match params.direction {
        PaddingDirection::Left => {
            ids = (0..pad_length).map(|_| pad_id).chain(ids).collect();
            type_ids = type_ids.map(|type_ids| {
                (0..pad_length)
                    .map(|_| params.pad_type_id as u8)
                    .chain(type_ids)
                    .collect()
            });
            attention_mask = (0..pad_length).map(|_| 0).chain(attention_mask).collect();
        }
        PaddingDirection::Right => {
            ids.extend((0..pad_length).map(|_| pad_id));
            if let Some(type_ids) = type_ids.as_mut() {
                type_ids.extend((0..pad_length).map(|_| params.pad_type_id as u8));
            }
            attention_mask.extend((0..pad_length).map(|_| 0));
        }
    }

    *encoding = Encoding {
        ids,
        type_ids,
        attention_mask: Some(attention_mask),
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
        }
    }

    #[test]
    fn pad_to_multiple() {
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
    fn pad_is_a_no_op_once_already_long_enough() {
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
    fn pad_right_appends_pad_id_and_marks_it_in_the_attention_mask() {
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
    fn pad_left_prepends_pad_id_and_marks_it_in_the_attention_mask() {
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
    fn pad_extends_type_ids_when_the_encoding_carries_them() {
        let mut encodings = [Encoding {
            ids: make_tokens(0..3),
            type_ids: Some(vec![1, 1, 1]),
            attention_mask: None,
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
    fn pad_leaves_type_ids_absent_when_the_encoding_never_carried_them() {
        let mut encodings = [make_encoding(0..3)];
        let params = PaddingParams {
            strategy: PaddingStrategy::Fixed(5),
            ..PaddingParams::default()
        };

        pad_encodings(&mut encodings, &params).unwrap();

        assert!(encodings[0].type_ids().is_none());
    }
}
