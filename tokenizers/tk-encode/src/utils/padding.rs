#[cfg(feature = "config")]
use crate::parallelism::*;
use crate::tokenizer::{Encoding, Result};
#[cfg(feature = "config")]
use serde::{Deserialize, Serialize};

/// The various possible padding directions.
#[cfg_attr(feature = "config", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy)]
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

#[cfg_attr(feature = "config", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
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

#[cfg_attr(feature = "config", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
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
        PaddingStrategy::BatchLongest => {
            #[cfg(feature = "config")]
            let lengths = encodings.maybe_par_iter();
            #[cfg(not(feature = "config"))]
            let lengths = encodings.iter();
            lengths.map(|e| e.get_ids().len()).max().unwrap()
        }
    };

    if let Some(multiple) = params.pad_to_multiple_of
        && multiple > 0
        && pad_length % multiple > 0
    {
        pad_length += multiple - pad_length % multiple;
    }

    #[cfg(feature = "config")]
    let targets = encodings.maybe_par_iter_mut();
    #[cfg(not(feature = "config"))]
    let targets = encodings.iter_mut();
    targets.for_each(|encoding| {
        encoding.pad(
            pad_length,
            params.pad_id,
            params.pad_type_id,
            &params.pad_token,
            params.direction,
        )
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::Encoding;
    use ahash::AHashMap;

    #[test]
    fn pad_to_multiple() {
        fn get_encodings() -> [Encoding; 2] {
            [
                Encoding::new(
                    vec![0, 1, 2, 3, 4],
                    vec![],
                    vec![],
                    vec![],
                    vec![],
                    vec![],
                    vec![],
                    vec![],
                    AHashMap::new(),
                ),
                Encoding::new(
                    vec![0, 1, 2],
                    vec![],
                    vec![],
                    vec![],
                    vec![],
                    vec![],
                    vec![],
                    vec![],
                    AHashMap::new(),
                ),
            ]
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
        assert!(encodings.iter().all(|e| e.get_ids().len() == 8));

        // Test batch
        let mut encodings = get_encodings();
        params.strategy = PaddingStrategy::BatchLongest;
        params.pad_to_multiple_of = Some(6);
        pad_encodings(&mut encodings, &params).unwrap();
        assert!(encodings.iter().all(|e| e.get_ids().len() == 6));

        // Do not crash with 0
        params.pad_to_multiple_of = Some(0);
        pad_encodings(&mut encodings, &params).unwrap();
    }
}
