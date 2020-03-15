use crate::tokenizer::{Encoding, Result};
use rayon::prelude::*;

/// The various possible padding directions.
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

#[derive(Debug, Clone)]
pub struct PaddingParams {
    pub strategy: PaddingStrategy,
    pub direction: PaddingDirection,
    pub pad_id: u32,
    pub pad_type_id: u32,
    pub pad_token: String,
}

#[derive(Debug, Clone)]
pub enum PaddingStrategy {
    BatchLongest,
    Fixed(usize),
}

pub fn pad_encodings(
    mut encodings: Vec<Encoding>,
    params: &PaddingParams,
) -> Result<Vec<Encoding>> {
    if encodings.is_empty() {
        return Ok(encodings);
    }

    let pad_length = match params.strategy {
        PaddingStrategy::Fixed(size) => size,
        PaddingStrategy::BatchLongest => encodings
            .par_iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap(),
    };

    encodings.par_iter_mut().for_each(|encoding| {
        encoding.pad(
            pad_length,
            params.pad_id,
            params.pad_type_id,
            &params.pad_token,
            params.direction,
        )
    });

    Ok(encodings)
}
