use pyo3::prelude::*;
use tk_encode::{PaddingDirection, PaddingParams, PaddingStrategy};

use crate::error::err;

/// How a `Tokenizer` pads what it encodes.
///
/// `length=None` pads every batch to its longest item; a number pads to exactly that.
/// The defaults are those of the released `Tokenizer.enable_padding`.
#[pyclass(frozen)]
pub struct Padding(PaddingParams);

impl Padding {
    pub(crate) fn params(&self) -> &PaddingParams {
        &self.0
    }
}

impl From<PaddingParams> for Padding {
    fn from(params: PaddingParams) -> Self {
        Self(params)
    }
}

#[pymethods]
impl Padding {
    #[new]
    #[pyo3(signature = (direction="right", pad_id=0, pad_type_id=0, pad_token="[PAD]".to_string(), length=None, pad_to_multiple_of=None))]
    fn new(
        direction: &str,
        pad_id: u32,
        pad_type_id: u32,
        pad_token: String,
        length: Option<usize>,
        pad_to_multiple_of: Option<usize>,
    ) -> PyResult<Self> {
        let direction = match direction {
            "left" => PaddingDirection::Left,
            "right" => PaddingDirection::Right,
            other => {
                return Err(err(format!(
                    "padding direction must be \"left\" or \"right\", not {other:?}"
                )));
            }
        };
        Ok(Self(PaddingParams {
            strategy: length.map_or(PaddingStrategy::BatchLongest, PaddingStrategy::Fixed),
            direction,
            pad_to_multiple_of,
            pad_id,
            pad_type_id,
            pad_token,
        }))
    }

    /// `"left"` or `"right"`.
    #[getter]
    fn direction(&self) -> &str {
        self.0.direction.as_ref()
    }

    #[getter]
    fn pad_id(&self) -> u32 {
        self.0.pad_id
    }

    #[getter]
    fn pad_type_id(&self) -> u32 {
        self.0.pad_type_id
    }

    #[getter]
    fn pad_token(&self) -> &str {
        &self.0.pad_token
    }

    /// The fixed length padded to, or `None` when padding to the longest item in the batch.
    #[getter]
    fn length(&self) -> Option<usize> {
        match self.0.strategy {
            PaddingStrategy::Fixed(length) => Some(length),
            PaddingStrategy::BatchLongest => None,
        }
    }

    #[getter]
    fn pad_to_multiple_of(&self) -> Option<usize> {
        self.0.pad_to_multiple_of
    }

    fn __repr__(&self) -> String {
        format!(
            "Padding(direction={:?}, pad_id={}, pad_type_id={}, pad_token={:?}, length={}, pad_to_multiple_of={})",
            self.direction(),
            self.pad_id(),
            self.pad_type_id(),
            self.pad_token(),
            self.length().map_or("None".to_string(), |l| l.to_string()),
            self.pad_to_multiple_of()
                .map_or("None".to_string(), |m| m.to_string()),
        )
    }
}
