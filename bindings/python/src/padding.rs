use std::convert::Infallible;

use pyo3::inspect::{PyStaticConstant, PyStaticExpr};
use pyo3::prelude::*;
use pyo3::types::PyString;
use pyo3::{Borrowed, type_hint_identifier, type_hint_subscript};
use tk_encode::{PaddingDirection, PaddingParams, PaddingStrategy};

use crate::error::err;
use crate::pickle::{self, Reduced};

/// Padding parameters for Tokenizer.encode
#[pyclass(frozen, eq, hash, module = "tokenizers")]
#[derive(PartialEq, Hash)]
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

/// Utils to type hint `PaddingDirection` as `Literal["left", "right"]`
struct Direction(PaddingDirection);

const DIRECTION_HINT: PyStaticExpr = type_hint_subscript!(
    type_hint_identifier!("typing", "Literal"),
    PyStaticExpr::Constant {
        value: PyStaticConstant::Str("left")
    },
    PyStaticExpr::Constant {
        value: PyStaticConstant::Str("right")
    }
);

impl FromPyObject<'_, '_> for Direction {
    type Error = PyErr;
    const INPUT_TYPE: PyStaticExpr = DIRECTION_HINT;

    fn extract(ob: Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        match ob.extract::<&str>()? {
            "left" => Ok(Self(PaddingDirection::Left)),
            "right" => Ok(Self(PaddingDirection::Right)),
            other => Err(err(format!(
                "padding direction must be \"left\" or \"right\", not {other:?}"
            ))),
        }
    }
}

impl<'py> IntoPyObject<'py> for Direction {
    type Target = PyString;
    type Output = Bound<'py, PyString>;
    type Error = Infallible;
    const OUTPUT_TYPE: PyStaticExpr = DIRECTION_HINT;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(PyString::new(py, self.0.as_ref()))
    }
}

/// What `Padding.__reduce__` hands pickle: every argument of its constructor.
type Arguments = (Direction, u32, u32, String, Option<usize>, Option<usize>);

#[pymethods]
impl Padding {
    /// Args:
    ///     direction: `"right"` (the default) or `"left"`
    ///         whether padding tokens are appended to the right or
    ///         prepended to the left  of encoded tokens.
    ///     pad_id: int
    ///         The id of the padding token.
    ///     pad_type_id: int
    ///         The type id of the padding token.
    ///     pad_token: str
    ///         The text of the padding token.
    ///     length: int (optional)
    ///         Pads every encoding to exactly this many tokens. `None` pads each batch to its
    ///         longest item.
    ///     pad_to_multiple_of: int (optional)
    ///         Rounds the padded length up to a multiple of this.
    #[new]
    #[pyo3(signature = (direction=Direction(PaddingDirection::Right), pad_id=0, pad_type_id=0, pad_token="[PAD]", length=None, pad_to_multiple_of=None))]
    fn new(
        direction: Direction,
        pad_id: u32,
        pad_type_id: u32,
        pad_token: &str,
        length: Option<usize>,
        pad_to_multiple_of: Option<usize>,
    ) -> Self {
        Self(PaddingParams {
            strategy: length.map_or(PaddingStrategy::BatchLongest, PaddingStrategy::Fixed),
            direction: direction.0,
            pad_to_multiple_of,
            pad_id,
            pad_type_id,
            pad_token: pad_token.to_owned(),
        })
    }

    /// `"left"` or `"right"`.
    /// Whether padding tokens are appended to the right or prepended to the left  of encoded tokens.
    #[getter]
    fn direction(&self) -> Direction {
        Direction(self.0.direction)
    }

    /// The id of the padding token.
    #[getter]
    fn pad_id(&self) -> u32 {
        self.0.pad_id
    }

    /// The type id of the padding token.
    #[getter]
    fn pad_type_id(&self) -> u32 {
        self.0.pad_type_id
    }

    /// The text of the padding token.
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

    /// The multiple the padded length is rounded up to, or `None`.
    #[getter]
    fn pad_to_multiple_of(&self) -> Option<usize> {
        self.0.pad_to_multiple_of
    }

    fn __reduce__(&self, py: Python<'_>) -> Reduced<Arguments> {
        let arguments = (
            self.direction(),
            self.pad_id(),
            self.pad_type_id(),
            self.pad_token().to_owned(),
            self.length(),
            self.pad_to_multiple_of(),
        );
        (pickle::class::<Self>(py), arguments)
    }

    pub(crate) fn __repr__(&self) -> String {
        format!(
            "Padding(direction={:?}, pad_id={}, pad_type_id={}, pad_token={:?}, length={}, pad_to_multiple_of={})",
            self.0.direction.as_ref(),
            self.pad_id(),
            self.pad_type_id(),
            self.pad_token(),
            self.length().map_or("None".to_string(), |l| l.to_string()),
            self.pad_to_multiple_of()
                .map_or("None".to_string(), |m| m.to_string()),
        )
    }
}
