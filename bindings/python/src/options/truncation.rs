use std::convert::Infallible;

use pyo3::{
    inspect::{PyStaticConstant, PyStaticExpr},
    prelude::*,
    type_hint_identifier, type_hint_subscript,
    types::{PyString, PyType},
};

use tk::{TruncationDirection, TruncationParams, TruncationStrategy};

use crate::error::err;

#[pyclass(frozen, eq, hash, module = "tokenizers")]
#[derive(Debug, PartialEq, Hash)]
/// A class exposing truncation options to Python consumers
pub struct Truncation(TruncationParams);

impl Truncation {
    pub(crate) fn params(&self) -> &TruncationParams {
        &self.0
    }

    pub(crate) fn from(params: TruncationParams) -> Self {
        Self(params)
    }
}

/// Utils to type hint `TruncationStrategy` as
/// `Literal["longest_first", "only_first", "only_second"]`
struct Strategy(TruncationStrategy);

const STRATEGY_HINT: PyStaticExpr = type_hint_subscript!(
    type_hint_identifier!("typing", "Literal"),
    PyStaticExpr::Constant {
        value: PyStaticConstant::Str("longest_first")
    },
    PyStaticExpr::Constant {
        value: PyStaticConstant::Str("only_first")
    },
    PyStaticExpr::Constant {
        value: PyStaticConstant::Str("only_second")
    }
);

impl FromPyObject<'_, '_> for Strategy {
    type Error = PyErr;
    const INPUT_TYPE: PyStaticExpr = STRATEGY_HINT;

    fn extract(ob: Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        match ob.extract::<&str>()? {
            "longest_first" => Ok(Self(TruncationStrategy::LongestFirst)),
            "only_first" => Ok(Self(TruncationStrategy::OnlyFirst)),
            "only_second" => Ok(Self(TruncationStrategy::OnlySecond)),
            other => Err(err(format!(
                "truncation strategy must be \"longest_first\", \"only_first\" or \"only_second\", not {other:?}"
            ))),
        }
    }
}

impl<'py> IntoPyObject<'py> for Strategy {
    type Target = PyString;
    type Output = Bound<'py, PyString>;
    type Error = Infallible;
    const OUTPUT_TYPE: PyStaticExpr = STRATEGY_HINT;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(PyString::new(py, self.0.as_ref()))
    }
}

/// Utils to type hint `TruncationDirection` as `Literal["left", "right"]`
struct Direction(TruncationDirection);

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
            "left" => Ok(Self(TruncationDirection::Left)),
            "right" => Ok(Self(TruncationDirection::Right)),
            other => Err(err(format!(
                "truncation direction must be \"left\" or \"right\", not {other:?}"
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

/// What `Truncation.__reduce__` gives pickle: every argument of the constructor, in its order.
/// A pickle written before `stride` existed carries the first three and gets `stride` 0.
type UnpickleArguments = (usize, Strategy, Direction, usize);

#[pymethods]
impl Truncation {
    /// Args:
    ///     max_length: int
    ///         The maximum number of tokens, including special tokens, to keep. Encodings with
    /// 		more tokens are truncated to one window of the input: from the start when
    /// 		`direction` is `"right"` (the default), from the end when it is `"left"`. For a
    /// 		pair, `strategy` decides which sequence gives up tokens and the room the special
    /// 		tokens leave is shared between the two.
    ///     strategy: `"longest_first"` (the default), `"only_first"` or `"only_second"`
    ///         Which sequence of a pair is truncated.
    ///     direction: `"right"` (the default) or `"left"`
    ///         Whether to truncate tokens at the end of the sequence (`"right"`) or at
    /// 		the beginning of the sequence (`"left"`).
    ///     stride: int
    ///         The length of previous content to be included in each overflowing piece.
    ///
    ///         The kept window is returned as the encoding and the rest as `Encoding.overflowing`.
    ///         For each truncated sequence, successive windows advance by its content room minus
    ///         `stride`. This room excludes special tokens and, for pairs, depends on `strategy`.
    ///         The last window can be shorter. With no content room, the whole sequence is returned
    ///         as one overflow window. `stride` must be strictly less than a truncated sequence's
    ///         nonzero content room; a sequence that fits is never windowed.
    #[new]
    #[pyo3(signature = (max_length, strategy=Strategy(TruncationStrategy::LongestFirst), direction=Direction(TruncationDirection::Right), stride=0))]
    fn new(max_length: usize, strategy: Strategy, direction: Direction, stride: usize) -> Self {
        Self(TruncationParams {
            max_length,
            strategy: strategy.0,
            direction: direction.0,
            stride,
        })
    }

    /// The maximum number of tokens, including special tokens, to keep. Encodings with more tokens will get truncated.
    #[getter]
    fn max_length(&self) -> usize {
        self.0.max_length
    }

    /// Which sequence of a pair is truncated.
    #[getter]
    fn strategy(&self) -> Strategy {
        Strategy(self.0.strategy)
    }

    /// Whether to truncate tokens at the end of the sequence (`"right"`) or at the beginning of the sequence (`"left"`).
    #[getter]
    fn direction(&self) -> Direction {
        Direction(self.0.direction)
    }

    /// The length of previous content to be included in each overflowing piece.
    #[getter]
    fn stride(&self) -> usize {
        self.0.stride
    }

    /// Pickle rebuilds a `Truncation` by calling the class with these constructor arguments.
    fn __reduce__<'py>(&self, py: Python<'py>) -> (Bound<'py, PyType>, UnpickleArguments) {
        let arguments = (
            self.max_length(),
            self.strategy(),
            self.direction(),
            self.stride(),
        );
        (py.get_type::<Self>(), arguments)
    }

    pub(crate) fn __repr__(&self) -> String {
        format!(
            "Truncation(max_length={}, strategy={:?}, direction={:?}, stride={})",
            self.max_length(),
            self.0.strategy.as_ref(),
            self.0.direction.as_ref(),
            self.0.stride,
        )
    }
}
