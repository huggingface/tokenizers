use pyo3::FromPyObject;
use tk_encode::pipeline::{Input, Inputs};

#[derive(FromPyObject)]
pub enum PyInput {
    Single(String),
    Pair((String, String)),
}

impl From<PyInput> for Input {
    fn from(value: PyInput) -> Self {
        match value {
            PyInput::Pair((left, right)) => Input::Pair(left, right),
            PyInput::Single(single) => Input::Single(single),
        }
    }
}

impl From<PyInput> for Inputs {
    fn from(value: PyInput) -> Self {
        Self::Single(value.into())
    }
}
