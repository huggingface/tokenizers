use pyo3::PyErr;
use pyo3::create_exception;
use pyo3::exceptions::PyException;

create_exception!(tokenizers, TokenizersError, PyException);

pub fn to_pyerr(e: tk_encode::Error) -> PyErr {
    TokenizersError::new_err(e.to_string())
}
