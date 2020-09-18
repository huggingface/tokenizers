use pyo3::prelude::*;
use tk::Token;

#[pyclass(module = "tokenizers", name=Token)]
#[derive(Clone)]
pub struct PyToken {
    token: Token,
}
impl From<Token> for PyToken {
    fn from(token: Token) -> Self {
        Self { token }
    }
}
impl From<PyToken> for Token {
    fn from(token: PyToken) -> Self {
        token.token
    }
}

#[pymethods]
impl PyToken {
    #[new]
    fn new(id: u32, value: String, offsets: (usize, usize)) -> PyToken {
        Token::new(id, value, offsets).into()
    }

    #[getter]
    fn get_id(&self) -> PyResult<u32> {
        Ok(self.token.id)
    }

    #[getter]
    fn get_value(&self) -> PyResult<&str> {
        Ok(&self.token.value)
    }

    #[getter]
    fn get_offsets(&self) -> PyResult<(usize, usize)> {
        Ok(self.token.offsets)
    }

    fn as_tuple(&self) -> PyResult<(u32, &str, (usize, usize))> {
        Ok((self.token.id, &self.token.value, self.token.offsets))
    }
}
