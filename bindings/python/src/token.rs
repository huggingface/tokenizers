extern crate tokenizers as tk;

use pyo3::prelude::*;

#[pyclass]
#[repr(transparent)]
pub struct Token {
    tok: tk::tokenizer::Token,
}
impl Token {
    pub fn _new(tok: tk::tokenizer::Token) -> Self {
        Token { tok }
    }
}

#[pymethods]
impl Token {
    #[getter]
    fn get_id(&self) -> PyResult<u32> {
        Ok(self.tok.id)
    }

    #[getter]
    fn get_value(&self) -> PyResult<&str> {
        Ok(&self.tok.value)
    }

    #[getter]
    fn get_offsets(&self) -> PyResult<(usize, usize)> {
        Ok(self.tok.offsets)
    }

    fn as_tuple(&self) -> PyResult<(u32, &str, (usize, usize))> {
        Ok((self.tok.id, &self.tok.value, self.tok.offsets))
    }
}
