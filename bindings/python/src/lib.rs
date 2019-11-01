extern crate tokenizers as tk;

use pyo3::prelude::*;

#[pyclass]
struct WhitespaceTokenizer {}

#[pymethods]
impl WhitespaceTokenizer {
    #[staticmethod]
    fn tokenize(s: String) -> PyResult<Vec<String>> {
        Ok(tk::WhitespaceTokenizer::tokenize(&s))
    }
}

#[pymodule]
fn tokenizers(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<WhitespaceTokenizer>()?;
    Ok(())
}
