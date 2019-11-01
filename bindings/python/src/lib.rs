extern crate tokenizers as tk;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyfunction]
/// Tokenize
fn tokenize(a: String) -> PyResult<Vec<u32>> {
    println!("Tokenize in rust");
    tk::test();
    Ok(vec![1, 2, 3])
}

#[pymodule]
fn tokenizers(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(tokenize))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
