extern crate tokenizers as tk;

use pyo3::prelude::*;

#[pyclass(dict)]
#[repr(transparent)]
pub struct Encoding {
    encoding: tk::tokenizer::Encoding,
}

impl Encoding {
    pub fn new(encoding: tk::tokenizer::Encoding) -> Self {
        Encoding { encoding }
    }
}

#[pymethods]
impl Encoding {
    // #[getter]
    // fn get_original(&self) -> String {
    //     self.encoding.get_original().to_owned()
    // }

    // #[getter]
    // fn get_normalized(&self) -> String {
    //     self.encoding.get_normalized().to_owned()
    // }

    #[getter]
    fn get_ids(&self) -> Vec<u32> {
        self.encoding.get_ids().to_vec()
    }

    #[getter]
    fn get_tokens(&self) -> Vec<String> {
        self.encoding.get_tokens().to_vec()
    }

    #[getter]
    fn get_type_ids(&self) -> Vec<u32> {
        self.encoding.get_type_ids().to_vec()
    }

    // #[getter]
    // fn get_offsets(&self) -> Vec<(usize, usize)> {
    //     self.encoding.get_offsets().to_vec()
    // }

    #[getter]
    fn get_special_tokens_mask(&self) -> Vec<u32> {
        self.encoding.get_special_tokens_mask().to_vec()
    }

    #[getter]
    fn get_attention_mask(&self) -> Vec<u32> {
        self.encoding.get_attention_mask().to_vec()
    }

    #[getter]
    fn get_overflowing(&self) -> Option<Encoding> {
        self.encoding.get_overflowing().cloned().map(Encoding::new)
    }
}
