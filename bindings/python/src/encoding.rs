extern crate tokenizers as tk;

use crate::error::PyError;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;
use pyo3::{PyMappingProtocol, PyObjectProtocol};
use tk::tokenizer::PaddingDirection;

fn get_range(item: PyObject, max_len: usize) -> PyResult<std::ops::Range<usize>> {
    let gil = Python::acquire_gil();
    let py = gil.python();

    let slice = if let Ok(index) = item.extract::<isize>(py) {
        if index >= max_len as isize || index < -(max_len as isize) {
            Err(exceptions::IndexError::py_err("Index out of bounds"))
        } else {
            Ok(if index == -1 {
                PySlice::new(py, index, max_len as isize, 1)
            } else {
                PySlice::new(py, index, index + 1, 1)
            })
        }
    } else if let Ok(slice) = item.cast_as::<PySlice>(py) {
        Ok(slice)
    } else if let Ok(offset) = item.cast_as::<PyTuple>(py) {
        if offset.len() == 2 {
            let start = offset.get_item(0).extract::<isize>()?;
            let end = offset.get_item(1).extract::<isize>()?;
            Ok(PySlice::new(py, start, end, 1))
        } else {
            Err(exceptions::TypeError::py_err("Expected Tuple[int, int]"))
        }
    } else {
        Err(exceptions::TypeError::py_err(
            "Expected number or slice or Tuple[int, int]",
        ))
    }?;

    // Find out range from the slice
    let len: std::os::raw::c_long = (max_len as i32) as _;
    let PySliceIndices { start, stop, .. } = slice.indices(len)?;

    Ok(start as usize..stop as usize)
}

enum IndexableStringType {
    Original,
    Normalized,
}

#[pyclass(dict)]
pub struct IndexableString {
    s: tk::tokenizer::NormalizedString,
    t: IndexableStringType,
}
#[pymethods]
impl IndexableString {
    fn offsets(&self, item: PyObject) -> PyResult<Option<(usize, usize)>> {
        let range = get_range(item, self.s.len())?;

        match self.t {
            IndexableStringType::Original => Ok(self
                .s
                .get_original_offsets(range)
                .map(|range| (range.start, range.end))),
            IndexableStringType::Normalized => Ok(Some((range.start, range.end))),
        }
    }
}

#[pyproto]
impl PyObjectProtocol for IndexableString {
    fn __repr__(&self) -> PyResult<String> {
        Ok(match self.t {
            IndexableStringType::Original => self.s.get_original().to_owned(),
            IndexableStringType::Normalized => self.s.get().to_owned(),
        })
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(match self.t {
            IndexableStringType::Original => self.s.get_original().to_owned(),
            IndexableStringType::Normalized => self.s.get().to_owned(),
        })
    }
}

#[pyproto]
impl PyMappingProtocol for IndexableString {
    fn __getitem__(&self, item: PyObject) -> PyResult<String> {
        // Find out the range
        let range = get_range(item, self.s.len())?;

        // Get the range from the relevant string
        let s = match self.t {
            IndexableStringType::Original => self.s.get_range_original(range),
            IndexableStringType::Normalized => self.s.get_range(range),
        };

        s.map(|s| s.to_owned())
            .ok_or_else(|| exceptions::IndexError::py_err("Wrong offsets"))
    }

    fn __len__(self) -> PyResult<usize> {
        Ok(match self.t {
            IndexableStringType::Original => self.s.len_original(),
            IndexableStringType::Normalized => self.s.len(),
        })
    }
}

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

#[pyproto]
impl PyObjectProtocol for Encoding {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "Encoding(num_tokens={}, attributes=[ids, type_ids, tokens, offsets, \
             attention_mask, special_tokens_mask, overflowing, original_str, normalized_str])",
            self.encoding.get_ids().len()
        ))
    }
}

#[pymethods]
impl Encoding {
    #[getter]
    fn get_normalized_str(&self) -> IndexableString {
        IndexableString {
            s: self.encoding.get_normalized().clone(),
            t: IndexableStringType::Normalized,
        }
    }

    #[getter]
    fn get_original_str(&self) -> IndexableString {
        IndexableString {
            s: self.encoding.get_normalized().clone(),
            t: IndexableStringType::Original,
        }
    }

    #[args(kwargs = "**")]
    fn get_range(
        &self,
        range: (usize, usize),
        kwargs: Option<&PyDict>,
    ) -> PyResult<Option<String>> {
        let mut original = false;
        if let Some(kwargs) = kwargs {
            if let Some(koriginal) = kwargs.get_item("original") {
                original = koriginal.extract()?;
            }
        }

        if original {
            Ok(self
                .encoding
                .get_normalized()
                .get_range_original(range.0..range.1)
                .map(|s| s.to_owned()))
        } else {
            Ok(self
                .encoding
                .get_normalized()
                .get_range(range.0..range.1)
                .map(|s| s.to_owned()))
        }
    }

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

    #[getter]
    fn get_offsets(&self) -> Vec<(usize, usize)> {
        self.encoding.get_offsets().to_vec()
    }

    #[getter]
    fn get_special_tokens_mask(&self) -> Vec<u32> {
        self.encoding.get_special_tokens_mask().to_vec()
    }

    #[getter]
    fn get_attention_mask(&self) -> Vec<u32> {
        self.encoding.get_attention_mask().to_vec()
    }

    #[getter]
    fn get_overflowing(&self) -> Vec<Encoding> {
        self.encoding
            .get_overflowing()
            .clone()
            .into_iter()
            .map(Encoding::new)
            .collect()
    }

    #[args(kwargs = "**")]
    fn pad(&mut self, length: usize, kwargs: Option<&PyDict>) -> PyResult<()> {
        let mut pad_id = 0;
        let mut pad_type_id = 0;
        let mut pad_token = "[PAD]";
        let mut direction = PaddingDirection::Right;

        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "direction" => {
                        let value: &str = value.extract()?;
                        direction = match value {
                            "left" => Ok(PaddingDirection::Left),
                            "right" => Ok(PaddingDirection::Right),
                            other => Err(PyError(format!(
                                "Unknown `direction`: `{}`. Use \
                                 one of `left` or `right`",
                                other
                            ))
                            .into_pyerr()),
                        }?;
                    }
                    "pad_id" => pad_id = value.extract()?,
                    "pad_type_id" => pad_type_id = value.extract()?,
                    "pad_token" => pad_token = value.extract()?,
                    _ => println!("Ignored unknown kwarg option {}", key),
                }
            }
        }

        Ok(self
            .encoding
            .pad(length, pad_id, pad_type_id, pad_token, direction))
    }

    #[args(kwargs = "**")]
    fn truncate(&mut self, max_length: usize, kwargs: Option<&PyDict>) -> PyResult<()> {
        let mut stride = 0;

        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "stride" => stride = value.extract()?,
                    _ => println!("Ignored unknown kwarg option {}", key),
                }
            }
        }

        Ok(self.encoding.truncate(max_length, stride))
    }
}
