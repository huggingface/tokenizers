use std::path::PathBuf;
use std::sync::Mutex;

use pyo3::prelude::*;
use tk_encode::PaddingParams;
use tk_encode::pipeline::PipelineTokenizer as Pipeline;

use crate::encoding::Encoding;
use crate::error::{convert_err, err};
use crate::type_hints::TokenIds;
use crate::padding::Padding;
use crate::repr;

/// A tokenizer. Encodes text into token ids, and decodes token ids back into text.
#[pyclass(frozen, module = "tokenizers")]
pub struct Tokenizer {
    pipeline: Pipeline,
    // Needs a mutex so a concurrent thread can access the value while
    // encode is running.
    padding: Mutex<Option<PaddingParams>>,
}

impl Tokenizer {
    fn clone_padding_params(&self) -> Option<PaddingParams> {
        self.padding.lock().unwrap().clone()
    }
}

#[pymethods]
impl Tokenizer {
    /// Loads a `tokenizer.json`.
    ///
    /// Args:
    ///     path:
    ///         The file to read.
    ///     padding:
    ///         Replaces the padding configuration the file declares. 
    ///         None` means use the file's padding configuration.
    #[staticmethod]
    #[pyo3(signature = (path, padding=None))]
    fn from_file(path: PathBuf, padding: Option<PyRef<'_, Padding>>) -> PyResult<Self> {
        let canonical = tk_convert::canonicalize_file(path).map_err(convert_err)?;
        let pipeline: Pipeline = tk_serialize::from_json(&canonical).map_err(err)?;
        let padding = match padding {
            Some(padding) => Some(padding.params().clone()),
            None => pipeline.get_padding().cloned(),
        };
        Ok(Self {
            pipeline,
            padding: Mutex::new(padding),
        })
    }

    /// The padding applied to every encode, or `None`.
    /// Assign `None` to switch padding off.
    #[getter]
    fn padding(&self) -> Option<Padding> {
        self.clone_padding_params().map(Padding::from)
    }

    #[setter]
    fn set_padding(&self, padding: Option<PyRef<'_, Padding>>) {
        *self.padding.lock().unwrap() = padding.map(|padding| padding.params().clone());
    }

    /// Encodes the given text to token ids.
    ///
    /// Args:
    ///     text: str
    ///         The text to encode.
    ///     add_special_tokens: bool
    ///          Whether the post-processor adds its special tokens, such as `[CLS]` and `[SEP]`.
    /// 
    /// Returns:
    ///     Encoding
    #[pyo3(signature = (text, add_special_tokens=true))]
    fn encode(&self, py: Python<'_>, text: String, add_special_tokens: bool) -> PyResult<Encoding> {
        let padding = self.clone_padding_params();
        // py.detach releases the GIL while encode runs on Rust side
        let encodings = py
            .detach(|| {
                self.pipeline
                    .encode(text, add_special_tokens)
                    .wait_with_padding(padding.as_ref())
            })
            .map_err(err)?;
        Ok(Encoding::from(&encodings[0]))
    }

    /// Encodes a batch of text.
    /// The encodings come back in input order.
    ///
    /// Args:
    ///     texts: List[str]
    ///         The batch of text to encode.
    ///     add_special_tokens: bool
    ///         Whether the post-processor adds its special tokens, such as `[CLS]` and `[SEP]`.
    /// 
    /// Returns:
    ///     List[Encoding]
    #[pyo3(signature = (texts, add_special_tokens=true))]
    fn encode_batch(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        add_special_tokens: bool,
    ) -> PyResult<Vec<Encoding>> {
        let padding = self.clone_padding_params();
        // py.detach releases the GIL while encode runs on Rust side
        let encodings = py
            .detach(|| {
                self.pipeline
                    .encode(texts, add_special_tokens)
                    .wait_with_padding(padding.as_ref())
            })
            .map_err(err)?;
        Ok(encodings.iter().map(Encoding::from).collect())
    }

    /// Decodes token ids back into text
    ///
    /// Args:
    ///     ids:
    ///         The ids to decode, a numpy array or any sequence of ints.
    ///     skip_special_tokens: bool
    ///         Whether special tokens should not be added to the decoded text.
    /// 
    /// Returns:
    ///     str
    #[pyo3(signature = (ids, skip_special_tokens=true))]
    fn decode(&self, py: Python<'_>, ids: TokenIds<'_>, skip_special_tokens: bool) -> PyResult<String> {
        let ids = ids.as_slice();
        // py.detach releases the GIL
        py.detach(|| self.pipeline.decode(ids, skip_special_tokens))
            .map_err(err)
    }

    fn __repr__(&self) -> PyResult<String> {
        let file = tk_serialize::to_json(&self.pipeline).map_err(err)?;
        let file: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(&file).map_err(err)?;
        let padding = self
            .padding()
            .map_or_else(|| "None".to_owned(), |padding| padding.__repr__());
        Ok(repr::tokenizer(&file, &padding))
    }
}
