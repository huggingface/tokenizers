use std::path::PathBuf;
use std::sync::Mutex;

use pyo3::prelude::*;
use tk_encode::PaddingParams;
use tk_encode::pipeline::PipelineTokenizer as Pipeline;

use crate::arrays::Ids;
use crate::encoding::Encoding;
use crate::error::{convert_err, err};
use crate::padding::Padding;
use crate::repr;

/// A tokenizer loaded from a `tokenizer.json`.
///
/// `encode` and `encode_batch` turn text into `Encoding`s, `decode` turns ids back into text.
/// `padding` is the only attribute that can be changed after `from_file`.
#[pyclass(frozen, module = "tokenizers")]
pub struct Tokenizer {
    pipeline: Pipeline,
    // Behind a lock rather than a `&mut self` setter so the class can be `frozen`: `encode`
    // releases the GIL, and another thread assigning `padding` meanwhile must not hit
    // pyo3's "already borrowed" error.
    padding: Mutex<Option<PaddingParams>>,
}

impl Tokenizer {
    fn padding_params(&self) -> Option<PaddingParams> {
        self.padding.lock().unwrap().clone()
    }
}

#[pymethods]
impl Tokenizer {
    /// Loads a `tokenizer.json`. Files written by older `tokenizers` versions are upgraded on
    /// the way in, so anything already on disk loads.
    ///
    /// Args:
    ///     path: The file to read.
    ///     padding: Replaces the padding the file declares. `None` keeps the file's, which for
    ///         most files means no padding.
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

    /// The padding applied to every encode, or `None`. Assign `None` to switch padding off.
    #[getter]
    fn padding(&self) -> Option<Padding> {
        self.padding_params().map(Padding::from)
    }

    #[setter]
    fn set_padding(&self, padding: Option<PyRef<'_, Padding>>) {
        *self.padding.lock().unwrap() = padding.map(|padding| padding.params().clone());
    }

    /// Encodes one text.
    ///
    /// Args:
    ///     text: The text to encode.
    ///     add_special_tokens: Whether the post-processor adds its special tokens, such as
    ///         `[CLS]` and `[SEP]`.
    #[pyo3(signature = (text, add_special_tokens=true))]
    fn encode(&self, py: Python<'_>, text: String, add_special_tokens: bool) -> PyResult<Encoding> {
        let padding = self.padding_params();
        let encodings = py
            .detach(|| {
                self.pipeline
                    .encode(text, add_special_tokens)
                    .wait_with_padding(padding.as_ref())
            })
            .map_err(err)?;
        Ok(Encoding::from(&encodings[0]))
    }

    /// Encodes every text in parallel. The encodings come back in input order.
    ///
    /// Args:
    ///     texts: The texts to encode.
    ///     add_special_tokens: Whether the post-processor adds its special tokens, such as
    ///         `[CLS]` and `[SEP]`.
    #[pyo3(signature = (texts, add_special_tokens=true))]
    fn encode_batch(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        add_special_tokens: bool,
    ) -> PyResult<Vec<Encoding>> {
        let padding = self.padding_params();
        let encodings = py
            .detach(|| {
                self.pipeline
                    .encode(texts, add_special_tokens)
                    .wait_with_padding(padding.as_ref())
            })
            .map_err(err)?;
        Ok(encodings.iter().map(Encoding::from).collect())
    }

    /// Turns ids back into text.
    ///
    /// Args:
    ///     ids: The ids to decode, a numpy array or any sequence of ints.
    ///     skip_special_tokens: Whether special tokens are left out of the text.
    #[pyo3(signature = (ids, skip_special_tokens=true))]
    fn decode(&self, py: Python<'_>, ids: Ids<'_>, skip_special_tokens: bool) -> PyResult<String> {
        let ids = ids.as_slice();
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
