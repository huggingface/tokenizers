use std::path::PathBuf;
use std::sync::Mutex;

use pyo3::prelude::*;
use tk_encode::PaddingParams;
use tk_encode::pipeline::{EncodeOptions, Override, PipelineTokenizer as Pipeline};

use crate::encoding::Encoding;
use crate::error::{convert_err, err, poison_err};
use crate::padding::{Padding, PaddingArg};
use crate::repr;
use crate::type_hints::TokenIds;

/// A tokenizer. Encodes text into token ids, and decodes token ids back into text.
#[pyclass(frozen, module = "tokenizers")]
pub struct Tokenizer {
    pipeline: Pipeline,
    // Needs a mutex so a concurrent thread can access the value while
    // encode is running.
    padding: Mutex<Option<PaddingParams>>,
}

impl Tokenizer {
    fn make_options(
        &self,
        add_special_tokens: bool,
        padding: PaddingArg,
    ) -> PyResult<EncodeOptions> {
        let padding = match padding {
            PaddingArg::InheritConfig => {
                self.clone_padding()?.map_or(Override::Off, Override::With)
            }
            PaddingArg::Off => Override::Off,
            PaddingArg::With(params) => Override::With(params),
        };
        Ok(EncodeOptions {
            add_special_tokens,
            padding,
        })
    }

    fn clone_padding(&self) -> PyResult<Option<PaddingParams>> {
        Ok(self.padding.lock().map_err(poison_err)?.clone())
    }
}

/// What `Tokenizer.__reduce__` gives to pickle
type UnpickleArguments = (String, Option<Padding>);

#[pymethods]
impl Tokenizer {
    /// Loads a `tokenizer.json`.
    ///
    /// Args:
    ///     path:
    ///         The file to read.
    #[staticmethod]
    #[pyo3(signature = (path))]
    fn from_file(path: PathBuf) -> PyResult<Self> {
        let canonical = tk_convert::canonicalize_file(path).map_err(convert_err)?;
        let pipeline: Pipeline = tk_serialize::from_json(&canonical).map_err(err)?;
        let padding = pipeline.get_padding().cloned();
        Ok(Self {
            pipeline,
            padding: Mutex::new(padding),
        })
    }

    /// The padding applied to every encode, or `None`.
    /// Assign `None` to switch padding off.
    #[getter]
    fn padding(&self) -> PyResult<Option<Padding>> {
        Ok(self.clone_padding()?.map(Padding::from))
    }

    #[setter]
    fn set_padding(&self, padding: Option<PyRef<'_, Padding>>) -> PyResult<()> {
        *self.padding.lock().map_err(poison_err)? = padding.map(|padding| padding.params().clone());
        Ok(())
    }

    /// Encodes the given text to token ids.
    ///
    /// Args:
    ///     text: str
    ///         The text to encode.
    ///     add_special_tokens: bool
    ///          Whether the post-processor adds its special tokens, such as `[CLS]` and `[SEP]`.
    ///     padding: `Padding` or `None`
    ///         Padding options. Pass `None` to disable padding.
    ///         When omitted, defaults to the padding options configured on the tokenizer.
    ///
    /// Returns:
    ///     Encoding
    #[pyo3(signature = (text, *, add_special_tokens=true, padding=PaddingArg::InheritConfig))]
    fn encode(
        &self,
        py: Python<'_>,
        text: String,
        add_special_tokens: bool,
        padding: PaddingArg,
    ) -> PyResult<Encoding> {
        let options = self.make_options(add_special_tokens, padding)?;
        // py.detach releases the GIL while encode runs on Rust side
        let encodings = py
            .detach(|| self.pipeline.encode(text, &options).wait())
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
    ///     padding: `Padding` or `None`
    ///         Padding options. Pass `None` to disable padding.
    ///         When omitted, defaults to the padding options configured on the tokenizer.
    ///
    /// Returns:
    ///     List[Encoding]
    #[pyo3(signature = (texts, *, add_special_tokens=true, padding=PaddingArg::InheritConfig))]
    fn encode_batch(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        add_special_tokens: bool,
        padding: PaddingArg,
    ) -> PyResult<Vec<Encoding>> {
        let options = self.make_options(add_special_tokens, padding)?;
        // py.detach releases the GIL while encode runs on Rust side
        let encodings = py
            .detach(|| self.pipeline.encode(texts, &options).wait())
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
    fn decode(
        &self,
        py: Python<'_>,
        ids: TokenIds<'_>,
        skip_special_tokens: bool,
    ) -> PyResult<String> {
        let ids = ids.as_slice();
        // py.detach releases the GIL
        py.detach(|| self.pipeline.decode(ids, skip_special_tokens))
            .map_err(err)
    }

    /// Pickle rebuilds a `Tokenizer` by calling `_unpickle` with these arguments.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, UnpickleArguments)> {
        let json = tk_serialize::to_json(&self.pipeline).map_err(err)?;
        Ok((
            py.get_type::<Self>().getattr("_unpickle")?,
            (json, self.padding()?),
        ))
    }

    /// Unpickles a `Tokenizer`
    #[staticmethod]
    fn _unpickle(json: &str, padding: Option<PyRef<'_, Padding>>) -> PyResult<Self> {
        let pipeline: Pipeline = tk_serialize::from_json(json).map_err(err)?;
        Ok(Self {
            pipeline,
            padding: Mutex::new(padding.map(|padding| padding.params().clone())),
        })
    }

    fn __repr__(&self) -> PyResult<String> {
        let file = tk_serialize::to_json(&self.pipeline).map_err(err)?;
        let file: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(&file).map_err(err)?;
        let padding = self
            .padding()?
            .map_or_else(|| "None".to_owned(), |padding| padding.__repr__());
        Ok(repr::tokenizer(&file, &padding))
    }
}
