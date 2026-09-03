use pyo3::prelude::*;
use tk_encode::PaddingParams;
use tk_encode::pipeline::PipelineTokenizer as Pipeline;

use crate::encoding::Encoding;
use crate::error::err;
use crate::padding::Padding;

/// The pipeline encode path. `padding` is the only thing that can be changed after `from_file`.
#[pyclass]
pub struct Tokenizer {
    pipeline: Pipeline,
    padding: Option<PaddingParams>,
}

#[pymethods]
impl Tokenizer {
    /// Read a `tokenizer.json`. The file is put through the legacy "1.0" -> canonical "2.0"
    /// upgrade first, so the tokenizers already on disk keep loading; `tk_serialize` itself
    /// only reads the canonical form.
    ///
    /// `padding` replaces the padding the file declares; `None` keeps the file's.
    #[staticmethod]
    #[pyo3(signature = (path, padding=None))]
    fn from_file(path: &str, padding: Option<PyRef<'_, Padding>>) -> PyResult<Self> {
        let canonical = tk_convert::canonicalize_file(path).map_err(err)?;
        let pipeline: Pipeline = tk_serialize::from_json(&canonical).map_err(err)?;
        let padding = match padding {
            Some(padding) => Some(padding.params().clone()),
            None => pipeline.get_padding().cloned(),
        };
        Ok(Self { pipeline, padding })
    }

    /// The padding applied to every encode, or `None`. Assign `None` to switch padding off.
    #[getter]
    fn padding(&self) -> Option<Padding> {
        self.padding.clone().map(Padding::from)
    }

    #[setter]
    fn set_padding(&mut self, padding: Option<PyRef<'_, Padding>>) {
        self.padding = padding.map(|padding| padding.params().clone());
    }

    #[pyo3(signature = (text, add_special_tokens=true))]
    fn encode(&self, text: String, add_special_tokens: bool) -> PyResult<Encoding> {
        Ok(self.encode_batch(vec![text], add_special_tokens)?.remove(0))
    }

    /// Encodes every text in parallel and returns the encodings in input order.
    #[pyo3(signature = (texts, add_special_tokens=true))]
    fn encode_batch(
        &self,
        texts: Vec<String>,
        add_special_tokens: bool,
    ) -> PyResult<Vec<Encoding>> {
        let encodings = self
            .pipeline
            .encode(texts, add_special_tokens)
            .wait_with_padding(self.padding.as_ref())
            .map_err(err)?;
        Ok(encodings.iter().map(Encoding::from).collect())
    }

    #[pyo3(signature = (ids, skip_special_tokens=true))]
    fn decode(&self, ids: Vec<u32>, skip_special_tokens: bool) -> PyResult<String> {
        self.pipeline.decode(&ids, skip_special_tokens).map_err(err)
    }
}
