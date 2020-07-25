use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;
use pyo3::PyObjectProtocol;
use tk::models::bpe::BPE;
use tk::tokenizer::{
    PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams,
    TruncationStrategy,
};
use tokenizers as tk;

use super::decoders::Decoder;
use super::encoding::PyEncoding;
use super::error::{PyError, ToPyResult};
use super::models::PyModel;
use super::normalizers::Normalizer;
use super::pre_tokenizers::PreTokenizer;
use super::processors::PostProcessor;
use super::trainers::PyTrainer;
use super::utils::Container;

#[pyclass(dict, module = "tokenizers", name=AddedToken)]
pub struct PyAddedToken {
    pub content: String,
    pub is_special_token: bool,
    pub single_word: Option<bool>,
    pub lstrip: Option<bool>,
    pub rstrip: Option<bool>,
    pub normalized: Option<bool>,
}
impl PyAddedToken {
    pub fn from<S: Into<String>>(content: S, is_special_token: Option<bool>) -> Self {
        Self {
            content: content.into(),
            is_special_token: is_special_token.unwrap_or(false),
            single_word: None,
            lstrip: None,
            rstrip: None,
            normalized: None,
        }
    }

    pub fn get_token(&self) -> tk::tokenizer::AddedToken {
        let mut token = tk::AddedToken::from(&self.content, self.is_special_token);

        if let Some(sw) = self.single_word {
            token = token.single_word(sw);
        }
        if let Some(ls) = self.lstrip {
            token = token.lstrip(ls);
        }
        if let Some(rs) = self.rstrip {
            token = token.rstrip(rs);
        }
        if let Some(n) = self.normalized {
            token = token.normalized(n);
        }

        token
    }

    pub fn as_pydict<'py>(&self, py: Python<'py>) -> PyResult<&'py PyDict> {
        let dict = PyDict::new(py);
        let token = self.get_token();

        dict.set_item("content", token.content)?;
        dict.set_item("single_word", token.single_word)?;
        dict.set_item("lstrip", token.lstrip)?;
        dict.set_item("rstrip", token.rstrip)?;
        dict.set_item("normalized", token.normalized)?;

        Ok(dict)
    }
}

#[pymethods]
impl PyAddedToken {
    #[new]
    #[args(kwargs = "**")]
    fn new(content: Option<&str>, kwargs: Option<&PyDict>) -> PyResult<Self> {
        let mut token = PyAddedToken::from(content.unwrap_or(""), None);

        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "single_word" => token.single_word = Some(value.extract()?),
                    "lstrip" => token.lstrip = Some(value.extract()?),
                    "rstrip" => token.rstrip = Some(value.extract()?),
                    "normalized" => token.normalized = Some(value.extract()?),
                    _ => println!("Ignored unknown kwarg option {}", key),
                }
            }
        }

        Ok(token)
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<&'py PyDict> {
        self.as_pydict(py)
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyDict>(py) {
            Ok(state) => {
                for (key, value) in state {
                    let key: &str = key.extract()?;
                    match key {
                        "content" => self.content = value.extract()?,
                        "single_word" => self.single_word = Some(value.extract()?),
                        "lstrip" => self.lstrip = Some(value.extract()?),
                        "rstrip" => self.rstrip = Some(value.extract()?),
                        "normalized" => self.normalized = Some(value.extract()?),
                        _ => {}
                    }
                }
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    #[getter]
    fn get_content(&self) -> &str {
        &self.content
    }

    #[getter]
    fn get_rstrip(&self) -> bool {
        self.get_token().rstrip
    }

    #[getter]
    fn get_lstrip(&self) -> bool {
        self.get_token().lstrip
    }

    #[getter]
    fn get_single_word(&self) -> bool {
        self.get_token().single_word
    }

    #[getter]
    fn get_normalized(&self) -> bool {
        self.get_token().normalized
    }
}
#[pyproto]
impl PyObjectProtocol for PyAddedToken {
    fn __str__(&'p self) -> PyResult<&'p str> {
        Ok(&self.content)
    }

    fn __repr__(&self) -> PyResult<String> {
        let bool_to_python = |p| match p {
            true => "True",
            false => "False",
        };

        let token = self.get_token();
        Ok(format!(
            "AddedToken(\"{}\", rstrip={}, lstrip={}, single_word={}, normalized={})",
            self.content,
            bool_to_python(token.rstrip),
            bool_to_python(token.lstrip),
            bool_to_python(token.single_word),
            bool_to_python(token.normalized)
        ))
    }
}

struct TextInputSequence(tk::InputSequence);
impl FromPyObject<'_> for TextInputSequence {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let err = exceptions::ValueError::py_err("TextInputSequence must be str");
        if let Ok(s) = ob.downcast::<PyString>() {
            let seq: String = s.extract().map_err(|_| err)?;
            Ok(Self(seq.into()))
        } else {
            Err(err)
        }
    }
}
impl From<TextInputSequence> for tk::InputSequence {
    fn from(s: TextInputSequence) -> Self {
        s.0
    }
}

struct PreTokenizedInputSequence(tk::InputSequence);
impl FromPyObject<'_> for PreTokenizedInputSequence {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let err = exceptions::ValueError::py_err(
            "PreTokenizedInputSequence must be Union[List[str], Tuple[str]]",
        );

        if let Ok(s) = ob.downcast::<PyList>() {
            let seq = s.extract::<Vec<String>>().map_err(|_| err)?;
            Ok(Self(seq.into()))
        } else if let Ok(s) = ob.downcast::<PyTuple>() {
            let seq = s.extract::<Vec<String>>().map_err(|_| err)?;
            Ok(Self(seq.into()))
        } else {
            Err(err)
        }
    }
}
impl From<PreTokenizedInputSequence> for tk::InputSequence {
    fn from(s: PreTokenizedInputSequence) -> Self {
        s.0
    }
}

struct TextEncodeInput(tk::EncodeInput);
impl FromPyObject<'_> for TextEncodeInput {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let err = exceptions::ValueError::py_err(
            "TextEncodeInput must be Union[TextInputSequence, Tuple[InputSequence, InputSequence]]",
        );

        let gil = Python::acquire_gil();
        let py = gil.python();
        let obj = ob.to_object(py);

        if let Ok(i) = obj.extract::<TextInputSequence>(py) {
            Ok(Self(i.into()))
        } else if let Ok((i1, i2)) = obj.extract::<(TextInputSequence, TextInputSequence)>(py) {
            Ok(Self((i1, i2).into()))
        } else {
            Err(err)
        }
    }
}
impl From<TextEncodeInput> for tk::tokenizer::EncodeInput {
    fn from(i: TextEncodeInput) -> Self {
        i.0
    }
}
struct PreTokenizedEncodeInput(tk::EncodeInput);
impl FromPyObject<'_> for PreTokenizedEncodeInput {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let err = exceptions::ValueError::py_err(
            "PreTokenizedEncodeInput must be Union[PreTokenizedInputSequence, \
            Tuple[PreTokenizedInputSequence, PreTokenizedInputSequence]]",
        );

        let gil = Python::acquire_gil();
        let py = gil.python();
        let obj = ob.to_object(py);

        if let Ok(i) = obj.extract::<PreTokenizedInputSequence>(py) {
            Ok(Self(i.into()))
        } else if let Ok((i1, i2)) =
            obj.extract::<(PreTokenizedInputSequence, PreTokenizedInputSequence)>(py)
        {
            Ok(Self((i1, i2).into()))
        } else {
            Err(err)
        }
    }
}
impl From<PreTokenizedEncodeInput> for tk::tokenizer::EncodeInput {
    fn from(i: PreTokenizedEncodeInput) -> Self {
        i.0
    }
}

type TokenizerImpl = Tokenizer<PyModel>;

#[pyclass(dict, module = "tokenizers")]
pub struct PyTokenizer {
    tokenizer: TokenizerImpl,
}

impl PyTokenizer {
    fn new(tokenizer: TokenizerImpl) -> Self {
        PyTokenizer { tokenizer }
    }

    fn from_model(model: PyModel) -> Self {
        PyTokenizer::new(TokenizerImpl::new(model))
    }
}

#[pymethods]
impl PyTokenizer {
    #[new]
    fn __new__(model: PyRef<PyModel>) -> Self {
        PyTokenizer::from_model(model.clone())
    }

    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        let data = serde_json::to_string(&self.tokenizer).map_err(|e| {
            exceptions::Exception::py_err(format!(
                "Error while attempting to pickle Tokenizer: {}",
                e.to_string()
            ))
        })?;
        Ok(PyBytes::new(py, data.as_bytes()).to_object(py))
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                self.tokenizer = serde_json::from_slice(s.as_bytes()).map_err(|e| {
                    exceptions::Exception::py_err(format!(
                        "Error while attempting to unpickle Tokenizer: {}",
                        e.to_string()
                    ))
                })?;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    fn __getnewargs__<'p>(&self, py: Python<'p>) -> PyResult<&'p PyTuple> {
        let model: PyObject = PyModel::new(Arc::new(BPE::default())).into_py(py);
        let args = PyTuple::new(py, vec![model]);
        Ok(args)
    }

    #[staticmethod]
    fn from_str(s: &str) -> PyResult<Self> {
        let tokenizer: PyResult<_> = ToPyResult(s.parse()).into();
        Ok(Self::new(tokenizer?))
    }

    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        let tokenizer: PyResult<_> = ToPyResult(Tokenizer::from_file(path)).into();
        Ok(Self {
            tokenizer: tokenizer?,
        })
    }

    #[staticmethod]
    fn from_buffer(buffer: &PyBytes) -> PyResult<Self> {
        let tokenizer = serde_json::from_slice(buffer.as_bytes()).map_err(|e| {
            exceptions::Exception::py_err(format!(
                "Cannot instantiate Tokenizer from buffer: {}",
                e.to_string()
            ))
        })?;
        Ok(Self { tokenizer })
    }

    #[args(pretty = false)]
    fn to_str(&self, pretty: bool) -> PyResult<String> {
        ToPyResult(self.tokenizer.to_string(pretty)).into()
    }

    #[args(pretty = false)]
    fn save(&self, path: &str, pretty: bool) -> PyResult<()> {
        ToPyResult(self.tokenizer.save(path, pretty)).into()
    }

    fn num_special_tokens_to_add(&self, is_pair: bool) -> PyResult<usize> {
        Ok(self
            .tokenizer
            .get_post_processor()
            .map_or(0, |p| p.as_ref().added_tokens(is_pair)))
    }

    #[args(with_added_tokens = true)]
    fn get_vocab(&self, with_added_tokens: bool) -> PyResult<HashMap<String, u32>> {
        Ok(self.tokenizer.get_vocab(with_added_tokens))
    }

    #[args(with_added_tokens = true)]
    fn get_vocab_size(&self, with_added_tokens: bool) -> PyResult<usize> {
        Ok(self.tokenizer.get_vocab_size(with_added_tokens))
    }

    #[args(kwargs = "**")]
    fn enable_truncation(&mut self, max_length: usize, kwargs: Option<&PyDict>) -> PyResult<()> {
        let mut params = TruncationParams::default();
        params.max_length = max_length;

        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "stride" => params.stride = value.extract()?,
                    "strategy" => {
                        let value: &str = value.extract()?;
                        params.strategy = match value {
                            "longest_first" => Ok(TruncationStrategy::LongestFirst),
                            "only_first" => Ok(TruncationStrategy::OnlyFirst),
                            "only_second" => Ok(TruncationStrategy::OnlySecond),
                            _ => Err(PyError(format!(
                                "Unknown `strategy`: `{}`. Use \
                                 one of `longest_first`, `only_first`, or `only_second`",
                                value
                            ))
                            .into_pyerr()),
                        }?
                    }
                    _ => println!("Ignored unknown kwarg option {}", key),
                }
            }
        }

        self.tokenizer.with_truncation(Some(params));

        Ok(())
    }

    fn no_truncation(&mut self) {
        self.tokenizer.with_truncation(None);
    }

    #[getter]
    fn get_truncation<'py>(&self, py: Python<'py>) -> PyResult<Option<&'py PyDict>> {
        self.tokenizer.get_truncation().map_or(Ok(None), |params| {
            let dict = PyDict::new(py);

            dict.set_item("max_length", params.max_length)?;
            dict.set_item("stride", params.stride)?;
            dict.set_item("strategy", params.strategy.as_ref())?;

            Ok(Some(dict))
        })
    }

    #[args(kwargs = "**")]
    fn enable_padding(&mut self, kwargs: Option<&PyDict>) -> PyResult<()> {
        let mut params = PaddingParams::default();

        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "direction" => {
                        let value: &str = value.extract()?;
                        params.direction = match value {
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
                    "pad_to_multiple_of" => {
                        if let Some(multiple) = value.extract()? {
                            params.pad_to_multiple_of = multiple;
                        }
                    }
                    "pad_id" => params.pad_id = value.extract()?,
                    "pad_type_id" => params.pad_type_id = value.extract()?,
                    "pad_token" => params.pad_token = value.extract()?,
                    "max_length" => {
                        println!(
                            "enable_padding(max_length=X) is deprecated, \
                                 use enable_padding(length=X) instead"
                        );
                        if let Some(l) = value.extract()? {
                            params.strategy = PaddingStrategy::Fixed(l);
                        } else {
                            params.strategy = PaddingStrategy::BatchLongest;
                        }
                    }
                    "length" => {
                        if let Some(l) = value.extract()? {
                            params.strategy = PaddingStrategy::Fixed(l);
                        } else {
                            params.strategy = PaddingStrategy::BatchLongest;
                        }
                    }
                    _ => println!("Ignored unknown kwarg option {}", key),
                }
            }
        }

        self.tokenizer.with_padding(Some(params));

        Ok(())
    }

    fn no_padding(&mut self) {
        self.tokenizer.with_padding(None);
    }

    #[getter]
    fn get_padding<'py>(&self, py: Python<'py>) -> PyResult<Option<&'py PyDict>> {
        self.tokenizer.get_padding().map_or(Ok(None), |params| {
            let dict = PyDict::new(py);

            dict.set_item(
                "length",
                match params.strategy {
                    tk::PaddingStrategy::BatchLongest => None,
                    tk::PaddingStrategy::Fixed(size) => Some(size),
                },
            )?;
            dict.set_item("pad_to_multiple_of", params.pad_to_multiple_of)?;
            dict.set_item("pad_id", params.pad_id)?;
            dict.set_item("pad_token", &params.pad_token)?;
            dict.set_item("pad_type_id", params.pad_type_id)?;
            dict.set_item("direction", params.direction.as_ref())?;

            Ok(Some(dict))
        })
    }

    fn normalize(&self, sentence: &str) -> PyResult<String> {
        ToPyResult(
            self.tokenizer
                .normalize(sentence)
                .map(|s| s.get().to_owned()),
        )
        .into()
    }

    /// Input can be:
    /// encode("A single sequence")
    /// encode("A sequence", "And its pair")
    /// encode([ "A", "pre", "tokenized", "sequence" ], is_pretokenized=True)
    /// encode(
    ///     [ "A", "pre", "tokenized", "sequence" ], [ "And", "its", "pair" ],
    ///     is_pretokenized=True
    /// )
    #[args(pair = "None", is_pretokenized = "false", add_special_tokens = "true")]
    fn encode(
        &self,
        sequence: &PyAny,
        pair: Option<&PyAny>,
        is_pretokenized: bool,
        add_special_tokens: bool,
    ) -> PyResult<PyEncoding> {
        let sequence: tk::InputSequence = if is_pretokenized {
            sequence.extract::<PreTokenizedInputSequence>()?.into()
        } else {
            sequence.extract::<TextInputSequence>()?.into()
        };
        let input = match pair {
            Some(pair) => {
                let pair: tk::InputSequence = if is_pretokenized {
                    pair.extract::<PreTokenizedInputSequence>()?.into()
                } else {
                    pair.extract::<TextInputSequence>()?.into()
                };
                tk::EncodeInput::Dual(sequence, pair)
            }
            None => tk::EncodeInput::Single(sequence),
        };

        ToPyResult(
            self.tokenizer
                .encode(input, add_special_tokens)
                .map(|e| e.into()),
        )
        .into()
    }

    /// Input can be:
    /// encode_batch([
    ///   "A single sequence",
    ///   ("A tuple with a sequence", "And its pair"),
    ///   [ "A", "pre", "tokenized", "sequence" ],
    ///   ([ "A", "pre", "tokenized", "sequence" ], "And its pair")
    /// ])
    #[args(is_pretokenized = "false", add_special_tokens = "true")]
    fn encode_batch(
        &self,
        input: Vec<&PyAny>,
        is_pretokenized: bool,
        add_special_tokens: bool,
    ) -> PyResult<Vec<PyEncoding>> {
        let input: Vec<tk::EncodeInput> = input
            .into_iter()
            .map(|o| {
                let input: tk::EncodeInput = if is_pretokenized {
                    o.extract::<PreTokenizedEncodeInput>()?.into()
                } else {
                    o.extract::<TextEncodeInput>()?.into()
                };
                Ok(input)
            })
            .collect::<PyResult<Vec<tk::EncodeInput>>>()?;
        let gil = Python::acquire_gil();
        gil.python().allow_threads(|| {
            ToPyResult(
                self.tokenizer
                    .encode_batch(input, add_special_tokens)
                    .map(|encodings| encodings.into_iter().map(|e| e.into()).collect()),
            )
            .into()
        })
    }

    fn decode(&self, ids: Vec<u32>, skip_special_tokens: Option<bool>) -> PyResult<String> {
        ToPyResult(
            self.tokenizer
                .decode(ids, skip_special_tokens.unwrap_or(true)),
        )
        .into()
    }

    fn decode_batch(
        &self,
        sentences: Vec<Vec<u32>>,
        skip_special_tokens: Option<bool>,
    ) -> PyResult<Vec<String>> {
        let gil = Python::acquire_gil();
        gil.python().allow_threads(|| {
            ToPyResult(
                self.tokenizer
                    .decode_batch(sentences, skip_special_tokens.unwrap_or(true)),
            )
            .into()
        })
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<&str> {
        self.tokenizer.id_to_token(id)
    }

    fn add_tokens(&mut self, tokens: &PyList) -> PyResult<usize> {
        let tokens = tokens
            .into_iter()
            .map(|token| {
                if let Ok(content) = token.extract::<String>() {
                    Ok(PyAddedToken::from(content, Some(false)).get_token())
                } else if let Ok(mut token) = token.extract::<PyRefMut<PyAddedToken>>() {
                    token.is_special_token = false;
                    Ok(token.get_token())
                } else {
                    Err(exceptions::Exception::py_err(
                        "Input must be a List[Union[str, AddedToken]]",
                    ))
                }
            })
            .collect::<PyResult<Vec<_>>>()?;

        Ok(self.tokenizer.add_tokens(&tokens))
    }

    fn add_special_tokens(&mut self, tokens: &PyList) -> PyResult<usize> {
        let tokens = tokens
            .into_iter()
            .map(|token| {
                if let Ok(content) = token.extract::<String>() {
                    Ok(tk::tokenizer::AddedToken::from(content, true))
                } else if let Ok(mut token) = token.extract::<PyRefMut<PyAddedToken>>() {
                    token.is_special_token = true;
                    Ok(token.get_token())
                } else {
                    Err(exceptions::Exception::py_err(
                        "Input must be a List[Union[str, AddedToken]]",
                    ))
                }
            })
            .collect::<PyResult<Vec<_>>>()?;

        Ok(self.tokenizer.add_special_tokens(&tokens))
    }

    fn train(&mut self, _trainer: &PyTrainer, _files: Vec<String>) -> PyResult<()> {
        // TODO enable training once Tokenizer derives Clone
        // self.tokenizer = self.tokenizer.clone().train(trainer, files).map_err(|e|
        //    exceptions::Exception::py_err(format!("{}", e))
        // )?;
        // Ok(())
        Err(exceptions::NotImplementedError::py_err(
            "Training currently disabled",
        ))
    }

    #[args(pair = "None", add_special_tokens = true)]
    fn post_process(
        &self,
        encoding: &PyEncoding,
        pair: Option<&PyEncoding>,
        add_special_tokens: bool,
    ) -> PyResult<PyEncoding> {
        ToPyResult(
            self.tokenizer
                .post_process(
                    encoding.encoding.clone(),
                    pair.map(|p| p.encoding.clone()),
                    add_special_tokens,
                )
                .map(|e| e.into()),
        )
        .into()
    }

    #[getter]
    fn get_model(&self) -> PyModel {
        self.tokenizer.get_model().clone()
    }

    #[setter]
    fn set_model(&mut self, model: PyRef<PyModel>) {
        self.tokenizer.with_model(model.clone());
    }

    #[getter]
    fn get_normalizer(&self) -> PyResult<Option<Normalizer>> {
        Ok(self
            .tokenizer
            .get_normalizer()
            .map(|normalizer| Normalizer {
                normalizer: Container::from_ref(normalizer),
            }))
    }

    #[setter]
    fn set_normalizer(&mut self, mut normalizer: PyRefMut<Normalizer>) -> PyResult<()> {
        if let Some(normalizer) = normalizer.normalizer.to_pointer() {
            self.tokenizer.with_normalizer(normalizer);
            Ok(())
        } else {
            Err(exceptions::Exception::py_err(
                "The Normalizer is already being used in another Tokenizer",
            ))
        }
    }

    #[getter]
    fn get_pre_tokenizer(&self) -> PyResult<Option<PreTokenizer>> {
        Ok(self
            .tokenizer
            .get_pre_tokenizer()
            .map(|pretok| PreTokenizer {
                pretok: Container::from_ref(pretok),
            }))
    }

    #[setter]
    fn set_pre_tokenizer(&mut self, mut pretok: PyRefMut<PreTokenizer>) -> PyResult<()> {
        if let Some(pretok) = pretok.pretok.to_pointer() {
            self.tokenizer.with_pre_tokenizer(pretok);
            Ok(())
        } else {
            Err(exceptions::Exception::py_err(
                "The PreTokenizer is already being used in another Tokenizer",
            ))
        }
    }

    #[getter]
    fn get_post_processor(&self) -> PyResult<Option<PostProcessor>> {
        Ok(self
            .tokenizer
            .get_post_processor()
            .map(|processor| PostProcessor {
                processor: Container::from_ref(processor),
            }))
    }

    #[setter]
    fn set_post_processor(&mut self, mut processor: PyRefMut<PostProcessor>) -> PyResult<()> {
        if let Some(processor) = processor.processor.to_pointer() {
            self.tokenizer.with_post_processor(processor);
            Ok(())
        } else {
            Err(exceptions::Exception::py_err(
                "The Processor is already being used in another Tokenizer",
            ))
        }
    }

    #[getter]
    fn get_decoder(&self) -> PyResult<Option<Decoder>> {
        Ok(self.tokenizer.get_decoder().map(|decoder| Decoder {
            decoder: Container::from_ref(decoder),
        }))
    }

    #[setter]
    fn set_decoder(&mut self, mut decoder: PyRefMut<Decoder>) -> PyResult<()> {
        if let Some(decoder) = decoder.decoder.to_pointer() {
            self.tokenizer.with_decoder(decoder);
            Ok(())
        } else {
            Err(exceptions::Exception::py_err(
                "The Decoder is already being used in another Tokenizer",
            ))
        }
    }
}
