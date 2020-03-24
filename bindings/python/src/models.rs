extern crate tokenizers as tk;

use super::encoding::Encoding;
use super::error::ToPyResult;
use super::utils::Container;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;
use rayon::prelude::*;
use std::path::Path;

#[pyclass]
struct EncodeInput {
    sequence: Vec<(String, (usize, usize))>,
}
impl EncodeInput {
    pub fn into_input(self) -> Vec<(String, (usize, usize))> {
        self.sequence
    }
}

impl<'source> FromPyObject<'source> for EncodeInput {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let sequence: &PyList = ob.downcast_ref()?;

        enum Mode {
            NoOffsets,
            Offsets,
        };
        let mode = sequence
            .iter()
            .next()
            .map(|item| {
                if item.extract::<String>().is_ok() {
                    Ok(Mode::NoOffsets)
                } else if item.extract::<(String, (usize, usize))>().is_ok() {
                    Ok(Mode::Offsets)
                } else {
                    Err(exceptions::ValueError::py_err(
                        "Input must be a list[str] or list[(str, (int, int))]",
                    ))
                }
            })
            .unwrap()?;

        let mut total_len = 0;
        let sequence = sequence
            .iter()
            .enumerate()
            .map(|(i, item)| match mode {
                Mode::NoOffsets => item
                    .extract::<String>()
                    .map_err(|_| {
                        exceptions::ValueError::py_err(format!(
                            "Value at index {} should be a `str`",
                            i
                        ))
                    })
                    .map(|s| {
                        let len = s.chars().count();
                        total_len += len;
                        (s, (total_len - len, total_len))
                    }),
                Mode::Offsets => item.extract::<(String, (usize, usize))>().map_err(|_| {
                    exceptions::ValueError::py_err(format!(
                        "Value at index {} should be a `(str, (int, int))`",
                        i
                    ))
                }),
            })
            .collect::<Result<Vec<_>, PyErr>>()?;

        Ok(EncodeInput { sequence })
    }
}

/// A Model represents some tokenization algorithm like BPE or Word
/// This class cannot be constructed directly. Please use one of the concrete models.
#[pyclass]
pub struct Model {
    pub model: Container<dyn tk::tokenizer::Model + Sync>,
}

#[pymethods]
impl Model {
    #[new]
    fn new(_obj: &PyRawObject) -> PyResult<()> {
        Err(exceptions::Exception::py_err(
            "Cannot create a Model directly. Use a concrete subclass",
        ))
    }

    fn save(&self, folder: &str, name: Option<&str>) -> PyResult<Vec<String>> {
        let saved: PyResult<Vec<_>> = ToPyResult(
            self.model
                .execute(|model| model.save(Path::new(folder), name)),
        )
        .into();

        Ok(saved?
            .into_iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect())
    }

    #[args(type_id = 0)]
    fn encode(&self, sequence: EncodeInput, type_id: u32) -> PyResult<Encoding> {
        let sequence = sequence.into_input();

        if sequence.is_empty() {
            return Ok(Encoding::new(tk::tokenizer::Encoding::default()));
        }

        ToPyResult(self.model.execute(|model| {
            model
                .tokenize(sequence)
                .map(|tokens| Encoding::new(tk::tokenizer::Encoding::from_tokens(tokens, type_id)))
        }))
        .into()
    }

    #[args(type_id = 0)]
    fn encode_batch(&self, sequences: Vec<EncodeInput>, type_id: u32) -> PyResult<Vec<Encoding>> {
        ToPyResult(self.model.execute(|model| {
            sequences
                .into_par_iter()
                .map(|sequence| {
                    let sequence = sequence.into_input();
                    if sequence.is_empty() {
                        Ok(Encoding::new(tk::tokenizer::Encoding::default()))
                    } else {
                        model.tokenize(sequence).map(|tokens| {
                            Encoding::new(tk::tokenizer::Encoding::from_tokens(tokens, type_id))
                        })
                    }
                })
                .collect::<Result<_, _>>()
        }))
        .into()
    }
}

/// BPE Model
/// Allows the creation of a BPE Model to be used with a Tokenizer
#[pyclass]
pub struct BPE {}

#[pymethods]
impl BPE {
    /// from_files(vocab, merges, /)
    /// --
    ///
    /// Instanciate a new BPE model using the provided vocab and merges files
    #[staticmethod]
    #[args(kwargs = "**")]
    fn from_files(vocab: &str, merges: &str, kwargs: Option<&PyDict>) -> PyResult<Model> {
        let mut builder = tk::models::bpe::BPE::from_files(vocab, merges);
        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "cache_capacity" => builder = builder.cache_capacity(value.extract()?),
                    "dropout" => {
                        if let Some(dropout) = value.extract()? {
                            builder = builder.dropout(dropout);
                        }
                    }
                    "unk_token" => {
                        if let Some(unk) = value.extract()? {
                            builder = builder.unk_token(unk);
                        }
                    }
                    "continuing_subword_prefix" => {
                        builder = builder.continuing_subword_prefix(value.extract()?)
                    }
                    "end_of_word_suffix" => builder = builder.end_of_word_suffix(value.extract()?),
                    _ => println!("Ignored unknown kwarg option {}", key),
                };
            }
        }

        match builder.build() {
            Err(e) => Err(exceptions::Exception::py_err(format!(
                "Error while initializing BPE: {}",
                e
            ))),
            Ok(bpe) => Ok(Model {
                model: Container::Owned(Box::new(bpe)),
            }),
        }
    }

    /// empty()
    /// --
    ///
    /// Instanciate a new BPE model with empty vocab and merges
    #[staticmethod]
    fn empty() -> Model {
        Model {
            model: Container::Owned(Box::new(tk::models::bpe::BPE::default())),
        }
    }
}

/// WordPiece Model
#[pyclass]
pub struct WordPiece {}

#[pymethods]
impl WordPiece {
    /// from_files(vocab, /)
    /// --
    ///
    /// Instantiate a new WordPiece model using the provided vocabulary file
    #[staticmethod]
    #[args(kwargs = "**")]
    fn from_files(vocab: &str, kwargs: Option<&PyDict>) -> PyResult<Model> {
        let mut builder = tk::models::wordpiece::WordPiece::from_files(vocab);

        if let Some(kwargs) = kwargs {
            for (key, val) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "unk_token" => {
                        builder = builder.unk_token(val.extract()?);
                    }
                    "max_input_chars_per_word" => {
                        builder = builder.max_input_chars_per_word(val.extract()?);
                    }
                    "continuing_subword_prefix" => {
                        builder = builder.continuing_subword_prefix(val.extract()?);
                    }
                    _ => println!("Ignored unknown kwargs option {}", key),
                }
            }
        }

        match builder.build() {
            Err(e) => {
                println!("Errors: {:?}", e);
                Err(exceptions::Exception::py_err(
                    "Error while initializing WordPiece",
                ))
            }
            Ok(wordpiece) => Ok(Model {
                model: Container::Owned(Box::new(wordpiece)),
            }),
        }
    }

    #[staticmethod]
    fn empty() -> Model {
        Model {
            model: Container::Owned(Box::new(tk::models::wordpiece::WordPiece::default())),
        }
    }
}

#[pyclass]
pub struct WordLevel {}

#[pymethods]
impl WordLevel {
    #[staticmethod]
    #[args(kwargs = "**")]
    fn from_files(vocab: &str, kwargs: Option<&PyDict>) -> PyResult<Model> {
        let mut unk_token = String::from("<unk>");

        if let Some(kwargs) = kwargs {
            for (key, val) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "unk_token" => unk_token = val.extract()?,
                    _ => println!("Ignored unknown kwargs option {}", key),
                }
            }
        }

        match tk::models::wordlevel::WordLevel::from_files(vocab, unk_token) {
            Err(e) => {
                println!("Errors: {:?}", e);
                Err(exceptions::Exception::py_err(
                    "Error while initializing WordLevel",
                ))
            }
            Ok(model) => Ok(Model {
                model: Container::Owned(Box::new(model)),
            }),
        }
    }
}
