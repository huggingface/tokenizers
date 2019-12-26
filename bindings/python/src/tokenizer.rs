extern crate tokenizers as tk;

use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;

use super::decoders::Decoder;
use super::encoding::Encoding;
use super::error::{PyError, ToPyResult};
use super::models::Model;
use super::pre_tokenizers::PreTokenizer;
use super::processors::PostProcessor;
use super::trainers::Trainer;

use tk::tokenizer::{
    PaddingDirection, PaddingParams, PaddingStrategy, TruncationParams, TruncationStrategy,
};

#[pyclass(dict)]
pub struct Tokenizer {
    tokenizer: tk::tokenizer::Tokenizer,
}

#[pymethods]
impl Tokenizer {
    #[new]
    fn new(obj: &PyRawObject, model: &mut Model) -> PyResult<()> {
        if let Some(model) = model.model.to_pointer() {
            let tokenizer = tk::tokenizer::Tokenizer::new(model);
            obj.init({ Tokenizer { tokenizer } });
            Ok(())
        } else {
            Err(exceptions::Exception::py_err(
                "The Model is already being used in another Tokenizer",
            ))
        }
    }

    #[args(kwargs = "**")]
    fn get_vocab_size(&self, kwargs: Option<&PyDict>) -> PyResult<usize> {
        let mut with_added_tokens = true;

        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "with_added_tokens" => with_added_tokens = value.extract()?,
                    _ => println!("Ignored unknown kwarg option {}", key),
                }
            }
        }

        Ok(self.tokenizer.get_vocab_size(with_added_tokens))
    }

    fn with_model(&mut self, model: &mut Model) -> PyResult<()> {
        if let Some(model) = model.model.to_pointer() {
            self.tokenizer.with_model(model);
            Ok(())
        } else {
            Err(exceptions::Exception::py_err(
                "The Model is already being used in another Tokenizer",
            ))
        }
    }

    fn with_pre_tokenizer(&mut self, pretok: &mut PreTokenizer) -> PyResult<()> {
        if let Some(pretok) = pretok.pretok.to_pointer() {
            self.tokenizer.with_pre_tokenizer(pretok);
            Ok(())
        } else {
            Err(exceptions::Exception::py_err(
                "The PreTokenizer is already being used in another Tokenizer",
            ))
        }
    }

    fn with_decoder(&mut self, decoder: &mut Decoder) -> PyResult<()> {
        if let Some(decoder) = decoder.decoder.to_pointer() {
            self.tokenizer.with_decoder(decoder);
            Ok(())
        } else {
            Err(exceptions::Exception::py_err(
                "The Decoder is already being used in another Tokenizer",
            ))
        }
    }

    fn with_post_processor(&mut self, processor: &mut PostProcessor) -> PyResult<()> {
        if let Some(processor) = processor.processor.to_pointer() {
            self.tokenizer.with_post_processor(processor);
            Ok(())
        } else {
            Err(exceptions::Exception::py_err(
                "The Processor is already being used in another Tokenizer",
            ))
        }
    }

    #[args(kwargs = "**")]
    fn with_truncation(&mut self, max_length: usize, kwargs: Option<&PyDict>) -> PyResult<()> {
        let mut stride = 0;
        let mut strategy = TruncationStrategy::LongestFirst;

        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "stride" => stride = value.extract()?,
                    "strategy" => {
                        let value: &str = value.extract()?;
                        strategy = match value {
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

        self.tokenizer.with_truncation(Some(TruncationParams {
            max_length,
            stride,
            strategy,
        }));

        Ok(())
    }

    fn without_truncation(&mut self) {
        self.tokenizer.with_truncation(None);
    }

    #[args(kwargs = "**")]
    fn with_padding(&mut self, kwargs: Option<&PyDict>) -> PyResult<()> {
        let mut direction = PaddingDirection::Right;
        let mut pad_id: u32 = 0;
        let mut pad_type_id: u32 = 0;
        let mut pad_token = String::from("[PAD]");
        let mut max_length: Option<usize> = None;

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
                    "max_length" => max_length = value.extract()?,
                    _ => println!("Ignored unknown kwarg option {}", key),
                }
            }
        }

        let strategy = if let Some(max_length) = max_length {
            PaddingStrategy::Fixed(max_length)
        } else {
            PaddingStrategy::BatchLongest
        };

        self.tokenizer.with_padding(Some(PaddingParams {
            strategy,
            direction,
            pad_id,
            pad_type_id,
            pad_token: pad_token.to_owned(),
        }));

        Ok(())
    }

    fn without_padding(&mut self) {
        self.tokenizer.with_padding(None);
    }

    fn encode(&self, sentence: &str, pair: Option<&str>) -> PyResult<Encoding> {
        ToPyResult(
            self.tokenizer
                .encode(if let Some(pair) = pair {
                    tk::tokenizer::EncodeInput::Dual(sentence.to_owned(), pair.to_owned())
                } else {
                    tk::tokenizer::EncodeInput::Single(sentence.to_owned())
                })
                .map(Encoding::new),
        )
        .into()
    }

    fn encode_batch(&self, sentences: &PyList) -> PyResult<Vec<Encoding>> {
        let inputs = sentences
            .into_iter()
            .map(|item| {
                if let Ok(s1) = item.extract::<String>() {
                    Ok(tk::tokenizer::EncodeInput::Single(s1))
                } else if let Ok((s1, s2)) = item.extract::<(String, String)>() {
                    Ok(tk::tokenizer::EncodeInput::Dual(s1, s2))
                } else {
                    Err(exceptions::Exception::py_err(
                        "Input must be a list[str] or list[(str, str)]",
                    ))
                }
            })
            .collect::<PyResult<Vec<_>>>()?;

        ToPyResult(
            self.tokenizer
                .encode_batch(inputs)
                .map(|encodings| encodings.into_iter().map(Encoding::new).collect()),
        )
        .into()
    }

    fn decode(&self, ids: Vec<u32>, skip_special_tokens: bool) -> PyResult<String> {
        ToPyResult(self.tokenizer.decode(ids, skip_special_tokens)).into()
    }

    fn decode_batch(
        &self,
        sentences: Vec<Vec<u32>>,
        skip_special_tokens: bool,
    ) -> PyResult<Vec<String>> {
        ToPyResult(self.tokenizer.decode_batch(sentences, skip_special_tokens)).into()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }

    fn add_tokens(&mut self, tokens: &PyList) -> PyResult<usize> {
        let tokens = tokens
            .into_iter()
            .map(|token| {
                if let Ok(content) = token.extract::<String>() {
                    Ok(tk::tokenizer::AddedToken {
                        content,
                        ..Default::default()
                    })
                } else if let Ok((content, single_word)) = token.extract::<(String, bool)>() {
                    Ok(tk::tokenizer::AddedToken {
                        content,
                        single_word,
                    })
                } else {
                    Err(exceptions::Exception::py_err(
                        "Input must be a list[str] or list[(str, bool)]",
                    ))
                }
            })
            .collect::<PyResult<Vec<_>>>()?;

        Ok(self.tokenizer.add_tokens(&tokens))
    }

    fn add_special_tokens(&mut self, tokens: Vec<&str>) -> PyResult<usize> {
        Ok(self.tokenizer.add_special_tokens(&tokens))
    }

    fn train(&mut self, trainer: &Trainer, files: Vec<String>) -> PyResult<()> {
        trainer.trainer.execute(|trainer| {
            if let Err(e) = self.tokenizer.train(trainer, files) {
                Err(exceptions::Exception::py_err(format!("{}", e)))
            } else {
                Ok(())
            }
        })
    }
}
