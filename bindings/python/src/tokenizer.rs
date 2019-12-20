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

    fn get_vocab_size(&self, with_added_tokens: bool) -> usize {
        self.tokenizer.get_vocab_size(with_added_tokens)
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

    fn with_truncation(
        &mut self,
        max_length: usize,
        stride: usize,
        strategy: &str,
    ) -> PyResult<()> {
        let strategy = match strategy {
            "longest_first" => Ok(TruncationStrategy::LongestFirst),
            "only_first" => Ok(TruncationStrategy::OnlyFirst),
            "only_second" => Ok(TruncationStrategy::OnlySecond),
            other => Err(PyError(format!(
                "Unknown `strategy`: `{}`. Use \
                 one of `longest_first`, `only_first`, or `only_second`",
                other
            ))
            .into_pyerr()),
        }?;

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

    fn with_padding(
        &mut self,
        size: Option<usize>,
        direction: &str,
        pad_id: u32,
        pad_type_id: u32,
        pad_token: &str,
    ) -> PyResult<()> {
        let strategy = if let Some(size) = size {
            PaddingStrategy::Fixed(size)
        } else {
            PaddingStrategy::BatchLongest
        };
        let direction = match direction {
            "left" => Ok(PaddingDirection::Left),
            "right" => Ok(PaddingDirection::Right),
            other => Err(PyError(format!(
                "Unknown `direction`: `{}`. Use \
                 one of `left` or `right`",
                other
            ))
            .into_pyerr()),
        }?;

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
