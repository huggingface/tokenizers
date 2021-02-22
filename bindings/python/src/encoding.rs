use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;
use pyo3::{PyObjectProtocol, PySequenceProtocol};
use tk::tokenizer::{Offsets, PaddingDirection};
use tokenizers as tk;

use crate::error::{deprecation_warning, PyError};

/// The :class:`~tokenizers.Encoding` represents the output of a :class:`~tokenizers.Tokenizer`.
#[pyclass(dict, module = "tokenizers", name=Encoding)]
#[repr(transparent)]
pub struct PyEncoding {
    pub encoding: tk::tokenizer::Encoding,
}

impl From<tk::tokenizer::Encoding> for PyEncoding {
    fn from(v: tk::tokenizer::Encoding) -> Self {
        Self { encoding: v }
    }
}

#[pyproto]
impl PyObjectProtocol for PyEncoding {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "Encoding(num_tokens={}, attributes=[ids, type_ids, tokens, offsets, \
             attention_mask, special_tokens_mask, overflowing])",
            self.encoding.get_ids().len()
        ))
    }
}

#[pyproto]
impl PySequenceProtocol for PyEncoding {
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.encoding.len())
    }
}

#[pymethods]
impl PyEncoding {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self {
            encoding: tk::tokenizer::Encoding::default(),
        })
    }

    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        let data = serde_json::to_string(&self.encoding).map_err(|e| {
            exceptions::PyException::new_err(format!(
                "Error while attempting to pickle Encoding: {}",
                e.to_string()
            ))
        })?;
        Ok(PyBytes::new(py, data.as_bytes()).to_object(py))
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                self.encoding = serde_json::from_slice(s.as_bytes()).map_err(|e| {
                    exceptions::PyException::new_err(format!(
                        "Error while attempting to unpickle Encoding: {}",
                        e.to_string()
                    ))
                })?;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// Merge the list of encodings into one final :class:`~tokenizers.Encoding`
    ///
    /// Args:
    ///     encodings (A :obj:`List` of :class:`~tokenizers.Encoding`):
    ///         The list of encodings that should be merged in one
    ///
    ///     growing_offsets (:obj:`bool`, defaults to :obj:`True`):
    ///         Whether the offsets should accumulate while merging
    ///
    /// Returns:
    ///     :class:`~tokenizers.Encoding`: The resulting Encoding
    #[staticmethod]
    #[args(growing_offsets = true)]
    #[text_signature = "(encodings, growing_offsets=True)"]
    fn merge(encodings: Vec<PyRef<PyEncoding>>, growing_offsets: bool) -> PyEncoding {
        tk::tokenizer::Encoding::merge(
            encodings.into_iter().map(|e| e.encoding.clone()),
            growing_offsets,
        )
        .into()
    }

    /// The number of sequences represented
    ///
    /// Returns:
    ///     :obj:`int`: The number of sequences in this :class:`~tokenizers.Encoding`
    #[getter]
    fn get_n_sequences(&self) -> usize {
        self.encoding.n_sequences()
    }

    /// Set the given sequence index
    ///
    /// Set the given sequence index for the whole range of tokens contained in this
    /// :class:`~tokenizers.Encoding`.
    #[text_signature = "(self, sequence_id)"]
    fn set_sequence_id(&mut self, sequence_id: usize) {
        self.encoding.set_sequence_id(sequence_id);
    }

    /// The generated IDs
    ///
    /// The IDs are the main input to a Language Model. They are the token indices,
    /// the numerical representations that a LM understands.
    ///
    /// Returns:
    ///     :obj:`List[int]`: The list of IDs
    #[getter]
    fn get_ids(&self) -> Vec<u32> {
        self.encoding.get_ids().to_vec()
    }

    /// The generated tokens
    ///
    /// They are the string representation of the IDs.
    ///
    /// Returns:
    ///     :obj:`List[str]`: The list of tokens
    #[getter]
    fn get_tokens(&self) -> Vec<String> {
        self.encoding.get_tokens().to_vec()
    }

    /// The generated word indices.
    ///
    /// .. warning::
    ///     This is deprecated and will be removed in a future version.
    ///     Please use :obj:`~tokenizers.Encoding.word_ids` instead.
    ///
    /// They represent the index of the word associated to each token.
    /// When the input is pre-tokenized, they correspond to the ID of the given input label,
    /// otherwise they correspond to the words indices as defined by the
    /// :class:`~tokenizers.pre_tokenizers.PreTokenizer` that was used.
    ///
    /// For special tokens and such (any token that was generated from something that was
    /// not part of the input), the output is :obj:`None`
    ///
    /// Returns:
    ///     A :obj:`List` of :obj:`Optional[int]`: A list of optional word index.
    #[getter]
    fn get_words(&self) -> PyResult<Vec<Option<u32>>> {
        deprecation_warning(
            "0.9.4",
            "Encoding.words is deprecated, please use Encoding.word_ids instead.",
        )?;
        Ok(self.get_word_ids())
    }

    /// The generated word indices.
    ///
    /// They represent the index of the word associated to each token.
    /// When the input is pre-tokenized, they correspond to the ID of the given input label,
    /// otherwise they correspond to the words indices as defined by the
    /// :class:`~tokenizers.pre_tokenizers.PreTokenizer` that was used.
    ///
    /// For special tokens and such (any token that was generated from something that was
    /// not part of the input), the output is :obj:`None`
    ///
    /// Returns:
    ///     A :obj:`List` of :obj:`Optional[int]`: A list of optional word index.
    #[getter]
    fn get_word_ids(&self) -> Vec<Option<u32>> {
        self.encoding.get_word_ids().to_vec()
    }

    /// The generated sequence indices.
    ///
    /// They represent the index of the input sequence associated to each token.
    /// The sequence id can be None if the token is not related to any input sequence,
    /// like for example with special tokens.
    ///
    /// Returns:
    ///     A :obj:`List` of :obj:`Optional[int]`: A list of optional sequence index.
    #[getter]
    fn get_sequence_ids(&self) -> Vec<Option<usize>> {
        self.encoding.get_sequence_ids()
    }

    /// The generated type IDs
    ///
    /// Generally used for tasks like sequence classification or question answering,
    /// these tokens let the LM know which input sequence corresponds to each tokens.
    ///
    /// Returns:
    ///     :obj:`List[int]`: The list of type ids
    #[getter]
    fn get_type_ids(&self) -> Vec<u32> {
        self.encoding.get_type_ids().to_vec()
    }

    /// The offsets associated to each token
    ///
    /// These offsets let's you slice the input string, and thus retrieve the original
    /// part that led to producing the corresponding token.
    ///
    /// Returns:
    ///     A :obj:`List` of :obj:`Tuple[int, int]`: The list of offsets
    #[getter]
    fn get_offsets(&self) -> Vec<(usize, usize)> {
        self.encoding.get_offsets().to_vec()
    }

    /// The special token mask
    ///
    /// This indicates which tokens are special tokens, and which are not.
    ///
    /// Returns:
    ///     :obj:`List[int]`: The special tokens mask
    #[getter]
    fn get_special_tokens_mask(&self) -> Vec<u32> {
        self.encoding.get_special_tokens_mask().to_vec()
    }

    /// The attention mask
    ///
    /// This indicates to the LM which tokens should be attended to, and which should not.
    /// This is especially important when batching sequences, where we need to applying
    /// padding.
    ///
    /// Returns:
    ///    :obj:`List[int]`: The attention mask
    #[getter]
    fn get_attention_mask(&self) -> Vec<u32> {
        self.encoding.get_attention_mask().to_vec()
    }

    /// A :obj:`List` of overflowing :class:`~tokenizers.Encoding`
    ///
    /// When using truncation, the :class:`~tokenizers.Tokenizer` takes care of splitting
    /// the output into as many pieces as required to match the specified maximum length.
    /// This field lets you retrieve all the subsequent pieces.
    ///
    /// When you use pairs of sequences, the overflowing pieces will contain enough
    /// variations to cover all the possible combinations, while respecting the provided
    /// maximum length.
    #[getter]
    fn get_overflowing(&self) -> Vec<PyEncoding> {
        self.encoding
            .get_overflowing()
            .clone()
            .into_iter()
            .map(|e| e.into())
            .collect()
    }

    /// Get the encoded tokens corresponding to the word at the given index
    /// in one of the input sequences.
    ///
    /// Args:
    ///     word_index (:obj:`int`):
    ///         The index of a word in one of the input sequences.
    ///     sequence_index (:obj:`int`, defaults to :obj:`0`):
    ///         The index of the sequence that contains the target word
    ///
    /// Returns:
    ///     :obj:`Tuple[int, int]`: The range of tokens: :obj:`(first, last + 1)`
    #[args(sequence_index = 0)]
    #[text_signature = "(self, word_index, sequence_index=0)"]
    fn word_to_tokens(&self, word_index: u32, sequence_index: usize) -> Option<(usize, usize)> {
        self.encoding.word_to_tokens(word_index, sequence_index)
    }

    /// Get the offsets of the word at the given index in one of the input sequences.
    ///
    /// Args:
    ///     word_index (:obj:`int`):
    ///         The index of a word in one of the input sequences.
    ///     sequence_index (:obj:`int`, defaults to :obj:`0`):
    ///         The index of the sequence that contains the target word
    ///
    /// Returns:
    ///     :obj:`Tuple[int, int]`: The range of characters (span) :obj:`(first, last + 1)`
    #[args(sequence_index = 0)]
    #[text_signature = "(self, word_index, sequence_index=0)"]
    fn word_to_chars(&self, word_index: u32, sequence_index: usize) -> Option<Offsets> {
        self.encoding.word_to_chars(word_index, sequence_index)
    }

    /// Get the index of the sequence represented by the given token.
    ///
    /// In the general use case, this method returns :obj:`0` for a single sequence or
    /// the first sequence of a pair, and :obj:`1` for the second sequence of a pair
    ///
    /// Args:
    ///     token_index (:obj:`int`):
    ///         The index of a token in the encoded sequence.
    ///
    /// Returns:
    ///     :obj:`int`: The sequence id of the given token
    #[text_signature = "(self, token_index)"]
    fn token_to_sequence(&self, token_index: usize) -> Option<usize> {
        self.encoding.token_to_sequence(token_index)
    }

    /// Get the offsets of the token at the given index.
    ///
    /// The returned offsets are related to the input sequence that contains the
    /// token.  In order to determine in which input sequence it belongs, you
    /// must call :meth:`~tokenizers.Encoding.token_to_sequence()`.
    ///
    /// Args:
    ///     token_index (:obj:`int`):
    ///         The index of a token in the encoded sequence.
    ///
    /// Returns:
    ///     :obj:`Tuple[int, int]`: The token offsets :obj:`(first, last + 1)`
    #[text_signature = "(self, token_index)"]
    fn token_to_chars(&self, token_index: usize) -> Option<Offsets> {
        let (_, offsets) = self.encoding.token_to_chars(token_index)?;
        Some(offsets)
    }

    /// Get the index of the word that contains the token in one of the input sequences.
    ///
    /// The returned word index is related to the input sequence that contains
    /// the token.  In order to determine in which input sequence it belongs, you
    /// must call :meth:`~tokenizers.Encoding.token_to_sequence()`.
    ///
    /// Args:
    ///     token_index (:obj:`int`):
    ///         The index of a token in the encoded sequence.
    ///
    /// Returns:
    ///     :obj:`int`: The index of the word in the relevant input sequence.
    #[text_signature = "(self, token_index)"]
    fn token_to_word(&self, token_index: usize) -> Option<u32> {
        let (_, word_idx) = self.encoding.token_to_word(token_index)?;
        Some(word_idx)
    }

    /// Get the token that contains the char at the given position in the input sequence.
    ///
    /// Args:
    ///     char_pos (:obj:`int`):
    ///         The position of a char in the input string
    ///     sequence_index (:obj:`int`, defaults to :obj:`0`):
    ///         The index of the sequence that contains the target char
    ///
    /// Returns:
    ///     :obj:`int`: The index of the token that contains this char in the encoded sequence
    #[args(sequence_index = 0)]
    #[text_signature = "(self, char_pos, sequence_index=0)"]
    fn char_to_token(&self, char_pos: usize, sequence_index: usize) -> Option<usize> {
        self.encoding.char_to_token(char_pos, sequence_index)
    }

    /// Get the word that contains the char at the given position in the input sequence.
    ///
    /// Args:
    ///     char_pos (:obj:`int`):
    ///         The position of a char in the input string
    ///     sequence_index (:obj:`int`, defaults to :obj:`0`):
    ///         The index of the sequence that contains the target char
    ///
    /// Returns:
    ///     :obj:`int`: The index of the word that contains this char in the input sequence
    #[args(sequence_index = 0)]
    #[text_signature = "(self, char_pos, sequence_index=0)"]
    fn char_to_word(&self, char_pos: usize, sequence_index: usize) -> Option<u32> {
        self.encoding.char_to_word(char_pos, sequence_index)
    }

    /// Pad the :class:`~tokenizers.Encoding` at the given length
    ///
    /// Args:
    ///     length (:obj:`int`):
    ///         The desired length
    ///
    ///     direction: (:obj:`str`, defaults to :obj:`right`):
    ///         The expected padding direction. Can be either :obj:`right` or :obj:`left`
    ///
    ///     pad_id (:obj:`int`, defaults to :obj:`0`):
    ///         The ID corresponding to the padding token
    ///
    ///     pad_type_id (:obj:`int`, defaults to :obj:`0`):
    ///         The type ID corresponding to the padding token
    ///
    ///     pad_token (:obj:`str`, defaults to `[PAD]`):
    ///         The pad token to use
    #[args(kwargs = "**")]
    #[text_signature = "(self, length, direction='right', pad_id=0, pad_type_id=0, pad_token='[PAD]')"]
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
                            .into_pyerr::<exceptions::PyValueError>()),
                        }?;
                    }
                    "pad_id" => pad_id = value.extract()?,
                    "pad_type_id" => pad_type_id = value.extract()?,
                    "pad_token" => pad_token = value.extract()?,
                    _ => println!("Ignored unknown kwarg option {}", key),
                }
            }
        }
        self.encoding
            .pad(length, pad_id, pad_type_id, pad_token, direction);
        Ok(())
    }

    /// Truncate the :class:`~tokenizers.Encoding` at the given length
    ///
    /// If this :class:`~tokenizers.Encoding` represents multiple sequences, when truncating
    /// this information is lost. It will be considered as representing a single sequence.
    ///
    /// Args:
    ///     max_length (:obj:`int`):
    ///         The desired length
    ///
    ///     stride (:obj:`int`, defaults to :obj:`0`):
    ///         The length of previous content to be included in each overflowing piece
    #[args(stride = "0")]
    #[text_signature = "(self, max_length, stride=0)"]
    fn truncate(&mut self, max_length: usize, stride: usize) -> PyResult<()> {
        self.encoding.truncate(max_length, stride);
        Ok(())
    }
}
