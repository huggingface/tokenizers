use tokenizers as tk;

use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;

use super::{
    DestroyPtr, PyNormalizedString, PyNormalizedStringRefMut, RefMutContainer, RefMutGuard,
};
use crate::encoding::PyEncoding;
use crate::error::ToPyResult;
use crate::token::PyToken;
use tk::{OffsetReferential, OffsetType, Offsets, PreTokenizedString, Token};

fn split(pretok: &mut PreTokenizedString, func: &PyAny) -> PyResult<()> {
    if !func.is_callable() {
        Err(exceptions::PyTypeError::new_err(
            "`split` expect a callable with the signature: \
           `fn(index: int, normalized: NormalizedString) -> List[NormalizedString]`",
        ))
    } else {
        ToPyResult(pretok.split(|i, normalized| {
            let output = func.call((i, PyNormalizedString::from(normalized)), None)?;
            Ok(output
                .extract::<Vec<PyNormalizedString>>()?
                .into_iter()
                .map(tk::NormalizedString::from))
        }))
        .into()
    }
}

fn normalize(pretok: &mut PreTokenizedString, func: &PyAny) -> PyResult<()> {
    if !func.is_callable() {
        Err(exceptions::PyTypeError::new_err(
            "`normalize` expect a callable with the signature: \
            `fn(normalized: NormalizedString)`",
        ))
    } else {
        ToPyResult(pretok.normalize(|normalized| {
            let norm = PyNormalizedStringRefMut::new(normalized);
            func.call((norm.get(),), None)?;
            Ok(())
        }))
        .into()
    }
}

fn tokenize(pretok: &mut PreTokenizedString, func: &PyAny) -> PyResult<()> {
    if !func.is_callable() {
        Err(exceptions::PyTypeError::new_err(
            "`tokenize` expect a callable with the signature: \
            `fn(str) -> List[Token]`",
        ))
    } else {
        ToPyResult(pretok.tokenize(|normalized| {
            let output = func.call((normalized.get(),), None)?;
            Ok(output
                .extract::<&PyList>()?
                .into_iter()
                .map(|obj| Ok(Token::from(obj.extract::<PyToken>()?)))
                .collect::<PyResult<Vec<_>>>()?)
        }))
        .into()
    }
}

/// This is an enum
#[derive(Clone)]
pub struct PyOffsetReferential(OffsetReferential);
impl FromPyObject<'_> for PyOffsetReferential {
    fn extract(obj: &PyAny) -> PyResult<Self> {
        let s = obj.extract::<&str>()?;

        Ok(Self(match s {
            "original" => Ok(OffsetReferential::Original),
            "normalized" => Ok(OffsetReferential::Normalized),
            _ => Err(exceptions::PyValueError::new_err(
                "Wrong value for OffsetReferential, expected one of `original, normalized`",
            )),
        }?))
    }
}

#[derive(Clone)]
pub struct PyOffsetType(OffsetType);
impl FromPyObject<'_> for PyOffsetType {
    fn extract(obj: &PyAny) -> PyResult<Self> {
        let s = obj.extract::<&str>()?;

        Ok(Self(match s {
            "byte" => Ok(OffsetType::Byte),
            "char" => Ok(OffsetType::Char),
            _ => Err(exceptions::PyValueError::new_err(
                "Wrong value for OffsetType, expected one of `byte, char`",
            )),
        }?))
    }
}

type PySplit = (String, Offsets, Option<Vec<PyToken>>);
fn get_splits(
    pretok: &PreTokenizedString,
    offset_referential: PyOffsetReferential,
    offset_type: PyOffsetType,
) -> Vec<PySplit> {
    pretok
        .get_splits(offset_referential.0, offset_type.0)
        .into_iter()
        .map(|(s, o, t)| {
            (
                s.to_owned(),
                o,
                t.as_ref()
                    .map(|tokens| tokens.iter().map(|t| t.clone().into()).collect()),
            )
        })
        .collect()
}

fn to_encoding(
    pretok: &PreTokenizedString,
    type_id: u32,
    word_idx: Option<u32>,
) -> PyResult<PyEncoding> {
    Ok(ToPyResult(
        pretok
            .clone()
            .into_encoding(word_idx, type_id, tk::OffsetType::Char),
    )
    .into_py()?
    .into())
}

/// PreTokenizedString
///
/// Wrapper over a string, that provides a way to normalize, pre-tokenize, tokenize the
/// underlying string, while keeping track of the alignment information (offsets).
///
/// The PreTokenizedString manages what we call `splits`. Each split represents a substring
/// which is a subpart of the original string, with the relevant offsets and tokens.
///
/// When calling one of the methods used to modify the PreTokenizedString (namely one of
/// `split`, `normalize` or `tokenize), only the `splits` that don't have any associated
/// tokens will get modified.
///
/// Args:
///     sequence: str:
///         The string sequence used to initialize this PreTokenizedString
#[pyclass(module = "tokenizers", name=PreTokenizedString)]
#[text_signature = "(self, sequence)"]
pub struct PyPreTokenizedString {
    pub(crate) pretok: tk::PreTokenizedString,
}

impl From<PreTokenizedString> for PyPreTokenizedString {
    fn from(pretok: PreTokenizedString) -> Self {
        Self { pretok }
    }
}

impl From<PyPreTokenizedString> for PreTokenizedString {
    fn from(pretok: PyPreTokenizedString) -> Self {
        pretok.pretok
    }
}

#[pymethods]
impl PyPreTokenizedString {
    #[new]
    fn new(s: &str) -> Self {
        PreTokenizedString::from(s).into()
    }

    /// Split the PreTokenizedString using the given `func`
    ///
    /// Args:
    ///     func: Callable[[index, NormalizedString], List[NormalizedString]]:
    ///         The function used to split each underlying split.
    ///         It is expected to return a list of `NormalizedString`, that represent the new
    ///         splits. If the given `NormalizedString` does not need any splitting, we can
    ///         just return it directly.
    ///         In order for the offsets to be tracked accurately, any returned `NormalizedString`
    ///         should come from calling either `.split` or `.slice` on the received one.
    #[text_signature = "(self, func)"]
    fn split(&mut self, func: &PyAny) -> PyResult<()> {
        split(&mut self.pretok, func)
    }

    /// Normalize each split of the `PreTokenizedString` using the given `func`
    ///
    /// Args:
    ///     func: Callable[[NormalizedString], None]:
    ///         The function used to normalize each underlying split. This function
    ///         does not need to return anything, just calling the methods on the provided
    ///         NormalizedString allow its modification.
    #[text_signature = "(self, func)"]
    fn normalize(&mut self, func: &PyAny) -> PyResult<()> {
        normalize(&mut self.pretok, func)
    }

    /// Tokenize each split of the `PreTokenizedString` using the given `func`
    ///
    /// Args:
    ///     func: Callable[[str], List[Token]]:
    ///         The function used to tokenize each underlying split. This function must return
    ///         a list of Token generated from the input str.
    #[text_signature = "(self, func)"]
    fn tokenize(&mut self, func: &PyAny) -> PyResult<()> {
        tokenize(&mut self.pretok, func)
    }

    /// Return an Encoding generated from this PreTokenizedString
    ///
    /// Args:
    ///     type_id: int = 0:
    ///         The type_id to be used on the generated Encoding.
    ///
    ///     word_idx: Optional[int] = None:
    ///         An optional word index to be used for each token of this Encoding. If provided,
    ///         all the word indices in the generated Encoding will use this value, instead
    ///         of the one automatically tracked during pre-tokenization.
    ///
    /// Returns:
    ///     An Encoding
    #[args(type_id = "0", word_idx = "None")]
    #[text_signature = "(self, type_id=0, word_idx=None)"]
    fn to_encoding(&self, type_id: u32, word_idx: Option<u32>) -> PyResult<PyEncoding> {
        to_encoding(&self.pretok, type_id, word_idx)
    }

    /// Get the splits currently managed by the PreTokenizedString
    ///
    /// Args:
    ///     offset_referential: :obj:`str`
    ///         Whether the returned splits should have offsets expressed relative
    ///         to the original string, or the normalized one. choices: "original", "normalized".
    ///
    ///     offset_type: :obj:`str`
    ///         Whether the returned splits should have offsets expressed in bytes or chars.
    ///         When slicing an str, we usually want to use chars, which is the default value.
    ///         Now in some cases it might be interesting to get these offsets expressed in bytes,
    ///         so it is possible to change this here.
    ///         choices: "char", "bytes"
    ///
    /// Returns
    ///     A list of splits
    #[args(
        offset_referential = "PyOffsetReferential(OffsetReferential::Original)",
        offset_type = "PyOffsetType(OffsetType::Char)"
    )]
    #[text_signature = "(self, offset_referential=\"original\", offset_type=\"char\")"]
    fn get_splits(
        &self,
        offset_referential: PyOffsetReferential,
        offset_type: PyOffsetType,
    ) -> Vec<PySplit> {
        get_splits(&self.pretok, offset_referential, offset_type)
    }
}

#[pyclass(module = "tokenizers", name=PreTokenizedString)]
#[derive(Clone)]
pub struct PyPreTokenizedStringRefMut {
    inner: RefMutContainer<PreTokenizedString>,
}

impl DestroyPtr for PyPreTokenizedStringRefMut {
    fn destroy(&mut self) {
        self.inner.destroy();
    }
}

impl PyPreTokenizedStringRefMut {
    pub fn new(pretok: &mut tk::PreTokenizedString) -> RefMutGuard<Self> {
        // SAFETY: This is safe because we return a RefMutGuard here.
        // The compiler will make sure the &mut stays valid as necessary.
        RefMutGuard::new(Self {
            inner: RefMutContainer::new(pretok),
        })
    }

    pub fn destroyed_error() -> PyErr {
        exceptions::PyException::new_err(
            "Cannot use a PreTokenizedStringRefMut outside `pre_tokenize`",
        )
    }
}

#[pymethods]
impl PyPreTokenizedStringRefMut {
    fn split(&mut self, func: &PyAny) -> PyResult<()> {
        self.inner
            .map_mut(|pretok| split(pretok, func))
            .ok_or_else(PyPreTokenizedStringRefMut::destroyed_error)?
    }

    fn normalize(&mut self, func: &PyAny) -> PyResult<()> {
        self.inner
            .map_mut(|pretok| normalize(pretok, func))
            .ok_or_else(PyPreTokenizedStringRefMut::destroyed_error)?
    }

    fn tokenize(&mut self, func: &PyAny) -> PyResult<()> {
        self.inner
            .map_mut(|pretok| tokenize(pretok, func))
            .ok_or_else(PyPreTokenizedStringRefMut::destroyed_error)?
    }

    #[args(type_id = "0", word_idx = "None")]
    fn to_encoding(&self, type_id: u32, word_idx: Option<u32>) -> PyResult<PyEncoding> {
        self.inner
            .map(|pretok| to_encoding(pretok, type_id, word_idx))
            .ok_or_else(PyPreTokenizedStringRefMut::destroyed_error)?
    }

    #[args(
        offset_referential = "PyOffsetReferential(OffsetReferential::Original)",
        offset_type = "PyOffsetType(OffsetType::Char)"
    )]
    fn get_splits(
        &self,
        offset_referential: PyOffsetReferential,
        offset_type: PyOffsetType,
    ) -> PyResult<Vec<PySplit>> {
        self.inner
            .map(|pretok| get_splits(pretok, offset_referential, offset_type))
            .ok_or_else(PyPreTokenizedStringRefMut::destroyed_error)
    }
}
