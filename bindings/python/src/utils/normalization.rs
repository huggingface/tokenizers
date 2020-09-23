use super::regex::PyRegex;
use super::{DestroyPtr, RefMutContainer, RefMutGuard};
use crate::error::ToPyResult;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;
use pyo3::{PyMappingProtocol, PyObjectProtocol};
use tk::normalizer::{char_to_bytes, NormalizedString, Range, SplitDelimiterBehavior};
use tk::pattern::Pattern;

#[derive(Clone, FromPyObject)]
pub enum PyPattern<'p> {
    #[pyo3(annotation = "str")]
    Str(&'p str),
    #[pyo3(annotation = "tokenizers.Regex")]
    Regex(Py<PyRegex>),
    // TODO: Add the compatibility for Fn(char) -> bool
}

impl Pattern for PyPattern<'_> {
    fn find_matches(&self, inside: &str) -> tk::Result<Vec<(tk::Offsets, bool)>> {
        match self {
            PyPattern::Str(s) => {
                let mut chars = s.chars();
                if let (Some(c), None) = (chars.next(), chars.next()) {
                    c.find_matches(inside)
                } else {
                    s.find_matches(inside)
                }
            }
            PyPattern::Regex(r) => {
                Python::with_gil(|py| (&r.borrow(py).inner).find_matches(inside))
            }
        }
    }
}

impl From<PyPattern<'_>> for tk::normalizers::replace::ReplacePattern {
    fn from(pattern: PyPattern<'_>) -> Self {
        match pattern {
            PyPattern::Str(s) => Self::String(s.to_owned()),
            PyPattern::Regex(r) => Python::with_gil(|py| Self::Regex(r.borrow(py).pattern.clone())),
        }
    }
}

#[derive(Debug, Clone, FromPyObject)]
pub enum PyRange<'s> {
    #[pyo3(annotation = "int")]
    Single(isize),
    #[pyo3(annotation = "Tuple[uint, uint]")]
    Range(usize, usize),
    #[pyo3(annotation = "slice")]
    Slice(&'s PySlice),
}
impl PyRange<'_> {
    pub fn to_range(&self, max_len: usize) -> PyResult<std::ops::Range<usize>> {
        match self {
            PyRange::Single(i) => {
                if i.is_negative() {
                    let i = -i as usize;
                    if i > max_len {
                        Err(exceptions::PyValueError::new_err(format!(
                            "{} is bigger than max len",
                            i
                        )))
                    } else {
                        Ok(max_len - i..max_len - i + 1)
                    }
                } else {
                    let i = *i as usize;
                    Ok(i..i + 1)
                }
            }
            PyRange::Range(s, e) => Ok(*s..*e),
            PyRange::Slice(s) => {
                let r = s.indices(max_len as std::os::raw::c_long)?;
                Ok(r.start as usize..r.stop as usize)
            }
        }
    }
}

#[derive(Clone)]
pub struct PySplitDelimiterBehavior(SplitDelimiterBehavior);

impl FromPyObject<'_> for PySplitDelimiterBehavior {
    fn extract(obj: &PyAny) -> PyResult<Self> {
        let s = obj.extract::<&str>()?;

        Ok(Self(match s {
            "removed" => Ok(SplitDelimiterBehavior::Removed),
            "isolated" => Ok(SplitDelimiterBehavior::Isolated),
            "merged_with_previous" => Ok(SplitDelimiterBehavior::MergedWithPrevious),
            "merged_with_next" => Ok(SplitDelimiterBehavior::MergedWithNext),
            "contiguous" => Ok(SplitDelimiterBehavior::Contiguous),
            _ => Err(exceptions::PyValueError::new_err(
                "Wrong value for SplitDelimiterBehavior, expected one of: \
                `removed, isolated, merged_with_previous, merged_with_next, contiguous`",
            )),
        }?))
    }
}

impl From<PySplitDelimiterBehavior> for SplitDelimiterBehavior {
    fn from(v: PySplitDelimiterBehavior) -> Self {
        v.0
    }
}

fn filter(normalized: &mut NormalizedString, func: &PyAny) -> PyResult<()> {
    let err = "`filter` expect a callable with the signature: `fn(char) -> bool`";

    if !func.is_callable() {
        Err(exceptions::PyTypeError::new_err(err))
    } else {
        normalized.filter(|c| {
            func.call1((c.to_string(),))
                .expect(err)
                .extract()
                .expect(err)
        });

        Ok(())
    }
}

fn for_each(normalized: &NormalizedString, func: &PyAny) -> PyResult<()> {
    let err = "`for_each` expect a callable with the signature: `fn(char)`";

    if !func.is_callable() {
        Err(exceptions::PyTypeError::new_err(err))
    } else {
        normalized.for_each(|c| {
            func.call1((c.to_string(),)).expect(err);
        });

        Ok(())
    }
}

fn map(normalized: &mut NormalizedString, func: &PyAny) -> PyResult<()> {
    let err = "`map` expect a callable with the signature: `fn(char) -> char`";

    if !func.is_callable() {
        Err(exceptions::PyTypeError::new_err(err))
    } else {
        normalized.map(|c| {
            let c: &str = func
                .call1((c.to_string(),))
                .expect(err)
                .extract()
                .expect(err);
            c.chars().next().expect(err)
        });

        Ok(())
    }
}

fn slice(
    normalized: &NormalizedString,
    range: &PyRange<'_>,
) -> PyResult<Option<PyNormalizedString>> {
    let n_char = normalized.len();
    let char_range = range.to_range(n_char)?;
    Ok(char_to_bytes(normalized.get(), char_range)
        .map(|bytes_range| {
            normalized
                .slice(Range::Normalized(bytes_range))
                .map(|n| n.into())
        })
        .flatten())
}

#[pyclass(module = "tokenizers", name=NormalizedString)]
#[derive(Clone)]
pub struct PyNormalizedString {
    pub(crate) normalized: NormalizedString,
}

#[pymethods]
impl PyNormalizedString {
    #[new]
    fn new(s: &str) -> Self {
        NormalizedString::from(s).into()
    }

    #[getter]
    fn get_normalized(&self) -> &str {
        self.normalized.get()
    }

    #[getter]
    fn get_original(&self) -> &str {
        self.normalized.get_original()
    }

    fn nfd(&mut self) {
        self.normalized.nfd();
    }

    fn nfkd(&mut self) {
        self.normalized.nfkd();
    }

    fn nfc(&mut self) {
        self.normalized.nfc();
    }

    fn nfkc(&mut self) {
        self.normalized.nfkc();
    }

    fn lowercase(&mut self) {
        self.normalized.lowercase();
    }

    fn uppercase(&mut self) {
        self.normalized.uppercase();
    }

    fn prepend(&mut self, s: &str) {
        self.normalized.prepend(s);
    }

    fn append(&mut self, s: &str) {
        self.normalized.append(s);
    }

    fn lstrip(&mut self) {
        self.normalized.lstrip();
    }

    fn rstrip(&mut self) {
        self.normalized.rstrip();
    }

    fn strip(&mut self) {
        self.normalized.strip();
    }

    fn clear(&mut self) {
        self.normalized.clear();
    }

    fn slice(&self, range: PyRange) -> PyResult<Option<PyNormalizedString>> {
        slice(&self.normalized, &range)
    }

    fn filter(&mut self, func: &PyAny) -> PyResult<()> {
        filter(&mut self.normalized, func)
    }

    fn for_each(&self, func: &PyAny) -> PyResult<()> {
        for_each(&self.normalized, func)
    }

    fn map(&mut self, func: &PyAny) -> PyResult<()> {
        map(&mut self.normalized, func)
    }

    fn split(
        &mut self,
        pattern: PyPattern,
        behavior: PySplitDelimiterBehavior,
    ) -> PyResult<Vec<PyNormalizedString>> {
        Ok(ToPyResult(self.normalized.split(pattern, behavior.into()))
            .into_py()?
            .into_iter()
            .map(|n| n.into())
            .collect())
    }

    fn replace(&mut self, pattern: PyPattern, content: &str) -> PyResult<()> {
        ToPyResult(self.normalized.replace(pattern, content)).into()
    }
}

#[pyproto]
impl PyObjectProtocol<'p> for PyNormalizedString {
    fn __repr__(&self) -> String {
        format!(
            r#"NormalizedString(original="{}", normalized="{}")"#,
            self.normalized.get_original(),
            self.normalized.get()
        )
    }

    fn __str__(&'p self) -> &'p str {
        self.normalized.get()
    }
}

#[pyproto]
impl PyMappingProtocol<'p> for PyNormalizedString {
    fn __getitem__(&self, range: PyRange<'p>) -> PyResult<Option<PyNormalizedString>> {
        slice(&self.normalized, &range)
    }
}

impl From<NormalizedString> for PyNormalizedString {
    fn from(normalized: NormalizedString) -> Self {
        Self { normalized }
    }
}

impl From<PyNormalizedString> for NormalizedString {
    fn from(normalized: PyNormalizedString) -> Self {
        normalized.normalized
    }
}

#[pyclass(module = "tokenizers", name=NormalizedStringRefMut)]
#[derive(Clone)]
pub struct PyNormalizedStringRefMut {
    inner: RefMutContainer<NormalizedString>,
}

impl DestroyPtr for PyNormalizedStringRefMut {
    fn destroy(&mut self) {
        self.inner.destroy();
    }
}

impl PyNormalizedStringRefMut {
    pub fn new(normalized: &mut NormalizedString) -> RefMutGuard<Self> {
        RefMutGuard::new(Self {
            inner: RefMutContainer::new(normalized),
        })
    }

    pub fn destroyed_error() -> PyErr {
        exceptions::PyException::new_err("Cannot use a NormalizedStringRefMut outside `normalize`")
    }
}

#[pymethods]
impl PyNormalizedStringRefMut {
    #[getter]
    fn get_normalized(&self) -> PyResult<String> {
        self.inner
            .map(|n| n.get().to_owned())
            .ok_or_else(PyNormalizedStringRefMut::destroyed_error)
    }

    #[getter]
    fn get_original(&self) -> PyResult<String> {
        self.inner
            .map(|n| n.get_original().to_owned())
            .ok_or_else(PyNormalizedStringRefMut::destroyed_error)
    }

    fn nfd(&mut self) -> PyResult<()> {
        self.inner
            .map_mut(|n| {
                n.nfd();
            })
            .ok_or_else(PyNormalizedStringRefMut::destroyed_error)?;
        Ok(())
    }

    fn nfkd(&mut self) -> PyResult<()> {
        self.inner
            .map_mut(|n| {
                n.nfkd();
            })
            .ok_or_else(PyNormalizedStringRefMut::destroyed_error)?;
        Ok(())
    }

    fn nfc(&mut self) -> PyResult<()> {
        self.inner
            .map_mut(|n| {
                n.nfc();
            })
            .ok_or_else(PyNormalizedStringRefMut::destroyed_error)?;
        Ok(())
    }

    fn nfkc(&mut self) -> PyResult<()> {
        self.inner
            .map_mut(|n| {
                n.nfkc();
            })
            .ok_or_else(PyNormalizedStringRefMut::destroyed_error)?;
        Ok(())
    }

    fn lowercase(&mut self) -> PyResult<()> {
        self.inner
            .map_mut(|n| {
                n.lowercase();
            })
            .ok_or_else(PyNormalizedStringRefMut::destroyed_error)?;
        Ok(())
    }

    fn uppercase(&mut self) -> PyResult<()> {
        self.inner
            .map_mut(|n| {
                n.uppercase();
            })
            .ok_or_else(PyNormalizedStringRefMut::destroyed_error)?;
        Ok(())
    }

    fn prepend(&mut self, s: &str) -> PyResult<()> {
        self.inner
            .map_mut(|n| {
                n.prepend(s);
            })
            .ok_or_else(PyNormalizedStringRefMut::destroyed_error)?;
        Ok(())
    }

    fn append(&mut self, s: &str) -> PyResult<()> {
        self.inner
            .map_mut(|n| {
                n.append(s);
            })
            .ok_or_else(PyNormalizedStringRefMut::destroyed_error)?;
        Ok(())
    }

    fn lstrip(&mut self) -> PyResult<()> {
        self.inner
            .map_mut(|n| {
                n.lstrip();
            })
            .ok_or_else(PyNormalizedStringRefMut::destroyed_error)?;
        Ok(())
    }

    fn rstrip(&mut self) -> PyResult<()> {
        self.inner
            .map_mut(|n| {
                n.rstrip();
            })
            .ok_or_else(PyNormalizedStringRefMut::destroyed_error)?;
        Ok(())
    }

    fn strip(&mut self) -> PyResult<()> {
        self.inner
            .map_mut(|n| {
                n.strip();
            })
            .ok_or_else(PyNormalizedStringRefMut::destroyed_error)?;
        Ok(())
    }

    fn clear(&mut self) -> PyResult<()> {
        self.inner
            .map_mut(|n| {
                n.clear();
            })
            .ok_or_else(PyNormalizedStringRefMut::destroyed_error)?;
        Ok(())
    }

    fn slice(&self, range: PyRange) -> PyResult<Option<PyNormalizedString>> {
        self.inner
            .map(|n| slice(&n, &range))
            .ok_or_else(PyNormalizedStringRefMut::destroyed_error)?
    }

    fn filter(&mut self, func: &PyAny) -> PyResult<()> {
        self.inner
            .map_mut(|mut n| filter(&mut n, func))
            .ok_or_else(PyNormalizedStringRefMut::destroyed_error)??;
        Ok(())
    }

    fn for_each(&self, func: &PyAny) -> PyResult<()> {
        self.inner
            .map(|n| for_each(&n, func))
            .ok_or_else(PyNormalizedStringRefMut::destroyed_error)??;
        Ok(())
    }

    fn map(&mut self, func: &PyAny) -> PyResult<()> {
        self.inner
            .map_mut(|mut n| map(&mut n, func))
            .ok_or_else(PyNormalizedStringRefMut::destroyed_error)??;
        Ok(())
    }

    fn split(
        &mut self,
        pattern: PyPattern,
        behavior: PySplitDelimiterBehavior,
    ) -> PyResult<Vec<PyNormalizedString>> {
        Ok(ToPyResult(
            self.inner
                .map_mut(|n| n.split(pattern, behavior.into()))
                .ok_or_else(PyNormalizedStringRefMut::destroyed_error)?,
        )
        .into_py()?
        .into_iter()
        .map(|n| n.into())
        .collect())
    }

    fn replace(&mut self, pattern: PyPattern, content: &str) -> PyResult<()> {
        ToPyResult(
            self.inner
                .map_mut(|n| n.replace(pattern, content))
                .ok_or_else(PyNormalizedStringRefMut::destroyed_error)?,
        )
        .into()
    }
}
