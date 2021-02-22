use pyo3::prelude::*;
use pyo3::{AsPyPointer, PyNativeType};
use std::collections::VecDeque;

/// An simple iterator that can be instantiated with a specified length.
/// We use this with iterators that don't have a size_hint but we might
/// know its size. This is useful with progress bars for example.
pub struct MaybeSizedIterator<I> {
    length: Option<usize>,
    iter: I,
}

impl<I> MaybeSizedIterator<I>
where
    I: Iterator,
{
    pub fn new(iter: I, length: Option<usize>) -> Self {
        Self { iter, length }
    }
}

impl<I> Iterator for MaybeSizedIterator<I>
where
    I: Iterator,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.length.unwrap_or(0), None)
    }
}

/// A buffered iterator that takes care of locking the GIL only when needed.
/// The `PyIterator` provided by PyO3 keeps a Python GIL token all along
/// and thus doesn't allow us to release the GIL to allow having other threads.
///
/// This iterator serves two purposes:
///   - First, as opposed to the `pyo3::PyIterator`, it is Send and can easily be parallelized
///   - Second, this let us release the GIL between two refills of the buffer, allowing other
///     Python threads to work
pub struct PyBufferedIterator<T, F> {
    iter: Option<Py<PyAny>>,
    converter: F,
    buffer: VecDeque<PyResult<T>>,
    size: usize,
}

impl<T, F, I> PyBufferedIterator<T, F>
where
    F: Fn(&PyAny) -> I,
    I: IntoIterator<Item = PyResult<T>>,
{
    /// Create a new PyBufferedIterator using the provided Python object.
    /// This object must implement the Python Iterator Protocol, and an error will
    /// be return if the contract is not respected.
    ///
    /// The `converter` provides a way to convert each item in the iterator into
    /// something that doesn't embed a 'py token and thus allows the GIL to be released
    ///
    /// The `buffer_size` represents the number of items that we buffer before we
    /// need to acquire the GIL again.
    pub fn new(iter: &PyAny, converter: F, buffer_size: usize) -> PyResult<Self> {
        let py = iter.py();
        let iter: Py<PyAny> = unsafe {
            py.from_borrowed_ptr_or_err::<PyAny>(pyo3::ffi::PyObject_GetIter(iter.as_ptr()))?
                .to_object(py)
        };

        Ok(Self {
            iter: Some(iter),
            converter,
            buffer: VecDeque::with_capacity(buffer_size),
            size: buffer_size,
        })
    }

    /// Refill the buffer, and set `self.iter` as `None` if nothing more to get
    fn refill(&mut self) -> PyResult<()> {
        if self.iter.is_none() {
            return Ok(());
        }

        Python::with_gil(|py| loop {
            if self.buffer.len() >= self.size {
                return Ok(());
            }

            match unsafe {
                py.from_owned_ptr_or_opt::<PyAny>(pyo3::ffi::PyIter_Next(
                    self.iter.as_ref().unwrap().as_ref(py).as_ptr(),
                ))
            } {
                Some(obj) => self.buffer.extend((self.converter)(obj)),
                None => {
                    if PyErr::occurred(py) {
                        return Err(PyErr::fetch(py));
                    } else {
                        self.iter = None;
                    }
                }
            };

            if self.iter.is_none() {
                return Ok(());
            }
        })
    }
}

impl<T, F, I> Iterator for PyBufferedIterator<T, F>
where
    F: Fn(&PyAny) -> I,
    I: IntoIterator<Item = PyResult<T>>,
{
    type Item = PyResult<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.buffer.is_empty() {
            self.buffer.pop_front()
        } else if self.iter.is_some() {
            if let Err(e) = self.refill() {
                return Some(Err(e));
            }
            self.next()
        } else {
            None
        }
    }
}
