use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

mod iterators;
mod normalization;
mod pretokenization;
mod regex;

pub use iterators::*;
pub use normalization::*;
pub use pretokenization::*;
pub use regex::*;

// PyChar
// This type is a temporary hack to accept `char` as argument
// To be removed once https://github.com/PyO3/pyo3/pull/1282 has been released
pub struct PyChar(pub char);

impl FromPyObject<'_> for PyChar {
    fn extract(obj: &PyAny) -> PyResult<Self> {
        let s = PyString::try_from(obj)?.to_str()?;
        let mut iter = s.chars();
        if let (Some(ch), None) = (iter.next(), iter.next()) {
            Ok(Self(ch))
        } else {
            Err(exceptions::PyValueError::new_err(
                "expected a string of length 1",
            ))
        }
    }
}

// RefMut utils

pub trait DestroyPtr {
    fn destroy(&mut self);
}

pub struct RefMutGuard<'r, T: DestroyPtr + Clone> {
    content: T,
    r: PhantomData<&'r mut T>,
}
impl<T: DestroyPtr + Clone> RefMutGuard<'_, T> {
    pub fn new(content: T) -> Self {
        Self {
            content,
            r: PhantomData,
        }
    }

    pub fn get(&self) -> T {
        self.content.clone()
    }
}

impl<T: DestroyPtr + Clone> Drop for RefMutGuard<'_, T> {
    fn drop(&mut self) {
        self.content.destroy()
    }
}

#[derive(Clone)]
pub struct RefMutContainer<T> {
    inner: Arc<Mutex<Option<*mut T>>>,
}
impl<T> RefMutContainer<T> {
    pub fn new(content: &mut T) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Some(content))),
        }
    }

    pub fn map<F: FnOnce(&T) -> U, U>(&self, f: F) -> Option<U> {
        let lock = self.inner.lock().unwrap();
        let ptr = lock.as_ref()?;
        Some(f(unsafe { ptr.as_ref().unwrap() }))
    }

    pub fn map_mut<F: FnOnce(&mut T) -> U, U>(&mut self, f: F) -> Option<U> {
        let lock = self.inner.lock().unwrap();
        let ptr = lock.as_ref()?;
        Some(f(unsafe { ptr.as_mut().unwrap() }))
    }
}

impl<T> DestroyPtr for RefMutContainer<T> {
    fn destroy(&mut self) {
        self.inner.lock().unwrap().take();
    }
}

unsafe impl<T: Send> Send for RefMutContainer<T> {}
unsafe impl<T: Sync> Sync for RefMutContainer<T> {}
