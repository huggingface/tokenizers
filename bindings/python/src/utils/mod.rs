use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

mod iterators;
mod normalization;
mod pretokenization;
mod regex;
pub mod serde_pyo3;

pub use iterators::*;
pub use normalization::*;
pub use pretokenization::*;
pub use regex::*;

// RefMut utils

pub trait DestroyPtr {
    fn destroy(&mut self);
}

pub struct RefMutGuard<'r, T: DestroyPtr> {
    content: T,
    r: PhantomData<&'r mut T>,
}
impl<T: DestroyPtr> RefMutGuard<'_, T> {
    pub fn new(content: T) -> Self {
        Self {
            content,
            r: PhantomData,
        }
    }

    pub fn get(&self) -> &T {
        &self.content
    }
}

impl<T: DestroyPtr> Drop for RefMutGuard<'_, T> {
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
