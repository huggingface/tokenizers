//! Compile-time enforcement of the crate's lock/GIL ordering.
//!
//! The rule: **never block on the tokenizer lock while attached to the
//! interpreter**. Training holds the write lock for its whole run and
//! re-attaches to refill from the Python iterator (lock → GIL); an attached
//! thread blocking on the lock would take GIL → lock, closing a deadlock
//! cycle.
//!
//! [`DetachedRwLock`] turns that rule from a review checklist into a type
//! property: the raw `RwLock` is private to this module, and the only way to
//! reach a lock guard is [`DetachedRwLock::with`], which detaches first. Code
//! that forgets to detach has no method to call.
//!
//! Residual hole (not expressible on stable Rust): re-attaching *inside*
//! `with` and locking from there. `clippy.toml` bans `Python::attach`
//! crate-wide as a backstop; the single vetted use is the training iterator
//! refill, which holds the lock and *then* attaches — the sanctioned
//! direction.

#![allow(
    clippy::disallowed_types,
    reason = "the one raw RwLock the wrapper encapsulates"
)]

use std::sync::{PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

use pyo3::Python;
use pyo3::marker::Ungil;

pub struct DetachedRwLock<T> {
    inner: RwLock<T>,
}

/// Proof of detachment: only constructed by [`DetachedRwLock::with`], never
/// clonable or storable beyond the closure (its lifetime is bound to the
/// borrow of the lock inside `with`).
pub struct Detached<'a, T> {
    lock: &'a RwLock<T>,
}

impl<T: Send + Sync> DetachedRwLock<T> {
    pub fn new(value: T) -> Self {
        Self {
            inner: RwLock::new(value),
        }
    }

    /// Detach from the interpreter, then run `f` with lock access.
    pub fn with<R, F>(&self, py: Python<'_>, f: F) -> R
    where
        F: for<'a> FnOnce(Detached<'a, T>) -> R + Ungil + Send,
        R: Ungil + Send,
    {
        py.detach(|| f(Detached { lock: &self.inner }))
    }
}

impl<'a, T> Detached<'a, T> {
    pub fn read(&self) -> Result<RwLockReadGuard<'a, T>, PoisonError<RwLockReadGuard<'a, T>>> {
        self.lock.read()
    }

    pub fn write(&self) -> Result<RwLockWriteGuard<'a, T>, PoisonError<RwLockWriteGuard<'a, T>>> {
        self.lock.write()
    }
}
