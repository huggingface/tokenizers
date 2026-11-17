//! Counts `fork()`s, so state a child inherits mid-use can be thrown away instead of trusted.
//!
//! Only the forking thread survives a `fork()`: a lock any other thread held at that moment
//! stays locked in the child forever. Keep a [`generation`] next to such state and replace the
//! state when it no longer matches.

use std::sync::atomic::{AtomicUsize, Ordering};

static FORKS: AtomicUsize = AtomicUsize::new(0);

/// How many `fork()`s led to this process since [`watch`] was first called.
pub(crate) fn generation() -> usize {
    // Relaxed: the handler runs in the child before any other thread exists, and spawning one
    // synchronizes with it
    FORKS.load(Ordering::Relaxed)
}

/// Start counting forks. Call it before reading [`generation`] for state worth protecting.
#[cfg(unix)]
pub(crate) fn watch() {
    use std::sync::Once;
    static REGISTERED: Once = Once::new();
    extern "C" fn count_fork() {
        FORKS.fetch_add(1, Ordering::Relaxed);
    }
    REGISTERED.call_once(|| {
        // SAFETY: the handler only touches an atomic
        unsafe {
            let _ = libc::pthread_atfork(None, None, Some(count_fork));
        }
    });
}

#[cfg(not(unix))]
pub(crate) fn watch() {}
