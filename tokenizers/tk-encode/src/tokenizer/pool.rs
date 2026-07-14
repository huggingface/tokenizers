//! Process-global worker pool for the parallel encode path: a lazily-built,
//! library-private `rayon::ThreadPool`.
//!
//! Scheduling itself lives in `pipeline.rs` as the shared-cursor job model —
//! drainer tasks and the consuming thread claim chunks off one atomic cursor
//! (caller-assist), which benching showed is where the throughput wins live; a
//! bespoke thread pool added nothing over rayon once the job model carried the
//! assist (see `PARALLEL_RUNTIME_DESIGN.md` §7 for the removal decision and
//! numbers; the handrolled pool lives in git history if ever needed).
//!
//! Fork safety: a `pthread_atfork` child handler bumps [`POOL_GEN`]; the
//! dispatch path compares generations and abandons a stale pool *without
//! touching it* — its threads died with the fork and its locks may be held by
//! ghost threads, so it is intentionally leaked (bounded: one pool per fork) —
//! then lazily builds a fresh one. Callers that cannot get a pool (lock
//! unavailable, spawn failure, parallelism disabled) fall back to the inline
//! path, so a fork can cost throughput but never liveness.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::utils::parallelism::get_parallelism;

/// Fork generation: bumped in the child by the `pthread_atfork` handler (a
/// single atomic store — async-signal-safe). Pools stamped with an older
/// generation are stale and must never be touched again.
static POOL_GEN: AtomicU64 = AtomicU64::new(0);

#[cfg(unix)]
fn register_fork_handler() {
    static REGISTERED: std::sync::Once = std::sync::Once::new();
    REGISTERED.call_once(|| {
        unsafe extern "C" fn child_after_fork() {
            POOL_GEN.fetch_add(1, Ordering::SeqCst);
        }
        // Registration failure means fork detection is off — same behavior as
        // before this module existed; nothing to do about it here.
        unsafe {
            let _ = libc::pthread_atfork(None, None, Some(child_after_fork));
        }
    });
}

#[cfg(not(unix))]
fn register_fork_handler() {}

/// Worker count: `TOKENIZERS_NUM_THREADS` if set, else the machine's available
/// parallelism.
fn num_threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("TOKENIZERS_NUM_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
            })
    })
}

/// The runtime a given encode call should dispatch on.
pub(crate) enum Backend {
    /// No pool available (parallelism disabled, spawn failure, or the pool cell
    /// lock unavailable after a fork): encode on the calling thread.
    Inline,
    Rayon(&'static rayon::ThreadPool),
}

/// Resolve the backend for one encode dispatch. Cheap: one atomic load and one
/// uncontended `try_lock` on the happy path.
pub(crate) fn backend() -> Backend {
    if !get_parallelism() {
        return Backend::Inline;
    }
    register_fork_handler();
    match rayon_pool() {
        Some(p) => Backend::Rayon(p),
        None => Backend::Inline,
    }
}

/// A generation-stamped, intentionally-leaked pool.
struct Stamped<T: 'static> {
    value: T,
    generation: u64,
}

type PoolCell<T> = Mutex<Option<&'static Stamped<T>>>;

/// Get the generation-current value from `cell`, building (and leaking) a fresh
/// one when absent or stale. Returns `None` — the caller falls back inline —
/// when `build` fails or the lock is unavailable (held mid-fork by a ghost
/// thread, or poisoned): `try_lock` is what guarantees this path can never hang.
fn current<T>(cell: &PoolCell<T>, build: impl FnOnce() -> Option<T>) -> Option<&'static T> {
    let mut guard = cell.try_lock().ok()?;
    let generation = POOL_GEN.load(Ordering::Acquire);
    if let Some(stamped) = *guard {
        if stamped.generation == generation {
            return Some(&stamped.value);
        }
    }
    // Absent, or stale after a fork: abandon the old pool without touching it
    // and build a fresh one for this generation.
    let value = build()?;
    let stamped: &'static Stamped<T> = Box::leak(Box::new(Stamped { value, generation }));
    *guard = Some(stamped);
    Some(&stamped.value)
}

fn rayon_pool() -> Option<&'static rayon::ThreadPool> {
    static CELL: PoolCell<rayon::ThreadPool> = Mutex::new(None);
    current(&CELL, || {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads())
            .thread_name(|i| format!("tk-encode-{i}"))
            .build()
            .ok()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fork-generation bump abandons the current pool and lazily builds a
    /// fresh one; a stable generation reuses the same pool.
    #[test]
    fn fork_generation_rebuilds_pool() {
        let a = rayon_pool().unwrap() as *const rayon::ThreadPool;
        let a2 = rayon_pool().unwrap() as *const rayon::ThreadPool;
        assert_eq!(a, a2, "stable generation must reuse the pool");
        POOL_GEN.fetch_add(1, Ordering::SeqCst);
        let b = rayon_pool().unwrap() as *const rayon::ThreadPool;
        assert_ne!(a, b, "generation bump must rebuild the pool");
        let b2 = rayon_pool().unwrap() as *const rayon::ThreadPool;
        assert_eq!(b, b2);
    }
}
