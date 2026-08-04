//! Process-global worker pool for the parallel encode path: a lazily-built,
//! library-private `rayon::ThreadPool`.
//!
//! Scheduling lives in `pipeline.rs` as the shared-cursor job model: drainer
//! tasks and the consuming thread claim chunks off one atomic cursor. This pool
//! only provides the worker threads.
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

use crate::utils::parallelism::{
    get_parallelism, mark_parallelism_used, num_threads_override, NUM_THREADS_ENV_VARIABLE,
};

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

/// Abandon the current pool (without touching it) so the next dispatch builds
/// a fresh one — used by `set_num_threads` to apply a new size, and by the
/// fork handler through the same generation mechanism.
pub(crate) fn invalidate() {
    POOL_GEN.fetch_add(1, Ordering::SeqCst);
}

/// Worker count, per the precedence in [`crate::utils::parallelism`]:
/// `set_num_threads` (live) > `TOKENIZERS_NUM_THREADS` (cached at first use) >
/// the machine's available parallelism.
fn num_threads() -> usize {
    if let Some(n) = num_threads_override() {
        return n;
    }
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var(NUM_THREADS_ENV_VARIABLE)
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

/// The pool for one encode dispatch, or `None` when encoding must run inline:
/// parallelism disabled, thread spawn failure, or the cell lock unavailable
/// (held mid-fork by a ghost thread, or poisoned) — `try_lock` is what
/// guarantees this path can never hang. Cheap on the happy path: one atomic
/// load and one uncontended `try_lock`.
pub(crate) fn rayon() -> Option<&'static rayon::ThreadPool> {
    /// The generation-stamped, intentionally-leaked pool.
    static CELL: Mutex<Option<(&'static rayon::ThreadPool, u64)>> = Mutex::new(None);

    if !get_parallelism() {
        return None;
    }
    register_fork_handler();
    let mut guard = CELL.try_lock().ok()?;
    let generation = POOL_GEN.load(Ordering::Acquire);
    if let Some((pool, stamp)) = *guard
        && stamp == generation
    {
        return Some(pool);
    }
    // A stamped-but-stale pool means we forked: abandon the old one without touching it
    // (its threads died with the fork; its locks may be held by ghosts) and build a fresh
    // one for this generation.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads())
        .thread_name(|i| format!("tk-encode-{i}"))
        .build()
        .ok()?;
    let pool: &'static rayon::ThreadPool = Box::leak(Box::new(pool));
    *guard = Some((pool, generation));
    // Feed the Python bindings' fork guard: an *unconfigured* forked child of
    // this process goes serial (v1 semantics, prevents DataLoader thread
    // oversubscription); explicitly configured parallelism survives forks at
    // full speed via the generation rebuild.
    mark_parallelism_used();
    Some(pool)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One sequential test: both scenarios mutate the process-global pool
    /// generation, so running them as separate (concurrent) tests would flake.
    ///
    /// (a) A fork-generation bump abandons the current pool and lazily builds a
    /// fresh one; a stable generation reuses the same pool. Building marks
    /// parallelism as used (fork-guard telemetry).
    /// (b) `set_num_threads` takes priority over env/default and applies
    /// *live*: the existing pool is abandoned and the next dispatch builds one
    /// at the requested size; `0` resets to env-or-default resolution.
    #[test]
    fn pool_generation_and_num_threads_control() {
        use crate::utils::parallelism::set_num_threads;

        // (a) fork generation
        let a = rayon().unwrap() as *const rayon::ThreadPool;
        let a2 = rayon().unwrap() as *const rayon::ThreadPool;
        assert_eq!(a, a2, "stable generation must reuse the pool");
        POOL_GEN.fetch_add(1, Ordering::SeqCst);
        let b = rayon().unwrap() as *const rayon::ThreadPool;
        assert_ne!(a, b, "generation bump must rebuild the pool");
        let b2 = rayon().unwrap() as *const rayon::ThreadPool;
        assert_eq!(b, b2);
        assert!(crate::utils::parallelism::has_parallelism_been_used());

        // (b) live worker-count control
        let default_threads = rayon().unwrap().current_num_threads();
        let target = if default_threads == 3 { 2 } else { 3 };
        set_num_threads(target);
        assert_eq!(
            rayon().unwrap().current_num_threads(),
            target,
            "setter must override env/default and rebuild live"
        );
        set_num_threads(0); // reset to env-or-default
        assert_eq!(rayon().unwrap().current_num_threads(), default_threads);
    }
}
