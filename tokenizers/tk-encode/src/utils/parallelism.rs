//! The worker pool that large inputs and batches encode on, and its size.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::TryLockError;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

/// The value set with [`set_max_threads`]; 0 defaults to the number of available cores.
static MAX_THREADS: AtomicUsize = AtomicUsize::new(0);

/// Bumped when [`set_max_threads`] changes value, so the next [`pool`] call builds a pool of the new size.
static POOL_GEN: AtomicUsize = AtomicUsize::new(0);

struct CachedPool {
    pool: Arc<rayon::ThreadPool>,
    generation: usize,
}

static POOL_CACHE: Mutex<Option<CachedPool>> = Mutex::new(None);

/// The id of the process that owns the pool. Other processes cannot use the pool and encode serially.
///
/// `fork()` copies memory but only the calling thread, so locks held by other threads deadlock in child processes.
/// To avoid complexity handling fork, the first process to ask for a pool owns it, and a
/// child forked after that encodes serially.
/// A child forked before the pool was created, like a pre-fork server's worker, can create and own a pool of its own.
static POOL_OWNER: AtomicU32 = AtomicU32::new(0);

/// Whether this process is the [`POOL_OWNER`], claiming it if no process is yet.
fn owns_pool() -> bool {
    let pid = process_id();
    match POOL_OWNER.compare_exchange(0, pid, Ordering::Relaxed, Ordering::Relaxed) {
        Ok(_) => true,
        Err(owner) => owner == pid,
    }
}

/// Only unix can fork, and [`std::process::id`] panics on targets without processes (wasm).
#[cfg(unix)]
fn process_id() -> u32 {
    std::process::id()
}

#[cfg(not(unix))]
fn process_id() -> u32 {
    1
}

/// How many times [`try_lock_briefly`] retries a held lock before giving up.
const LOCK_ATTEMPTS: u32 = 64;

/// Takes the [`POOL_CACHE`] lock, or gives up after [`LOCK_ATTEMPTS`] tries: the holder may be
/// building a pool, and encoding serially beats waiting for it.
fn try_lock_briefly() -> Option<MutexGuard<'static, Option<CachedPool>>> {
    for attempt in 0..LOCK_ATTEMPTS {
        match POOL_CACHE.try_lock() {
            Ok(g) => return Some(g),
            Err(TryLockError::Poisoned(p)) => return Some(p.into_inner()),
            Err(TryLockError::WouldBlock) if attempt < LOCK_ATTEMPTS / 2 => std::hint::spin_loop(),
            Err(TryLockError::WouldBlock) => std::thread::yield_now(),
        }
    }
    None
}

/// The shared worker pool, or `None` to encode serially: on one core, under contention, or in a
/// forked child.
pub(crate) fn pool() -> Option<Arc<rayon::ThreadPool>> {
    if !owns_pool() {
        return None;
    }

    let generation = POOL_GEN.load(Ordering::Acquire);
    // callers that find the lock held encode serially, so only one of them builds a pool
    let mut cache = try_lock_briefly()?;
    if let Some(cached) = cache.as_ref()
        && cached.generation == generation
    {
        return Some(cached.pool.clone());
    }
    // the calling thread encodes too, so the pool holds one thread fewer
    let workers = max_threads() - 1;
    if workers == 0 {
        *cache = None;
        return None;
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .thread_name(|i| format!("tk-encode-{i}"))
        .build()
        .ok()?;
    let pool = Arc::new(pool);
    *cache = Some(CachedPool {
        pool: pool.clone(),
        generation,
    });
    Some(pool)
}

/// The most threads one encode uses, the calling thread included. `1` means serial.
///
/// The value set with [`set_max_threads`], else every available core. A child forked after this
/// process asked for its pool encodes serially regardless.
pub fn max_threads() -> usize {
    match MAX_THREADS.load(Ordering::Acquire) {
        0 => std::thread::available_parallelism().map_or(1, |n| n.get()),
        n => n,
    }
}

/// Sets [`max_threads`] for the whole process. `0` restores the default.
pub fn set_max_threads(max_threads: usize) {
    // an unchanged value keeps the pool, instead of restarting every thread
    if MAX_THREADS.swap(max_threads, Ordering::AcqRel) != max_threads {
        POOL_GEN.fetch_add(1, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use crate::test_utils::in_forked_child;
    use std::sync::PoisonError;

    /// Tests touching the process-wide pool run one at a time: they hold its lock.
    static POOL_TESTS: Mutex<()> = Mutex::new(());

    /// Contended callers must reuse the cached pool, not build their own.
    #[test]
    fn concurrent_callers_share_one_pool() {
        let _serial = POOL_TESTS.lock().unwrap_or_else(PoisonError::into_inner);
        if max_threads() == 1 {
            return; // no pool on a one-core machine
        }
        // start stale, so the callers race the rebuild too
        POOL_GEN.fetch_add(1, Ordering::Release);
        let callers: Vec<_> = (0..32)
            .map(|_| {
                std::thread::spawn(|| {
                    (0..2_000)
                        .filter_map(|_| pool())
                        .map(|pool| Arc::as_ptr(&pool) as usize)
                        .collect::<std::collections::HashSet<_>>()
                })
            })
            .collect();
        let seen: std::collections::HashSet<_> = callers
            .into_iter()
            .flat_map(|caller| caller.join().unwrap())
            .collect();
        assert_eq!(seen.len(), 1, "contended callers built pools of their own");
    }

    /// The calling thread encodes too, so the pool must hold one thread fewer than the maximum.
    #[test]
    fn max_threads_counts_the_calling_thread() {
        let _serial = POOL_TESTS.lock().unwrap_or_else(PoisonError::into_inner);
        set_max_threads(3);
        let workers = pool().map(|pool| pool.current_num_threads());
        set_max_threads(1);
        let serial = pool().is_none();
        set_max_threads(0);
        assert_eq!(workers, Some(2), "the pool leaves no thread for the caller");
        assert!(serial, "max_threads 1 still built a pool");
    }

    /// Setting the value it already has must keep the pool, not restart every thread.
    #[test]
    fn setting_the_same_max_threads_keeps_the_pool() {
        let _serial = POOL_TESTS.lock().unwrap_or_else(PoisonError::into_inner);
        set_max_threads(3);
        let before = pool().expect("3 threads get a pool");
        set_max_threads(3);
        let after = pool().expect("3 threads get a pool");
        set_max_threads(0);
        assert!(
            Arc::ptr_eq(&before, &after),
            "setting the same value rebuilt the pool"
        );
    }

    /// A thread started in a forked child can wait forever on a lock held by a thread the fork
    /// didn't copy, so the child must encode serially.
    #[cfg(unix)]
    #[test]
    fn forked_child_gets_no_pool() {
        let _serial = POOL_TESTS.lock().unwrap_or_else(PoisonError::into_inner);
        if max_threads() == 1 {
            return; // no pool on a one-core machine
        }
        assert!(pool().is_some(), "the parent gets a pool");
        let child_ok = in_forked_child(|| i32::from(pool().is_some()));
        assert!(child_ok, "the forked child got a pool");
    }

    /// Pre-fork servers (gunicorn) load the tokenizer, then fork their workers. Loading must not
    /// claim the pool, or every worker encodes serially.
    #[cfg(unix)]
    #[test]
    fn worker_forked_after_load_owns_a_pool() {
        use crate::models::bpe::{BPE, BpeConfig};
        use crate::pipeline::{Model, PostProcessor, PreTokenizer, Tokenizer};

        // Runs in a child, where no other test can claim the pool between the reset and the fork.
        let child_ok = in_forked_child(|| {
            POOL_OWNER.store(0, Ordering::Relaxed);
            let bpe = BPE::from_config(BpeConfig {
                vocab: [("a".to_string(), 0u32)].into_iter().collect(),
                ..BpeConfig::default()
            })
            .expect("a one-token vocab is valid");
            let _loaded = Tokenizer::from_parts(
                Default::default(),
                Vec::new(),
                PreTokenizer::None,
                Model::BPE(bpe),
                PostProcessor::default(),
                None,
                Default::default(),
                None,
                None,
            );
            // `owns_pool`, not `pool`: starting threads in a fork of this multi-threaded test
            // binary can hang on a lock a test thread held.
            let worker_ok = in_forked_child(|| i32::from(!owns_pool()));
            i32::from(!worker_ok)
        });
        assert!(child_ok, "a worker forked after load doesn't own a pool");
    }
}
