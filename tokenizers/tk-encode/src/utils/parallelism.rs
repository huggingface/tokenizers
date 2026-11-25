//!
//! This module defines helpers to allow optional Rayon usage.
//!

use rayon::iter::IterBridge;
use rayon::prelude::*;
use rayon_cond::CondIterator;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::TryLockError;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

// Re-export rayon current_num_threads
pub use rayon::current_num_threads;

/// TODO: deprecate
pub(crate) const ENV_VARIABLE: &str = "TOKENIZERS_PARALLELISM";

/// 0 means deafult value
static NUM_THREADS: AtomicUsize = AtomicUsize::new(0);
/// Bumped by [`set_num_threads`], so the next [`pool`] call builds a pool with the new count.
static POOL_GEN: AtomicUsize = AtomicUsize::new(0);

struct CachedPool {
    pool: Arc<rayon::ThreadPool>,
    generation: usize,
}

static POOL_CACHE: Mutex<Option<CachedPool>> = Mutex::new(None);

/// The id of the process that owns the pool. Any other process is a forked child.
///
/// `fork()` copies memory but only the calling thread, so locks held by other threads stay locked
/// in the child. A thread the child starts can wait on one of them forever, so children run
/// serially.
static POOL_OWNER: AtomicU32 = AtomicU32::new(0);

/// Makes this process the [`POOL_OWNER`], unless a process already is.
///
/// Called when a tokenizer is built, not only in [`pool`]. Workers are often forked before the
/// first input long enough to ask for a pool.
pub(crate) fn claim_pool() {
    let _ = POOL_OWNER.compare_exchange(0, process_id(), Ordering::Relaxed, Ordering::Relaxed);
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
    claim_pool();
    if POOL_OWNER.load(Ordering::Relaxed) != process_id() {
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
    let num_threads = num_threads();
    // no pool for a single thread
    if num_threads == 1 {
        *cache = None;
        return None;
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
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

pub(crate) fn num_threads() -> usize {
    match NUM_THREADS.load(Ordering::Acquire) {
        // 0 == default value
        0 => std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        n => n,
    }
}

fn invalidate() {
    POOL_GEN.fetch_add(1, Ordering::SeqCst);
}

/// Passing in 0 will reset to the default value
pub fn set_num_threads(n: usize) {
    NUM_THREADS.store(n, Ordering::Release);
    invalidate();
}

/// Get the currently set value for `TOKENIZERS_PARALLELISM` env variable
pub(crate) fn get_parallelism() -> bool {
    match std::env::var(ENV_VARIABLE) {
        Ok(mut v) => {
            v.make_ascii_lowercase();
            !matches!(v.as_ref(), "" | "off" | "false" | "f" | "no" | "n" | "0")
        }
        Err(_) => true, // If we couldn't get the variable, we use the default
    }
}

/// Allows to convert into an iterator that can be executed either parallelly or serially.
///
/// The choice is made according to the currently set `TOKENIZERS_PARALLELISM` environment variable.
/// This variable can have one of the following values
///   - False => "" (empty value), "false", "f", "off", "no", "n", "0"
///   - True => Any other value
///
pub trait MaybeParallelIterator<P, S>
where
    P: ParallelIterator,
    S: Iterator<Item = P::Item>,
{
    /// Convert ourself in a CondIterator, that will be executed either in parallel or serially,
    /// based solely on the `TOKENIZERS_PARALLELISM` environment variable
    fn into_maybe_par_iter(self) -> CondIterator<P, S>;
}

impl<P, S, I> MaybeParallelIterator<P, S> for I
where
    I: IntoParallelIterator<Iter = P, Item = P::Item> + IntoIterator<IntoIter = S, Item = S::Item>,
    P: ParallelIterator,
    S: Iterator<Item = P::Item>,
{
    fn into_maybe_par_iter(self) -> CondIterator<P, S> {
        CondIterator::new(self, get_parallelism())
    }
}

/// Shared reference version of MaybeParallelIterator, works the same but returns an iterator
/// over references, does not consume self
pub trait MaybeParallelRefIterator<'data, P, S>
where
    P: ParallelIterator,
    S: Iterator<Item = P::Item>,
    P::Item: 'data,
{
    fn maybe_par_iter(&'data self) -> CondIterator<P, S>;
}

impl<'data, P, S, I: 'data + ?Sized> MaybeParallelRefIterator<'data, P, S> for I
where
    &'data I: MaybeParallelIterator<P, S>,
    P: ParallelIterator,
    S: Iterator<Item = P::Item>,
    P::Item: 'data,
{
    fn maybe_par_iter(&'data self) -> CondIterator<P, S> {
        self.into_maybe_par_iter()
    }
}

/// Converts any serial iterator into a CondIterator, that can either run parallelly or serially.
pub trait MaybeParallelBridge<T, S>
where
    S: Iterator<Item = T> + Send,
    T: Send,
{
    fn maybe_par_bridge(self) -> CondIterator<IterBridge<S>, S>;
}

impl<T, S> MaybeParallelBridge<T, S> for S
where
    S: Iterator<Item = T> + Send,
    T: Send,
{
    fn maybe_par_bridge(self) -> CondIterator<IterBridge<S>, S> {
        let iter = CondIterator::from_serial(self);

        if get_parallelism() {
            CondIterator::from_parallel(iter.into_parallel().right().unwrap())
        } else {
            iter
        }
    }
}

/// Allows to convert into `chunks` that can be executed either parallelly or serially.
pub trait MaybeParallelSlice<'data, T>
where
    T: Sync,
{
    /// Create a CondIterator, that will be executed either in parallel or serially,
    /// based solely on the `TOKENIZERS_PARALLELISM` environment variable
    fn maybe_par_chunks(
        &'_ self,
        chunk_size: usize,
    ) -> CondIterator<rayon::slice::Chunks<'_, T>, std::slice::Chunks<'_, T>>;
}

impl<T> MaybeParallelSlice<'_, T> for [T]
where
    T: Sync,
{
    fn maybe_par_chunks(
        &'_ self,
        chunk_size: usize,
    ) -> CondIterator<rayon::slice::Chunks<'_, T>, std::slice::Chunks<'_, T>> {
        let parallelism = get_parallelism();
        if parallelism {
            CondIterator::from_parallel(self.par_chunks(chunk_size))
        } else {
            CondIterator::from_serial(self.chunks(chunk_size))
        }
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

    #[test]
    fn test_maybe_parallel_iterator() {
        let v = vec![1u32, 2, 3, 4, 5, 6];

        assert_eq!(v.maybe_par_iter().sum::<u32>(), 21);
        assert_eq!(v.into_maybe_par_iter().sum::<u32>(), 21);
    }

    /// Contended callers must reuse the cached pool, not build their own.
    #[test]
    fn concurrent_callers_share_one_pool() {
        let _serial = POOL_TESTS.lock().unwrap_or_else(PoisonError::into_inner);
        if num_threads() == 1 {
            return; // no pool on a one-core machine
        }
        // start stale, so the callers race the rebuild too
        invalidate();
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

    /// A thread started in a forked child can wait forever on a lock held by a thread the fork
    /// didn't copy, so the child must encode serially.
    #[cfg(unix)]
    #[test]
    fn forked_child_gets_no_pool() {
        let _serial = POOL_TESTS.lock().unwrap_or_else(PoisonError::into_inner);
        if num_threads() == 1 {
            return; // no pool on a one-core machine
        }
        assert!(pool().is_some(), "the parent gets a pool");
        let child_ok = in_forked_child(|| i32::from(pool().is_some()));
        assert!(child_ok, "the forked child got a pool");
    }

    /// Workers are usually forked after loading, before any input long enough to ask for a pool.
    #[cfg(unix)]
    #[test]
    fn child_forked_after_load_gets_no_pool() {
        use crate::models::bpe::{BPE, BpeConfig};
        use crate::pipeline::{Model, PostProcessor, PreTokenizer, Tokenizer};

        let _serial = POOL_TESTS.lock().unwrap_or_else(PoisonError::into_inner);
        if num_threads() == 1 {
            return; // no pool on a one-core machine
        }
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
        let child_ok = in_forked_child(|| i32::from(pool().is_some()));
        assert!(child_ok, "a child forked after load got a pool");
    }

    #[test]
    fn test_maybe_parallel_slice() {
        let v = [1, 2, 3, 4, 5];

        let chunks: Vec<_> = v.maybe_par_chunks(2).collect();
        assert_eq!(chunks, vec![&[1, 2][..], &[3, 4], &[5]]);
    }
}
