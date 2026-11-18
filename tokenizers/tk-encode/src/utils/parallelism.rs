//!
//! This module defines helpers to allow optional Rayon usage.
//!

use rayon::iter::IterBridge;
use rayon::prelude::*;
use rayon_cond::CondIterator;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::OnceLock;
use std::sync::TryLockError;
use std::sync::atomic::AtomicPtr;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

// Re-export rayon current_num_threads
pub use rayon::current_num_threads;

/// See [`max_threads`]. 0 means the default.
static MAX_THREADS: AtomicUsize = AtomicUsize::new(0);

/// The cached pool, keyed by the worker count it was built for: a change rebuilds it.
struct CachedPool {
    /// `None` when `max_threads()` is 1 or the build failed: cached too, so a serial process skips
    /// the rebuild path.
    pool: Option<Arc<rayon::ThreadPool>>,
    /// `max_threads() - 1` at build time.
    workers: usize,
}

type PoolCache = Mutex<Option<CachedPool>>;

/// The pool cache until the first `fork()`.
static FIRST_POOL_CACHE: PoolCache = Mutex::new(None);

/// The pool cache [`pool`] reads: [`FIRST_POOL_CACHE`], or a fresh one in a forked child.
///
/// `fork()` copies the whole memory but only the calling thread: the child's copy of the cache may
/// be locked by a thread that doesn't exist there, and its pool has no worker threads (dropping it
/// can deadlock). So the child points this at a fresh cache, and leaks the old one.
static POOL_CACHE: AtomicPtr<PoolCache> = AtomicPtr::new((&raw const FIRST_POOL_CACHE).cast_mut());

fn pool_cache() -> &'static PoolCache {
    // SAFETY: points to `FIRST_POOL_CACHE` or to a leaked `Box`, neither ever freed
    unsafe { &*POOL_CACHE.load(Ordering::Acquire) }
}

/// Registers the fork handler that gives a forked child a fresh pool cache.
///
/// `Once`: no caller proceeds before the handler is registered.
#[cfg(unix)]
fn register_fork_handler() {
    use std::sync::Once;
    static REGISTERED: Once = Once::new();
    // allocating is fine here: glibc and libSystem unlock malloc before running child handlers
    extern "C" fn switch_to_fresh_cache() {
        POOL_CACHE.store(Box::leak(Box::new(Mutex::new(None))), Ordering::Release);
    }
    REGISTERED.call_once(|| {
        // SAFETY: the handler only allocates and stores an atomic
        unsafe {
            let _ = libc::pthread_atfork(None, None, Some(switch_to_fresh_cache));
        }
    });
}

#[cfg(not(unix))]
fn register_fork_handler() {}

/// How many times [`try_lock_briefly`] retries a held lock before giving up.
const LOCK_ATTEMPTS: u32 = 64;

/// Takes the [`pool_cache`] lock, or gives up after [`LOCK_ATTEMPTS`] tries: the holder may be
/// building a pool, and encoding serially beats waiting for it.
fn try_lock_briefly() -> Option<MutexGuard<'static, Option<CachedPool>>> {
    for attempt in 0..LOCK_ATTEMPTS {
        match pool_cache().try_lock() {
            Ok(g) => return Some(g),
            Err(TryLockError::Poisoned(p)) => return Some(p.into_inner()),
            Err(TryLockError::WouldBlock) if attempt < LOCK_ATTEMPTS / 2 => std::hint::spin_loop(),
            Err(TryLockError::WouldBlock) => std::thread::yield_now(),
        }
    }
    None
}

pub(crate) fn pool() -> Option<Arc<rayon::ThreadPool>> {
    register_fork_handler();

    // the caller is one of the `max_threads`. Read once and stored as the key: a racing
    // `set_max_threads` is caught by the next call.
    let workers = max_threads() - 1;
    // callers that find the lock held encode serially, so only one of them builds a pool
    let mut cache = try_lock_briefly()?;
    if let Some(cached) = cache.as_ref()
        && cached.workers == workers
    {
        return cached.pool.clone();
    }
    let pool = if workers == 0 {
        None
    } else {
        // a failed build (thread limit) is cached as serial too, rather than retried every encode
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .thread_name(|i| format!("tk-encode-{i}"))
            .build()
            .ok()
            .map(Arc::new)
    };
    *cache = Some(CachedPool {
        pool: pool.clone(),
        workers,
    });
    pool
}

/// Max threads per encode, caller included: the shared pool holds `max_threads() - 1`, so 1 means
/// serial. Defaults to `available_parallelism()` (affinity mask, cgroup quota).
pub fn max_threads() -> usize {
    match MAX_THREADS.load(Ordering::Acquire) {
        0 => available_parallelism(),
        n => n,
    }
}

/// Cached: on Linux it reads the cgroup quota from files, too slow for every encode.
fn available_parallelism() -> usize {
    static CORES: OnceLock<usize> = OnceLock::new();
    *CORES.get_or_init(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
}

/// Set [`max_threads`] process-wide; `0` restores the default. Takes effect on the next encode.
pub fn set_max_threads(n: usize) {
    MAX_THREADS.store(n, Ordering::Release);
}

/// Whether the `MaybeParallel*` iterators run in parallel: always. They run on rayon's global pool,
/// which `RAYON_NUM_THREADS` sizes.
pub(crate) fn get_parallelism() -> bool {
    true
}

/// Allows to convert into an iterator that can be executed either parallelly or serially,
/// according to [`get_parallelism`].
pub trait MaybeParallelIterator<P, S>
where
    P: ParallelIterator,
    S: Iterator<Item = P::Item>,
{
    /// Convert ourself in a CondIterator, that will be executed either in parallel or serially,
    /// based solely on [`get_parallelism`]
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
    /// based solely on [`get_parallelism`]
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
    use std::sync::PoisonError;

    /// Tests touching the process-wide pool run one at a time: they hold its lock.
    static POOL_TESTS: Mutex<()> = Mutex::new(());

    /// Forks a child that runs `child` and `_exit`s with its code, killed after 10s instead of
    /// hanging; returns whether it exited 0.
    #[cfg(unix)]
    fn in_forked_child(child: impl FnOnce() -> i32) -> bool {
        // SAFETY: the child only runs `child` and `_exit`s
        let pid = unsafe { libc::fork() };
        if pid == 0 {
            unsafe {
                libc::alarm(10);
                libc::_exit(child());
            }
        }
        let mut status = 0;
        assert_eq!(unsafe { libc::waitpid(pid, &mut status, 0) }, pid);
        libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0
    }

    #[test]
    fn test_maybe_parallel_iterator() {
        let v = vec![1u32, 2, 3, 4, 5, 6];

        assert_eq!(v.maybe_par_iter().sum::<u32>(), 21);
        assert_eq!(v.into_maybe_par_iter().sum::<u32>(), 21);
    }

    /// [`pool`], waiting out a build by another test's encode: meanwhile, `pool()` says serial.
    fn built_pool() -> Option<Arc<rayon::ThreadPool>> {
        (0..1_000).find_map(|_| {
            pool().or_else(|| {
                std::thread::sleep(std::time::Duration::from_millis(1));
                None
            })
        })
    }

    /// The caller counts: the pool holds `max - 1` threads, none for a max of 1.
    #[test]
    fn max_threads_counts_the_caller() {
        let _serial = POOL_TESTS.lock().unwrap_or_else(PoisonError::into_inner);
        set_max_threads(3);
        assert_eq!(max_threads(), 3);
        assert_eq!(built_pool().expect("a pool").current_num_threads(), 2);
        set_max_threads(1);
        assert!(pool().is_none(), "one thread is the caller alone");
        set_max_threads(0);
        assert_eq!(
            max_threads(),
            std::thread::available_parallelism().map_or(1, |n| n.get())
        );
    }

    /// Contended callers must reuse the cached pool, not build their own.
    #[test]
    fn concurrent_callers_share_one_pool() {
        let _serial = POOL_TESTS.lock().unwrap_or_else(PoisonError::into_inner);
        if max_threads() == 1 {
            return; // no pool on a one-core machine
        }
        // start stale (cached for another size), so the callers race the rebuild too
        pool();
        set_max_threads(max_threads() + 1);
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
        set_max_threads(0);
        assert_eq!(seen.len(), 1, "contended callers built pools of their own");
    }

    /// A child forked while another thread holds the cache lock (as a build does) still gets a pool.
    #[cfg(unix)]
    #[test]
    fn child_forked_mid_lock_gets_a_pool() {
        let _serial = POOL_TESTS.lock().unwrap_or_else(PoisonError::into_inner);
        if max_threads() == 1 {
            return; // no pool on a one-core machine
        }
        register_fork_handler();
        let (held_tx, held_rx) = std::sync::mpsc::channel();
        let holder = std::thread::spawn(move || {
            let _held = try_lock_briefly();
            held_tx.send(()).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(50));
        });
        held_rx.recv().unwrap();
        let ok = in_forked_child(|| i32::from(pool().is_none()));
        holder.join().unwrap();
        assert!(ok, "the child got no pool");
    }

    /// The parent's pool has no workers in a forked child: the child must build and use its own.
    #[cfg(unix)]
    #[test]
    fn child_runs_on_a_pool_of_its_own() {
        let _serial = POOL_TESTS.lock().unwrap_or_else(PoisonError::into_inner);
        if max_threads() == 1 {
            return; // no pool on a one-core machine
        }
        let parent = pool().expect("a pool");
        let parent = Arc::as_ptr(&parent) as usize;
        let ok = in_forked_child(|| {
            let Some(pool) = pool() else { return 1 };
            if Arc::as_ptr(&pool) as usize == parent {
                return 2;
            }
            // hangs (and the alarm kills the child) if the pool has no workers
            let sum: u32 = pool.install(|| (0..1_000u32).into_par_iter().sum());
            i32::from(sum != 499_500)
        });
        assert!(ok, "the child didn't get a working pool of its own");
    }

    #[test]
    fn test_maybe_parallel_slice() {
        let v = [1, 2, 3, 4, 5];

        let chunks: Vec<_> = v.maybe_par_chunks(2).collect();
        assert_eq!(chunks, vec![&[1, 2][..], &[3, 4], &[5]]);
    }
}
