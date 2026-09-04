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
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

// Re-export rayon current_num_threads
pub use rayon::current_num_threads;

/// TODO: deprecate
pub const ENV_VARIABLE: &str = "TOKENIZERS_PARALLELISM";

/// TODO: deprecate
static USED_PARALLELISM: AtomicBool = AtomicBool::new(false);

/// TODO: deprecate
static PARALLELISM: AtomicU8 = AtomicU8::new(0);

/// 0 means deafult value
static NUM_THREADS: AtomicUsize = AtomicUsize::new(0);
/// Counter to track the current version of the pool
/// After forking or changing the number of threads, we need to invalidate and recreate a new pool
/// Old pools will be dropped when they go out of scope (arc refcount goes to 0)
static POOL_GEN: AtomicUsize = AtomicUsize::new(0);

/// register an invalidation callback to be called after a fork with pthread_atfork
/// this is required because when forking only the parent thread is copied to the child process so
/// you lose access to the previously built thread pool -> rebuild needed
/// cf the POSIX spec: https://pubs.opengroup.org/onlinepubs/9699919799/functions/fork.html
#[cfg(unix)]
fn register_fork_handler() {
    static REGISTERED: AtomicBool = AtomicBool::new(false);
    if REGISTERED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
    {
        unsafe extern "C" fn child_after_fork() {
            POOL_GEN.fetch_add(1, Ordering::SeqCst);
        }
        unsafe {
            let _ = libc::pthread_atfork(None, None, Some(child_after_fork));
        }
    }
}

#[cfg(not(unix))]
fn register_fork_handler() {}

#[derive(Clone)]
struct Slot {
    pool: Arc<rayon::ThreadPool>,
    version: usize,
    pid: u32,
    idle_workers: Arc<[AtomicBool]>,
}

static CELL: Mutex<Option<Slot>> = Mutex::new(None);

type MaybeLockGuard = Option<MutexGuard<'static, Option<Slot>>>;

fn lock() -> MaybeLockGuard {
    match CELL.try_lock() {
        Ok(g) => Some(g),
        Err(TryLockError::Poisoned(p)) => Some(p.into_inner()),
        Err(TryLockError::WouldBlock) => None,
    }
}

pub(crate) fn pool() -> Option<Arc<rayon::ThreadPool>> {
    register_fork_handler();

    let generation = POOL_GEN.load(Ordering::Acquire);
    if let Some(guard) = lock()
        && let Some(slot) = guard.as_ref()
        && generation == slot.version
    {
        keep_workers_warm(slot);
        return Some(slot.pool.clone());
    }

    let num_threads = num_threads();
    // We don't create a thread pool when thread == 1
    let slot = if num_threads == 1 {
        None
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|i| format!("tk-encode-{i}"))
            .build()
            .ok()?;
        let slot = Slot {
            idle_workers: (0..pool.current_num_threads())
                .map(|_| AtomicBool::new(false))
                .collect(),
            pool: Arc::new(pool),
            version: generation,
            pid: std::process::id(),
        };
        Some(slot)
    };

    let old = lock().and_then(|mut guard| match &slot {
        Some(slot) => guard.replace(slot.clone()),
        None => guard.take(),
    });

    if let Some(old) = old
        && old.pid != std::process::id()
    {
        // mem::forget is here to avoid deadlocking on the pool drop after forking
        std::mem::forget(old.pool);
    }

    slot.map(|slot| {
        keep_workers_warm(&slot);
        slot.pool.clone()
    })
}

pub fn num_threads() -> usize {
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

/// Check if the TOKENIZERS_PARALLELISM env variable has been explicitly set
pub fn is_parallelism_configured() -> bool {
    std::env::var(ENV_VARIABLE).is_ok() || get_override_parallelism().is_some()
}

/// Check if at some point we used a parallel iterator
pub fn has_parallelism_been_used() -> bool {
    USED_PARALLELISM.load(Ordering::SeqCst)
}

/// Get internally set parallelism
fn get_override_parallelism() -> Option<bool> {
    match PARALLELISM.load(Ordering::SeqCst) {
        0 => None,
        1 => Some(false),
        2 => Some(true),
        _ => unreachable!(),
    }
}

/// Get the currently set value for `TOKENIZERS_PARALLELISM` env variable
fn get_env_parallelism() -> bool {
    match std::env::var(ENV_VARIABLE) {
        Ok(mut v) => {
            v.make_ascii_lowercase();
            !matches!(v.as_ref(), "" | "off" | "false" | "f" | "no" | "n" | "0")
        }
        Err(_) => true, // If we couldn't get the variable, we use the default
    }
}

pub fn get_parallelism() -> bool {
    if let Some(parallel) = get_override_parallelism() {
        parallel
    } else {
        get_env_parallelism()
    }
}

/// Set the value for `TOKENIZERS_PARALLELISM` for the current process
pub fn set_parallelism(val: bool) {
    PARALLELISM.store(if val { 2 } else { 1 }, Ordering::SeqCst);
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
    /// Convert ourself in a CondIterator, that will be executed either in parallel or serially,
    /// based on both the `TOKENIZERS_PARALLELISM` environment variable and the provided bool.
    /// Both must be true to run with parallelism activated.
    fn into_maybe_par_iter_cond(self, cond: bool) -> CondIterator<P, S>;
}

impl<P, S, I> MaybeParallelIterator<P, S> for I
where
    I: IntoParallelIterator<Iter = P, Item = P::Item> + IntoIterator<IntoIter = S, Item = S::Item>,
    P: ParallelIterator,
    S: Iterator<Item = P::Item>,
{
    fn into_maybe_par_iter(self) -> CondIterator<P, S> {
        let parallelism = get_parallelism();
        if parallelism {
            USED_PARALLELISM.store(true, Ordering::SeqCst);
        }
        CondIterator::new(self, parallelism)
    }

    fn into_maybe_par_iter_cond(self, cond: bool) -> CondIterator<P, S> {
        if cond {
            self.into_maybe_par_iter()
        } else {
            CondIterator::from_serial(self)
        }
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
    fn maybe_par_iter_cond(&'data self, cond: bool) -> CondIterator<P, S>;
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

    fn maybe_par_iter_cond(&'data self, cond: bool) -> CondIterator<P, S> {
        self.into_maybe_par_iter_cond(cond)
    }
}

/// Exclusive reference version of MaybeParallelIterator, works the same but returns an iterator
/// over mutable references, does not consume self
pub trait MaybeParallelRefMutIterator<'data, P, S>
where
    P: ParallelIterator,
    S: Iterator<Item = P::Item>,
    P::Item: 'data,
{
    fn maybe_par_iter_mut(&'data mut self) -> CondIterator<P, S>;
    fn maybe_par_iter_mut_cond(&'data mut self, cond: bool) -> CondIterator<P, S>;
}

impl<'data, P, S, I: 'data + ?Sized> MaybeParallelRefMutIterator<'data, P, S> for I
where
    &'data mut I: MaybeParallelIterator<P, S>,
    P: ParallelIterator,
    S: Iterator<Item = P::Item>,
    P::Item: 'data,
{
    fn maybe_par_iter_mut(&'data mut self) -> CondIterator<P, S> {
        self.into_maybe_par_iter()
    }

    fn maybe_par_iter_mut_cond(&'data mut self, cond: bool) -> CondIterator<P, S> {
        self.into_maybe_par_iter_cond(cond)
    }
}

/// Converts any serial iterator into a CondIterator, that can either run parallelly or serially.
pub trait MaybeParallelBridge<T, S>
where
    S: Iterator<Item = T> + Send,
    T: Send,
{
    fn maybe_par_bridge(self) -> CondIterator<IterBridge<S>, S>;
    fn maybe_par_bridge_cond(self, cond: bool) -> CondIterator<IterBridge<S>, S>;
}

impl<T, S> MaybeParallelBridge<T, S> for S
where
    S: Iterator<Item = T> + Send,
    T: Send,
{
    fn maybe_par_bridge(self) -> CondIterator<IterBridge<S>, S> {
        let iter = CondIterator::from_serial(self);

        if get_parallelism() {
            USED_PARALLELISM.store(true, Ordering::SeqCst);
            CondIterator::from_parallel(iter.into_parallel().right().unwrap())
        } else {
            iter
        }
    }

    fn maybe_par_bridge_cond(self, cond: bool) -> CondIterator<IterBridge<S>, S> {
        if cond {
            self.maybe_par_bridge()
        } else {
            CondIterator::from_serial(self)
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
    /// Create a CondIterator, that will be executed either in parallel or serially,
    /// based on both the `TOKENIZERS_PARALLELISM` environment variable and the provided bool.
    /// Both must be true to run with parallelism activated.
    fn maybe_par_chunks_cond(
        &'_ self,
        cond: bool,
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
    fn maybe_par_chunks_cond(
        &'_ self,
        cond: bool,
        chunk_size: usize,
    ) -> CondIterator<rayon::slice::Chunks<'_, T>, std::slice::Chunks<'_, T>> {
        if cond {
            self.maybe_par_chunks(chunk_size)
        } else {
            CondIterator::from_serial(self.chunks(chunk_size))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maybe_parallel_iterator() {
        let mut v = vec![1u32, 2, 3, 4, 5, 6];

        assert_eq!(v.maybe_par_iter().sum::<u32>(), 21);
        assert_eq!(
            v.maybe_par_iter_mut()
                .map(|v| {
                    *v *= 2;
                    *v
                })
                .sum::<u32>(),
            42
        );
        assert_eq!(v.maybe_par_iter().sum::<u32>(), 42);
        assert_eq!(v.into_maybe_par_iter().sum::<u32>(), 42);
    }

    #[test]
    fn test_maybe_parallel_slice() {
        let v = [1, 2, 3, 4, 5];

        let chunks: Vec<_> = v.maybe_par_chunks(2).collect();
        assert_eq!(chunks, vec![&[1, 2][..], &[3, 4], &[5]]);
    }
}

static IDLE_SPIN_MICROS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Keep encoding workers polling for work briefly between consecutive batches.
///
/// The default is zero: workers use Rayon's normal idle policy. A nonzero timeout
/// can improve throughput for repeated short batches by avoiding OS yields and
/// wakeups. It consumes CPU while idle for up to this duration after the last job.
/// The setting applies to this crate's pools, and takes effect on subsequent work.
/// Sub-microsecond durations are rounded down. Setting zero disables polling.
pub fn set_idle_spin_timeout(timeout: std::time::Duration) {
    IDLE_SPIN_MICROS.store(
        timeout.as_micros().min(u64::MAX as u128) as u64,
        Ordering::Relaxed,
    );
}

fn keep_workers_warm(slot: &Slot) {
    if IDLE_SPIN_MICROS.load(Ordering::Relaxed) == 0
        || slot
            .idle_workers
            .iter()
            .all(|active| active.load(Ordering::Acquire))
    {
        return;
    }
    let workers = Arc::clone(&slot.idle_workers);
    slot.pool.spawn_broadcast(move |ctx| {
        let active = &workers[ctx.index()];
        if active.swap(true, Ordering::AcqRel) {
            return;
        }
        struct Reset<'a>(&'a AtomicBool);
        impl Drop for Reset<'_> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::Release);
            }
        }
        let _reset = Reset(active);
        let mut idle_since = std::time::Instant::now();
        loop {
            let timeout = IDLE_SPIN_MICROS.load(Ordering::Relaxed);
            if timeout == 0 || idle_since.elapsed().as_micros() >= u128::from(timeout) {
                break;
            }
            // Cooperatively execute queued work without yielding the OS thread.
            // The per-worker flag prevents recursive polling loops when another
            // batch requests warm workers while this task is executing its jobs.
            if matches!(rayon::yield_now(), Some(rayon::Yield::Executed)) {
                idle_since = std::time::Instant::now();
            } else {
                std::hint::spin_loop();
            }
        }
    });
}

#[cfg(test)]
mod idle_tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn cooperative_idle_executes_work_and_expires() {
        let saved = IDLE_SPIN_MICROS.swap(200, Ordering::Relaxed);
        struct Restore(u64);
        impl Drop for Restore {
            fn drop(&mut self) {
                IDLE_SPIN_MICROS.store(self.0, Ordering::Relaxed);
            }
        }
        let _restore = Restore(saved);
        let slot = Slot {
            pool: Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(4)
                    .build()
                    .unwrap(),
            ),
            version: 0,
            pid: std::process::id(),
            idle_workers: (0..4).map(|_| AtomicBool::new(false)).collect(),
        };
        for _ in 0..100 {
            keep_workers_warm(&slot);
            let completed = slot.pool.broadcast(|ctx| ctx.index());
            assert_eq!(completed, vec![0, 1, 2, 3]);
        }
        let workers = Arc::clone(&slot.idle_workers);
        // A final broadcast is behind all posted idle jobs, so every idle job
        // has either entered its loop or returned before shutdown is checked.
        slot.pool.broadcast(|_| {});
        drop(slot);
        let deadline = Instant::now() + Duration::from_secs(2);
        while workers.iter().any(|active| active.load(Ordering::Acquire)) {
            assert!(Instant::now() < deadline, "idle work did not expire");
            std::thread::yield_now();
        }
    }
}
