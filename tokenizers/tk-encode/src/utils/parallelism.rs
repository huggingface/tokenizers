//! Process-wide parallelism controls.
//!
//! The **programmatic setters are the primary interface**; the environment
//! variables are the backwards-compatibility channel. Precedence, highest
//! first:
//!
//! 1. [`set_parallelism`] / [`set_num_threads`] — always live; a threads
//!    change rebuilds the encode pool on its next use.
//! 2. `TOKENIZERS_PARALLELISM` / `TOKENIZERS_NUM_THREADS` — read **once, at
//!    first use** (set them before the first encode; to change behavior after
//!    that, use the setters).
//! 3. Sound defaults: parallelism on, one worker per available core
//!    (cgroup-aware via `std::thread::available_parallelism`).
//!
//! This module also defines the `Maybe*` helpers the legacy paths use for
//! optional Rayon usage.

use rayon::iter::IterBridge;
use rayon::prelude::*;
use rayon_cond::CondIterator;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::OnceLock;

// Re-export rayon current_num_threads
pub use rayon::current_num_threads;

pub const ENV_VARIABLE: &str = "TOKENIZERS_PARALLELISM";
pub const NUM_THREADS_ENV_VARIABLE: &str = "TOKENIZERS_NUM_THREADS";

static USED_PARALLELISM: AtomicBool = AtomicBool::new(false);
static PARALLELISM: AtomicU8 = AtomicU8::new(0);
/// Programmatic worker-count override; `0` means "not set".
static NUM_THREADS: AtomicUsize = AtomicUsize::new(0);

/// Check if parallelism has been explicitly configured, via either setter or
/// env var. (The Python bindings' fork guard leaves configured processes
/// alone and only forces *unconfigured* forked children serial.)
pub fn is_parallelism_configured() -> bool {
    std::env::var(ENV_VARIABLE).is_ok() || get_override_parallelism().is_some()
}

/// Check if parallelism has been used at some point (a parallel iterator, or
/// the encode worker pool).
pub fn has_parallelism_been_used() -> bool {
    USED_PARALLELISM.load(Ordering::SeqCst)
}

/// Record that parallelism was exercised (feeds the fork guard above).
pub(crate) fn mark_parallelism_used() {
    USED_PARALLELISM.store(true, Ordering::SeqCst);
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

/// Whether encoding may parallelize: [`set_parallelism`] wins, then the
/// `TOKENIZERS_PARALLELISM` env var (cached at first use), then on.
pub fn get_parallelism() -> bool {
    if let Some(parallel) = get_override_parallelism() {
        parallel
    } else {
        // The env var is the backwards-compat channel, frozen at first use;
        // runtime changes go through `set_parallelism`.
        static ENV: OnceLock<bool> = OnceLock::new();
        *ENV.get_or_init(get_env_parallelism)
    }
}

/// Enable or disable parallelism for the current process. Always live; takes
/// priority over the `TOKENIZERS_PARALLELISM` env var.
pub fn set_parallelism(val: bool) {
    PARALLELISM.store(if val { 2 } else { 1 }, Ordering::SeqCst);
}

/// Set the number of worker threads for the parallel encode pool. Takes
/// priority over the `TOKENIZERS_NUM_THREADS` env var; `0` resets to the
/// env-var-or-default resolution. Always live: an existing pool is abandoned
/// and rebuilt at the new size on the next encode (the abandoned pool is
/// leaked — bounded by the number of calls, so don't call this in a loop).
pub fn set_num_threads(num_threads: usize) {
    NUM_THREADS.store(num_threads, Ordering::SeqCst);
    crate::tokenizer::pool::invalidate();
}

/// The programmatic worker-count override, if one was set.
pub(crate) fn num_threads_override() -> Option<usize> {
    match NUM_THREADS.load(Ordering::SeqCst) {
        0 => None,
        n => Some(n),
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
