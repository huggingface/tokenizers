//! Iterators the trainers run in parallel, or serially when [`max_threads`] is 1.
//!
//! They run on rayon's global pool. A forked child has a copy of that pool without its threads,
//! and work sent to it never runs, so a forked child must train serially.

use rayon::iter::IterBridge;
use rayon::prelude::*;
use rayon_cond::CondIterator;
use tk_encode::parallelism::max_threads;

pub(crate) use rayon::current_num_threads;

fn parallel() -> bool {
    max_threads() > 1
}

/// Allows to convert into an iterator that can be executed either parallelly or serially.
///
/// Parallel unless [`max_threads`] is 1, which it always is in a forked child.
pub(crate) trait MaybeParallelIterator<P, S>
where
    P: ParallelIterator,
    S: Iterator<Item = P::Item>,
{
    /// Convert ourself in a CondIterator, that will be executed either in parallel or serially,
    /// in parallel unless [`max_threads`] is 1
    fn into_maybe_par_iter(self) -> CondIterator<P, S>;
}

impl<P, S, I> MaybeParallelIterator<P, S> for I
where
    I: IntoParallelIterator<Iter = P, Item = P::Item> + IntoIterator<IntoIter = S, Item = S::Item>,
    P: ParallelIterator,
    S: Iterator<Item = P::Item>,
{
    fn into_maybe_par_iter(self) -> CondIterator<P, S> {
        CondIterator::new(self, parallel())
    }
}

/// Shared reference version of MaybeParallelIterator, works the same but returns an iterator
/// over references, does not consume self
pub(crate) trait MaybeParallelRefIterator<'data, P, S>
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
pub(crate) trait MaybeParallelBridge<T, S>
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

        if parallel() {
            CondIterator::from_parallel(iter.into_parallel().right().unwrap())
        } else {
            iter
        }
    }
}

/// Allows to convert into `chunks` that can be executed either parallelly or serially.
pub(crate) trait MaybeParallelSlice<'data, T>
where
    T: Sync,
{
    /// Create a CondIterator, that will be executed either in parallel or serially,
    /// in parallel unless [`max_threads`] is 1
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
        let parallelism = parallel();
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

    /// The parent's global pool has no threads in a forked child: work sent there never runs.
    #[cfg(unix)]
    #[test]
    fn child_forked_after_parallel_training_still_trains() {
        let words: Vec<u32> = (0..10_000).collect();
        assert_eq!(words.maybe_par_iter().sum::<u32>(), 49_995_000);
        // SAFETY: the child only sums, then `_exit`s
        let pid = unsafe { libc::fork() };
        if pid == 0 {
            // SAFETY: plain libc calls in the child; a hung child gets killed after 10 s
            unsafe {
                libc::alarm(10);
                libc::_exit(i32::from(words.maybe_par_iter().sum::<u32>() != 49_995_000));
            }
        }
        let mut status = 0;
        // SAFETY: waits on the child forked above
        assert_eq!(unsafe { libc::waitpid(pid, &mut status, 0) }, pid);
        assert!(
            libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0,
            "the forked child hung on the parent's pool"
        );
    }
}
