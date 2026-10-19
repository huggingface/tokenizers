//! One pool with reusable ID directories and encoding scratch.
use super::merge::MergeScratch;
use super::storage::{IdAccumulator, IdDirectory, PositionEncodingScratch};
use super::{merge::SelectedRuleIndex, pair_index::ShardRouter};
use std::sync::Mutex;
use tk_encode::Result;

// Enabled only by the isolated thread-policy test. Observe executing tasks so a
// configured pool size alone cannot make the test pass.
#[cfg(test)]
pub(super) static EXPECTED_WORKERS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
pub(super) static OBSERVED_TASKS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// One training call's pool and resources reused by actual executing workers.
/// Logical pair-owner shards can run on any worker. Directory, arena, and codec
/// leases require sequential work without nested pool tasks; joined phases define
/// when the coordinator may release scratch or use the next corpus snapshot.
pub(super) struct Execution {
    pub(super) pool: rayon::ThreadPool,
    encoding: Vec<Mutex<PositionEncodingScratch>>,
    directories: Vec<Mutex<[IdDirectory; 2]>>,
    selected: Mutex<SelectedRuleIndex>,
    router: ShardRouter,
}
impl Execution {
    pub(super) fn new(workers: usize) -> Result<Self> {
        Ok(Self {
            router: ShardRouter::new(workers),
            pool: rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()?,
            encoding: (0..workers)
                .map(|_| Mutex::new(PositionEncodingScratch::default()))
                .collect(),
            directories: (0..workers).map(|_| Mutex::default()).collect(),
            selected: Mutex::default(),
        })
    }
    /// Set a nonbinding capacity hint after zero-merge guards. Allocate lazily
    /// on the first actual accumulator use, not while constructing this pool.
    pub(super) fn expect_id_domain(&self, domain: usize) {
        for directories in &self.directories {
            for directory in directories
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .iter_mut()
            {
                directory.expect_domain(domain);
            }
        }
    }
    pub(super) fn router(&self) -> ShardRouter {
        self.router
    }
    fn directories(&self) -> std::sync::MutexGuard<'_, [IdDirectory; 2]> {
        // The caller must finish sequential work before releasing this lease;
        // nested pool work could re-enter the same worker's directory lock.
        self.directories[self.current_worker()]
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }
    /// Run `work` sequentially and return its result.
    /// After `work` returns, restore lookup directories and drop remaining scratch.
    /// Errors also reset touched IDs. Unwinding drops accumulator values and unlocks
    /// the worker; the next task starts with empty directories. Never nest pool work.
    pub(super) fn with_merge_scratch<T>(
        &self,
        token_id_count: usize,
        work: impl FnOnce(&mut MergeScratch) -> Result<T>,
    ) -> Result<T> {
        let mut directories = self.directories();
        let mut scratch = MergeScratch::new(token_id_count, std::mem::take(&mut *directories));
        let result = work(&mut scratch);
        *directories = scratch.into_directories();
        result
    }
    /// Reuse one directory for owner-local accumulation within a joined phase.
    /// The closure is sequential and must not start nested pool work.
    pub(super) fn with_accumulator<V: Default, T>(
        &self,
        token_id_count: usize,
        directory_index: usize,
        work: impl FnOnce(&mut IdAccumulator<V>) -> Result<T>,
    ) -> Result<T> {
        let mut directories = self.directories();
        let mut values = IdAccumulator::with_directory(
            token_id_count,
            std::mem::take(&mut directories[directory_index]),
        );
        let result = work(&mut values);
        directories[directory_index] = values.into_directory();
        result
    }
    pub(super) fn selected_rules(&self) -> std::sync::MutexGuard<'_, SelectedRuleIndex> {
        self.selected
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }
    pub(super) fn release_scratch(&self) {
        // Every pool phase has joined. Output construction no longer needs
        // directories or codec buffers, so release them before duplicating
        // vocabulary strings for the public model.
        for directory in &self.directories {
            *directory.lock().unwrap_or_else(|error| error.into_inner()) = Default::default();
        }
        for encoding in &self.encoding {
            *encoding.lock().unwrap_or_else(|error| error.into_inner()) = Default::default();
        }
        *self
            .selected
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Default::default();
    }
    pub(super) fn workers(&self) -> usize {
        self.encoding.len()
    }
    /// Return the executing worker in this training pool, not a logical owner ID.
    /// Call only inside this pool's tasks. Worker-local leases must not span nested
    /// pool work, which could re-enter the same resource locks.
    pub(super) fn current_worker(&self) -> usize {
        #[cfg(test)]
        {
            use std::sync::atomic::Ordering;
            let expected = EXPECTED_WORKERS.load(Ordering::Relaxed);
            if expected != 0 {
                assert_eq!(rayon::current_num_threads(), expected);
                OBSERVED_TASKS.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.pool
            .current_thread_index()
            .expect("BPE allocation runs inside its training pool")
    }
    /// Borrow codec scratch for the executing worker returned by `current_worker`.
    /// Keep work sequential until this guard is dropped.
    pub(super) fn encoding(
        &self,
        worker_id: usize,
    ) -> std::sync::MutexGuard<'_, PositionEncodingScratch> {
        self.encoding[worker_id]
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn worker_directories_are_clean_after_errors_and_unwinding() {
        let execution = Execution::new(1).unwrap();
        execution.pool.install(|| {
            let result = execution.with_accumulator::<u64, ()>(4, 0, |values| {
                *values.touch(2) = 99;
                Err("interrupted task".into())
            });
            assert!(result.is_err());
            execution
                .with_accumulator::<u64, _>(4, 0, |values| {
                    assert_eq!(*values.touch(2), 0);
                    Ok(())
                })
                .unwrap();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = execution.with_accumulator::<u64, ()>(4, 0, |values| {
                    *values.touch(2) = 99;
                    panic!("task unwound");
                });
            }));
            assert!(result.is_err());
            execution
                .with_accumulator::<u64, _>(4, 0, |values| {
                    assert_eq!(*values.touch(2), 0);
                    Ok(())
                })
                .unwrap();
        });
    }
}
