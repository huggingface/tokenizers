// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lock-free batch work distribution with dynamic window sizing.
//!
//! Replaces rayon's parallel iteration for batch encode with a simpler
//! mechanism: a single atomic counter hands out contiguous windows of
//! item indices to worker threads running on rayon's persistent thread
//! pool.  The only cross-thread synchronization on the hot path is the
//! `AtomicUsize::fetch_add` that claims each window.
//!
//! ## Cache-line / loop-tiling rationale
//!
//! Shared-memory parallel loops are bottlenecked by the cache coherence
//! protocol when two cores alternate writes to the same cache line: the
//! line "ping-pongs" between their private L1d caches, each transfer
//! costing dozens of cycles.  To avoid that, every line should be filled
//! by one producer core, drained (or no longer needed), and only then
//! touched by a different core.  This is the cache-aware equivalent of
//! loop tiling / blocking.
//!
//! The work queue enforces this three ways:
//!
//! 1. The counter itself lives on its own 64-byte cache line
//!    (`#[repr(C, align(64))]` on `AlignedCounter`).  A worker's
//!    `fetch_add` does not evict any neighbouring data, and reads of the
//!    counter do not pull input or result payloads into the core's L1d.
//!
//! 2. Each window is a contiguous run of `window_size` indices, so every
//!    worker owns a run of adjacent slots for the duration of one
//!    window.  With `MAX_WINDOW_SIZE = 8`, a window covers roughly
//!    `8 * sizeof(slot)` bytes -- for `Option<EncodeInput>` (~48 B) that
//!    is ~6 cache lines; for `Option<Result<Encoding>>` (multi-line per
//!    slot) it is even more.  So within one window, a worker writes
//!    several whole cache lines before any other worker comes near them.
//!
//! 3. Each slot has its own `UnsafeCell`
//!    (`Vec<UnsafeCell<Option<T>>>`).  `UnsafeCell<T>` is
//!    `#[repr(transparent)]` so the heap layout is identical to a plain
//!    `Vec<Option<T>>` (no padding, no indirection), but concurrent
//!    accesses to different indices never materialise a shared `&mut`
//!    reference to the enclosing `Vec` (which would be UB, regardless of
//!    which element each access ultimately reached).
//!
//! At window boundaries a single cache line can be shared between two
//! successive windows when the slot size does not divide 64 bytes.  That
//! is a *sequential* handoff (window N finishes writes; window N+1 then
//! reads/writes), not a concurrent ping-pong.
//!
//! ## Window sizing
//!
//! `window_size = ceil(total / (num_threads * WINDOWS_PER_THREAD))`,
//! clamped to `[1, MAX_WINDOW_SIZE]`.
//!
//! - `WINDOWS_PER_THREAD = 4` keeps several windows per thread so a
//!   slow worker on its last item does not stall the whole batch.
//! - `MAX_WINDOW_SIZE = 8` caps per-claim atomic latency and keeps the
//!   per-window memory footprint small enough to fit comfortably in L1d.
//!
//! Example: 100 items / 16 threads yields window_size = 2 (50 windows);
//! 10000 items / 16 threads yields window_size = 8 (1250 windows).

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Minimum number of windows each thread should get for load balancing.
const WINDOWS_PER_THREAD: usize = 4;

/// Maximum window size (items per atomic claim).  Larger values reduce
/// atomic contention but worsen tail-latency from uneven last windows.
const MAX_WINDOW_SIZE: usize = 8;

/// Cache-line-aligned atomic counter.
/// Ensures the counter does not share a cache line with any other data.
#[repr(C, align(64))]
struct AlignedCounter(AtomicUsize);

/// Lock-free work distributor.
///
/// Workers atomically claim non-overlapping windows of item indices.
/// The window size is chosen dynamically based on `total` and
/// `num_threads` so that every thread gets several windows of work.
/// The counter is on its own cache line so claiming work does not
/// contend with result writes.
pub(crate) struct BatchWorkQueue {
    next: AlignedCounter,
    total: usize,
    window_size: usize,
}

impl BatchWorkQueue {
    /// Create a new queue distributing `total` items across `num_threads`.
    ///
    /// The window size is chosen to give each thread at least
    /// `WINDOWS_PER_THREAD` windows, capped at `MAX_WINDOW_SIZE`.
    pub(crate) fn new(total: usize, num_threads: usize) -> Self {
        let target_windows = num_threads.saturating_mul(WINDOWS_PER_THREAD).max(1);
        let window_size = total.div_ceil(target_windows).clamp(1, MAX_WINDOW_SIZE);
        Self {
            next: AlignedCounter(AtomicUsize::new(0)),
            total,
            window_size,
        }
    }

    /// Claim the next window of work items.
    /// Returns `Some((start, end))` half-open range, or `None` when all
    /// items have been claimed.
    pub(crate) fn claim_window(&self) -> Option<(usize, usize)> {
        let start = self.next.0.fetch_add(self.window_size, Ordering::Relaxed);
        if start >= self.total {
            return None;
        }
        Some((start, (start + self.window_size).min(self.total)))
    }
}

/// A `Vec` whose elements can each be *taken* exactly once from any thread.
///
/// The `BatchWorkQueue` guarantees that no two threads access the same
/// index, so no synchronization is needed beyond the queue itself.
///
/// Layout: each slot has its own `UnsafeCell<Option<T>>`.  Because
/// `UnsafeCell<U>` is `#[repr(transparent)]` over `U`, this heap layout
/// is byte-identical to a plain `Vec<Option<T>>`: no added padding,
/// identical slot alignment, identical contiguous packing.  The only
/// difference is that `self.0[i].get()` gives a raw `*mut Option<T>`
/// pointing straight at slot `i`, without ever materialising a
/// `&mut Vec<Option<T>>` (which would alias the enclosing container and
/// be UB when two threads touch any distinct indices concurrently).
pub(crate) struct TakeVec<T>(Vec<UnsafeCell<Option<T>>>);

// SAFETY: callers guarantee each index is accessed by at most one thread;
// `take` produces a raw pointer to a single slot's `UnsafeCell` without
// aliasing the surrounding `Vec`.
unsafe impl<T: Send> Sync for TakeVec<T> {}

impl<T> TakeVec<T> {
    /// Wrap a `Vec<T>` so items can be taken by index.
    pub(crate) fn new(items: Vec<T>) -> Self {
        Self(
            items
                .into_iter()
                .map(|t| UnsafeCell::new(Some(t)))
                .collect(),
        )
    }

    /// Take the item at `index`, leaving `None` in its place.
    /// Panics if the item was already taken.
    pub(crate) fn take(&self, index: usize) -> T {
        // SAFETY: the `BatchWorkQueue` guarantees that each `index` is passed
        // to `take` by at most one thread.  `self.0[index].get()` returns a
        // raw pointer to that slot's `Option<T>`; reborrowing it as `&mut`
        // does not alias any sibling slot's data.
        unsafe {
            (*self.0[index].get())
                .take()
                .expect("batch item already taken")
        }
    }
}

/// A `Vec<Option<T>>` where each slot is written exactly once from any
/// thread.
///
/// The `BatchWorkQueue` guarantees non-overlapping index access.
///
/// Layout: same note as `TakeVec`.  Each slot is a
/// `UnsafeCell<Option<T>>` (`#[repr(transparent)]` over `Option<T>`), so
/// the heap layout is byte-identical to a plain `Vec<Option<T>>`
/// and `self.0[i].get()` yields a raw `*mut Option<T>` to slot `i`
/// without materialising a `&mut Vec<Option<T>>`.
pub(crate) struct ResultVec<T>(Vec<UnsafeCell<Option<T>>>);

// SAFETY: callers guarantee each index is written by at most one thread;
// `set` produces a raw pointer to a single slot's `UnsafeCell` without
// aliasing the surrounding `Vec`.
unsafe impl<T: Send> Sync for ResultVec<T> {}

impl<T> ResultVec<T> {
    /// Allocate `len` empty result slots.
    pub(crate) fn new(len: usize) -> Self {
        Self((0..len).map(|_| UnsafeCell::new(None)).collect())
    }

    /// Write a result to the slot at `index`.
    pub(crate) fn set(&self, index: usize, value: T) {
        // SAFETY: the `BatchWorkQueue` guarantees that each `index` is passed
        // to `set` by at most one thread, so no other reference to this
        // slot's `Option<T>` exists concurrently.
        unsafe {
            *self.0[index].get() = Some(value);
        }
    }

    /// Consume self and return the results in order.
    /// Panics if any slot was not written.
    pub(crate) fn into_vec(self) -> Vec<T> {
        self.0
            .into_iter()
            .map(|cell| cell.into_inner().expect("result slot was never written"))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_work_queue_single_thread() {
        // 20 items, 1 thread => target 4 windows => window_size = 5.
        let queue = BatchWorkQueue::new(20, 1);
        let mut ranges = Vec::new();
        while let Some(range) = queue.claim_window() {
            ranges.push(range);
        }
        assert_eq!(ranges.len(), 4);
        assert_eq!(ranges[0], (0, 5));
        assert_eq!(ranges[1], (5, 10));
        assert_eq!(ranges[2], (10, 15));
        assert_eq!(ranges[3], (15, 20));
    }

    #[test]
    fn test_batch_work_queue_many_threads() {
        // 100 items, 16 threads => target 64 windows => window_size = 2.
        let queue = BatchWorkQueue::new(100, 16);
        let mut ranges = Vec::new();
        while let Some(range) = queue.claim_window() {
            ranges.push(range);
        }
        assert_eq!(ranges.len(), 50);
        assert_eq!(ranges[0], (0, 2));
        assert_eq!(ranges[49], (98, 100));
    }

    #[test]
    fn test_batch_work_queue_window_capped() {
        // 10000 items, 4 threads => target 16 windows => window_size = 625,
        // but capped at MAX_WINDOW_SIZE (8).
        let queue = BatchWorkQueue::new(10000, 4);
        let mut count = 0;
        while queue.claim_window().is_some() {
            count += 1;
        }
        // 10000 / 8 = 1250 windows.
        assert_eq!(count, 1250);
    }

    #[test]
    fn test_batch_work_queue_empty() {
        let queue = BatchWorkQueue::new(0, 4);
        assert!(queue.claim_window().is_none());
    }

    #[test]
    fn test_take_vec() {
        let tv = TakeVec::new(vec![10, 20, 30]);
        assert_eq!(tv.take(1), 20);
        assert_eq!(tv.take(0), 10);
        assert_eq!(tv.take(2), 30);
    }

    #[test]
    fn test_result_vec() {
        let rv = ResultVec::<i32>::new(3);
        rv.set(2, 30);
        rv.set(0, 10);
        rv.set(1, 20);
        assert_eq!(rv.into_vec(), vec![10, 20, 30]);
    }

    #[test]
    fn test_parallel_distribution() {
        let n = 100;
        let num_threads = 4;
        let queue = BatchWorkQueue::new(n, num_threads);
        let results = ResultVec::new(n);

        std::thread::scope(|s| {
            for _ in 0..num_threads {
                s.spawn(|| {
                    while let Some((start, end)) = queue.claim_window() {
                        for i in start..end {
                            results.set(i, i * 2);
                        }
                    }
                });
            }
        });

        let v = results.into_vec();
        for (i, &item) in v.iter().enumerate() {
            assert_eq!(item, i * 2);
        }
    }
}
