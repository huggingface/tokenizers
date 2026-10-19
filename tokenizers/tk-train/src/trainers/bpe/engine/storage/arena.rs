use super::{Result, StorageError};
use bumpalo::Bump;
use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::{Mutex, MutexGuard};

/// Owns small position allocations for one training attempt.
/// Position lists borrow this storage, so the arena must outlive those lists.
/// Dropping an allocation lease releases cursor access and keeps the storage alive.
pub(in super::super) struct AllocationArena {
    workers: Vec<Mutex<Bump>>,
    cutoff: usize,
}
impl AllocationArena {
    pub(in super::super) fn new(workers: usize, physical_items: usize) -> Self {
        Self {
            workers: (0..workers).map(|_| Mutex::new(Bump::new())).collect(),
            // A fixed decision for the scope; large buffers remain individually owned.
            cutoff: ((physical_items as u128 / 256).isqrt() as usize).max(256),
        }
    }
    /// Lock the allocation cursor for executing-worker index `worker`.
    /// Pass a worker index from this training pool, rather than a pair-owner index.
    /// Keep work sequential while the guard is held; nested pool work could try to
    /// lock the same cursor. Dropping the guard keeps published storage alive.
    pub(in super::super) fn lease(&self, worker: usize) -> AllocationLease<'_> {
        AllocationLease {
            arena: self,
            cursor: self.workers[worker]
                .lock()
                .unwrap_or_else(|e| e.into_inner()),
        }
    }
}
/// Temporary exclusive access to one executing worker's allocation cursor.
/// Published position lists borrow the arena for `'arena`.
/// The cursor guard can be dropped while those lists remain in use.
pub(in super::super) struct AllocationLease<'arena> {
    arena: &'arena AllocationArena,
    cursor: MutexGuard<'arena, Bump>,
}
impl AllocationLease<'_> {
    /// Allocate `layout` from the arena and return `Some(pointer)` on success.
    /// If the requested byte size exceeds the arena cutoff, return `None` so the
    /// caller can allocate an owned buffer. A failed small allocation returns an error.
    pub(in super::super) fn allocate(&self, layout: Layout) -> Result<Option<NonNull<u8>>> {
        if layout.size() > self.arena.cutoff {
            return Ok(None);
        }
        self.cursor
            .try_alloc_layout(layout)
            .map(Some)
            .map_err(|_| StorageError("position arena allocation failed"))
    }
}
