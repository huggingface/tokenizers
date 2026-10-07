const NO_ENTRY: u32 = u32::MAX;
/// Reusable ID-to-entry storage. Only touched IDs own accumulated values.
#[derive(Default)]
pub(in super::super) struct IdDirectory {
    indices: Vec<u32>,
    expected_domain: usize,
    drain_pending: bool,
}
impl IdDirectory {
    pub(in super::super) fn expect_domain(&mut self, domain: usize) {
        self.expected_domain = domain.min(u32::MAX as usize);
    }
    fn ensure_domain(&mut self, domain: usize) {
        if domain > self.indices.capacity() {
            // This is a capacity hint, never an ID limit. Reserve the exact
            // expected bound once, rather than round 100k/200k up to powers of
            // two. Only the actual domain is initialized; later jobs append
            // its new tail and preserve previously initialized indices.
            self.indices
                .reserve_exact(self.expected_domain.max(domain) - self.indices.len());
        }
        if domain > self.indices.len() {
            self.indices.resize(domain, NO_ENTRY);
        }
    }
}
/// Accumulate by ID in a dense u32 directory; values occupy touched storage only.
/// A D-entry reservation requests 4*D bytes: two directions on eight workers
/// request about 6.10 MiB at D=100k or 12.21 MiB at D=200k. An allocator may
/// grant more capacity and has additional overhead; storage is reused per job.
/// Vocabulary IDs are contiguous and exclude u32::MAX (the word separator).
pub(in super::super) struct IdAccumulator<T> {
    directory: IdDirectory,
    entries: Vec<(u32, T)>,
}
impl<T: Default> IdAccumulator<T> {
    /// Reuse a returned directory. IDs must belong to the supplied actual domain;
    /// an expected capacity hint never prevents a larger actual domain. Shrinking
    /// a later domain retains initialized storage, so invalid IDs beyond that
    /// domain are not guaranteed to panic. Engine callers always pass valid IDs.
    pub(in super::super) fn with_directory(domain: usize, mut directory: IdDirectory) -> Self {
        // Unique touched IDs are fewer than or equal to this contiguous domain.
        // Domain <= u32::MAX therefore bounds entry indices by u32::MAX-1,
        // leaving NO_ENTRY unavailable as a real entry index. Retained lengths
        // obey the same bound because every earlier constructor checked it.
        assert!(
            domain <= u32::MAX as usize,
            "BPE ID domain exceeds the reserved separator"
        );
        directory.ensure_domain(domain);
        Self {
            directory,
            entries: Vec::new(),
        }
    }
    /// Return a touched value, creating its default value on the first visit.
    #[inline(always)]
    pub(in super::super) fn touch(&mut self, id: u32) -> &mut T {
        let index = &mut self.directory.indices[id as usize];
        if *index == NO_ENTRY {
            let next = self.entries.len() as u32;
            debug_assert!(next < NO_ENTRY);
            let value = T::default();
            self.entries.push((id, value));
            // Real IDs exclude the separator, so at most u32::MAX distinct
            // entries exist and their last index is at most u32::MAX-1.
            // Publish after construction: a caught Default panic leaves no
            // stale index in a reusable directory.
            *index = next;
        }
        &mut self.entries[*index as usize].1
    }
    #[cfg(test)]
    fn get(&self, id: u32) -> Option<&T> {
        self.directory
            .indices
            .get(id as usize)
            .filter(|&&index| index != NO_ENTRY)
            .map(|&index| &self.entries[index as usize].1)
    }
    /// Move out touched values, retaining allocations. Dropping the iterator
    /// also removes unconsumed values and clears their directory entries.
    pub(in super::super) fn drain(&mut self) -> impl Iterator<Item = (u32, T)> + '_ {
        let prior_pending = self.directory.drain_pending;
        self.directory.drain_pending = true;
        IdDrain {
            entries: self.entries.drain(..),
            indices: &mut self.directory.indices,
            pending: &mut self.directory.drain_pending,
            prior_pending,
        }
    }
}
impl<T> IdAccumulator<T> {
    /// Number of touched IDs, including empty birth chains.
    pub(in super::super) fn touched_len(&self) -> usize {
        self.entries.len()
    }
    /// Release values and return only reusable ID lookup storage.
    pub(in super::super) fn into_directory(self) -> IdDirectory {
        let Self {
            mut directory,
            entries,
        } = self;
        if directory.drain_pending {
            // A forgotten Vec::Drain leaked its values and bypassed touched-ID
            // cleanup. Only this exceptional transfer needs a complete reset.
            directory.indices.fill(NO_ENTRY);
            directory.drain_pending = false;
        } else {
            for (id, _) in entries {
                directory.indices[id as usize] = NO_ENTRY;
            }
        }
        directory
    }
}
struct IdDrain<'a, T> {
    entries: std::vec::Drain<'a, (u32, T)>,
    indices: &'a mut [u32],
    pending: &'a mut bool,
    prior_pending: bool,
}
impl<T> Iterator for IdDrain<'_, T> {
    type Item = (u32, T);
    fn next(&mut self) -> Option<Self::Item> {
        self.entries.next().map(|(id, value)| {
            self.indices[id as usize] = NO_ENTRY;
            (id, value)
        })
    }
}
impl<T> Drop for IdDrain<'_, T> {
    fn drop(&mut self) {
        for (id, _) in &mut self.entries {
            self.indices[id as usize] = NO_ENTRY;
        }
        // A later drain cannot clean IDs leaked by an earlier forgotten drain.
        *self.pending = self.prior_pending;
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn touched_values_do_not_survive_drain_across_large_domains() {
        let mut directory = IdDirectory::default();
        directory.expect_domain(usize::MAX);
        assert_eq!(directory.expected_domain, u32::MAX as usize);
        assert_eq!(super::super::super::expected_id_domain(usize::MAX, 3, 2), 5);
        for domain in [4, 100_000, 200_000] {
            let mut directory = IdDirectory::default();
            directory.expect_domain(domain);
            assert_eq!(directory.indices.capacity(), 0);
            let mut counts = IdAccumulator::<usize>::with_directory(4, directory);
            let allocation = (
                counts.directory.indices.as_ptr(),
                counts.directory.indices.capacity(),
            );
            assert!(allocation.1 >= domain);
            *counts.touch(1) += 3;
            *counts.touch(1) += 7;
            *counts.touch(3) += 11;
            assert_eq!(counts.touched_len(), 2);
            let mut result: Vec<_> = counts.drain().collect();
            assert_eq!(counts.touched_len(), 0);
            result.sort_unstable();
            assert_eq!(result, [(1, 10), (3, 11)]);
            assert_eq!(counts.get(1), None);
            assert_eq!(*counts.touch(1), 0);
            *counts.touch(3) = 99;
            drop(counts.drain().take(1));
            assert_eq!(counts.get(1), None);
            assert_eq!(counts.get(3), None);
            let mut counts = IdAccumulator::<u64>::with_directory(domain, counts.into_directory());
            assert_eq!(
                (
                    counts.directory.indices.as_ptr(),
                    counts.directory.indices.capacity()
                ),
                allocation
            );
            assert_eq!(*counts.touch((domain - 1) as u32), 0);
            let mut counts =
                IdAccumulator::<u64>::with_directory(domain + 1, counts.into_directory());
            assert_eq!(*counts.touch(domain as u32), 0);
        }
    }
    #[test]
    fn recycled_directory_drops_values_and_resets_unconsumed_ids() {
        use std::{cell::Cell, rc::Rc};
        let drops = Rc::new(Cell::new(0));
        #[derive(Default)]
        struct Value(Option<Rc<Cell<usize>>>);
        impl Drop for Value {
            fn drop(&mut self) {
                if let Some(drops) = &self.0 {
                    drops.set(drops.get() + 1);
                }
            }
        }
        let mut values = IdAccumulator::<Value>::with_directory(4, IdDirectory::default());
        values.touch(1).0 = Some(drops.clone());
        values.touch(3).0 = Some(drops.clone());
        let mut drain = values.drain();
        drop(drain.next());
        drop(drain);
        values.touch(2).0 = Some(drops.clone());
        let directory = values.into_directory();
        assert_eq!(drops.get(), 3);
        let mut counts = IdAccumulator::<usize>::with_directory(8, directory);
        for id in [1, 2, 3, 7] {
            assert_eq!(*counts.touch(id), 0);
        }
        *counts.touch(7) = 19;
        let mut small = IdAccumulator::<usize>::with_directory(2, counts.into_directory());
        assert_eq!(*small.touch(1), 0);
    }

    #[test]
    fn caught_value_construction_panic_leaves_a_reusable_directory() {
        struct Panics;
        impl Default for Panics {
            fn default() -> Self {
                panic!("value construction failed")
            }
        }
        let mut values = IdAccumulator::<Panics>::with_directory(4, IdDirectory::default());
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = values.touch(2);
            }))
            .is_err()
        );
        let mut counts = IdAccumulator::<u64>::with_directory(4, values.into_directory());
        assert_eq!(counts.get(2), None);
        assert_eq!(*counts.touch(2), 0);
    }
    #[test]
    fn forgotten_drain_does_not_transfer_dirty_ids_to_another_accumulator() {
        for (domain, later_drains) in [(4, 0), (100_000, 1), (200_000, 3)] {
            let id = (domain - 1) as u32;
            let mut counts = IdAccumulator::<u64>::with_directory(domain, IdDirectory::default());
            *counts.touch(id) = 7;
            std::mem::forget(counts.drain());
            for _ in 0..later_drains {
                drop(counts.drain());
            }
            let mut counts =
                IdAccumulator::<usize>::with_directory(domain, counts.into_directory());
            *counts.touch(0) = 99;
            assert_eq!(counts.get(id), None);
            assert_eq!(*counts.touch(id), 0);
        }
    }
}
