//! Nondecreasing u64 positions stored as absolute seeds and unsigned LEB128 gaps.
//! Each group contains at most 128 positions. A restart-offset directory locates
//! its eight-byte seed; later values store differences from their predecessors.
//! Seeking decodes at most 127 gaps. Small lists have inline representations.
//! This is a private in-memory representation, not a serialized file format.
//! Allocation tags distinguish arena storage from independently owned buffers;
//! cursors never share state. For the integer encoding alone, see
//! <https://protobuf.dev/programming-guides/encoding/#base-128-varints>.
use super::{AllocationLease, Result, StorageError};
use std::alloc::{Layout, alloc, dealloc};
use std::collections::BinaryHeap;
use std::marker::PhantomData;
use std::ops::Range;

// Positions per restart group; independent of the integer codec's base 128.
const RESTART_INTERVAL: usize = 128;
const INLINE: usize = 1 << (usize::BITS - 1);
const PAIR: usize = 1 << (usize::BITS - 2);
const DELTA_MASK: usize = PAIR - 1;
const ARENA: usize = 1;
const MULTI: usize = 2;
const RESERVED: usize = 4;

/// Reusable encoding space. Only its written suffix is initialized.
#[derive(Default)]
pub(in super::super) struct PositionEncodingScratch {
    stream: Vec<u8>,
    restarts: Vec<usize>,
}

/// Owns a nondecreasing position list. Equal values remain distinct elements.
///
/// Small lists are inline; larger lists own a heap buffer or borrow arena storage.
/// The arena must outlive lists that borrow its storage. Lists can move between
/// workers. Mutation requires exclusive access. The storage format is private.
#[derive(Default)]
pub(in super::super) struct SortedPositions<'arena> {
    count_and_flags: usize,
    payload: *mut u8,
    arena_lifetime: PhantomData<&'arena super::AllocationArena>,
}
// SAFETY: each list exclusively owns its heap buffer or borrows stable arena
// storage. Moving it does not move that storage; the arena is Send + Sync.
unsafe impl Send for SortedPositions<'_> {}
// SAFETY: all writes require &mut self. Shared methods only read initialized
// bytes, and each cursor owns its decoder state. The arena outlives the list.
unsafe impl Sync for SortedPositions<'_> {}

fn prefix_bytes(groups: usize, reserved: bool) -> Option<usize> {
    if groups > 1 {
        groups.checked_mul(8)?.checked_add(24)
    } else {
        Some(if reserved { 16 } else { 8 })
    }
}
fn layout(groups: usize, capacity: usize, reserved: bool) -> Result<Layout> {
    let bytes = prefix_bytes(groups, reserved)
        .and_then(|head| head.checked_add(capacity))
        .ok_or(StorageError("position allocation size overflow"))?;
    Layout::from_size_align(bytes, 8)
        .map_err(|_| StorageError("position allocation exceeds resident bounds"))
}
fn varint_bytes(value: u64) -> usize {
    let width = 64 - (value | 1).leading_zeros();
    ((width * 9 + 64) >> 6) as usize
}
/// # Safety
/// `target` has space for `varint_bytes(value)` writable bytes.
unsafe fn write_varint(mut target: *mut u8, mut value: u64) {
    // SAFETY: the caller reserves the full encoded width, at most ten bytes.
    unsafe {
        while value >= 128 {
            target.write((value as u8 & 127) | 128);
            target = target.add(1);
            value >>= 7;
        }
        target.write(value as u8);
    }
}
/// # Safety
/// `source` points to a complete varint emitted by this module.
unsafe fn read_varint(source: &mut *const u8) -> u64 {
    let mut value = 0;
    let mut shift = 0;
    // SAFETY: the encoder emits at most ten bytes and terminates the last byte.
    unsafe {
        loop {
            let byte = (*source).read();
            *source = (*source).add(1);
            value |= u64::from(byte & 127) << shift;
            if byte < 128 {
                return value;
            }
            shift += 7;
        }
    }
}

// Visit exactly one reverse run, checking every gap, including seed boundaries.
// For compressed lists, scratch-backed construction encodes once into scratch.
// Direct construction and appends measure first, then replay into final storage.
// Appends write beyond the published prefix.
fn reverse_codes(
    start: usize,
    end: usize,
    previous: u64,
    mut input: impl Iterator<Item = u64>,
    mut visit: impl FnMut(usize, u64) -> Result<()>,
) -> Result<()> {
    let mut value = input
        .next()
        .ok_or(StorageError("position run ended early"))?;
    for index in (start..end).rev() {
        let lower = if index == start {
            previous
        } else {
            input
                .next()
                .ok_or(StorageError("position run ended early"))?
        };
        let gap = value
            .checked_sub(lower)
            .ok_or(StorageError("positions are not sorted"))?;
        visit(
            index,
            if index.is_multiple_of(RESTART_INTERVAL) {
                value
            } else {
                gap
            },
        )?;
        value = lower;
    }
    if input.next().is_some() {
        return Err(StorageError("position run exceeds its declared length"));
    }
    Ok(())
}

fn encoded_size(
    start: usize,
    end: usize,
    previous: u64,
    input: impl Iterator<Item = u64>,
) -> Result<usize> {
    let mut used = 0_usize;
    reverse_codes(start, end, previous, input, |index, code| {
        let bytes = if index.is_multiple_of(RESTART_INTERVAL) {
            8
        } else {
            varint_bytes(code)
        };
        used = used
            .checked_add(bytes)
            .ok_or(StorageError("position stream size overflow"))?;
        Ok(())
    })?;

    Ok(used)
}

impl PositionEncodingScratch {
    fn encode_reverse(
        &mut self,
        start: usize,
        end: usize,
        previous: u64,
        input: impl Iterator<Item = u64>,
    ) -> Result<usize> {
        self.stream.clear();
        self.restarts.clear();
        let seeds = end.div_ceil(RESTART_INTERVAL) - start.div_ceil(RESTART_INTERVAL);
        let bound = (end - start)
            .checked_mul(10)
            .map(|bytes| bytes - 2 * seeds)
            .ok_or(StorageError(
                "position encoding scratch exceeds resident bounds",
            ))?;
        self.restarts
            .try_reserve_exact(seeds)
            .map_err(|_| StorageError("position restart scratch allocation failed"))?;
        if self.stream.capacity() < bound {
            // A fresh allocation avoids copying an old, mostly unwritten buffer.
            let mut stream = Vec::new();
            stream
                .try_reserve_exact(bound)
                .map_err(|_| StorageError("position encoding scratch allocation failed"))?;
            self.stream = stream;
        }
        let data = self.stream.as_mut_ptr();
        let mut cursor = self.stream.capacity();
        reverse_codes(start, end, previous, input, |index, code| {
            let seed = index % RESTART_INTERVAL == 0;
            let bytes = if seed { 8 } else { varint_bytes(code) };
            cursor = cursor
                .checked_sub(bytes)
                .ok_or(StorageError("position run exceeds its declared length"))?;
            // SAFETY: the checked cursor and the width bound keep every write
            // inside scratch capacity. Vec length stays zero until reuse/drop.
            unsafe {
                let target = data.add(cursor);
                if seed {
                    target.cast::<u64>().write_unaligned(code);
                    self.restarts.push(cursor);
                } else {
                    write_varint(target, code);
                }
            }
            Ok(())
        })?;
        Ok(cursor)
    }
}

impl<'arena> SortedPositions<'arena> {
    /// Construct an empty list without an allocation.
    pub(in super::super) fn new() -> Self {
        Self::default()
    }
    /// Return the number of positions, including duplicates.
    #[inline]
    pub(in super::super) fn len(&self) -> usize {
        if self.count_and_flags & INLINE == 0 {
            self.count_and_flags
        } else if self.count_and_flags & PAIR == 0 {
            1
        } else {
            2
        }
    }
    /// Return whether there are no positions.
    #[inline]
    pub(in super::super) fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn is_inline(&self) -> bool {
        self.is_empty() || self.count_and_flags & INLINE != 0
    }
    fn allocation_ptr(&self) -> *mut u8 {
        self.payload.map_addr(|address| address & !7)
    }
    fn multi(&self) -> bool {
        self.payload.addr() & MULTI != 0
    }
    fn reserved(&self) -> bool {
        self.payload.addr() & RESERVED != 0
    }
    fn stream_len(&self) -> usize {
        // SAFETY: non-inline lists have an initialized, aligned prefix word.
        unsafe { self.allocation_ptr().cast::<usize>().read() }
    }
    fn stream_capacity(&self) -> usize {
        if self.multi() || self.reserved() {
            // SAFETY: both allocation forms initialize the second prefix word.
            unsafe { self.allocation_ptr().cast::<usize>().add(1).read() }
        } else {
            self.stream_len()
        }
    }
    fn group_capacity(&self) -> usize {
        if self.multi() {
            // SAFETY: multi-group buffers initialize the third prefix word.
            unsafe { self.allocation_ptr().cast::<usize>().add(2).read() }
        } else {
            1
        }
    }
    fn data_ptr(&self) -> *mut u8 {
        let prefix = prefix_bytes(self.group_capacity(), self.reserved())
            .expect("the allocation constructor checked the prefix size");
        // SAFETY: prefix size belongs to this validated allocation layout.
        unsafe { self.allocation_ptr().add(prefix) }
    }
    fn group_offset(&self, group: usize) -> usize {
        if self.multi() {
            // SAFETY: cursor ranges restrict group to an initialized directory entry.
            unsafe { self.allocation_ptr().cast::<usize>().add(3 + group).read() }
        } else {
            0
        }
    }
    fn set_group_offset(&mut self, group: usize, offset: usize) {
        if self.multi() {
            // SAFETY: construction/append reserve this group before publishing it.
            unsafe {
                self.allocation_ptr()
                    .cast::<usize>()
                    .add(3 + group)
                    .write(offset);
            }
        }
    }
    fn allocate(
        count: usize,
        groups: usize,
        capacity: usize,
        used: usize,
        reserved: bool,
        lease: &AllocationLease<'arena>,
    ) -> Result<Self> {
        let allocation = layout(groups, capacity, reserved)?;
        let (ptr, arena) = if let Some(ptr) = lease.allocate(allocation)? {
            (ptr.as_ptr(), true)
        } else {
            // SAFETY: allocation is a nonzero validated layout.
            let ptr = unsafe { alloc(allocation) };
            if ptr.is_null() {
                return Err(StorageError("position allocation failed"));
            }
            (ptr, false)
        };
        // SAFETY: the layout reserves these aligned prefix words; the stream
        // and directory are initialized before the list leaves its constructor.
        unsafe {
            ptr.cast::<usize>().write(used);
            if groups > 1 || reserved {
                ptr.cast::<usize>().add(1).write(capacity);
            }
            if groups > 1 {
                ptr.cast::<usize>().add(2).write(groups);
            }
        }
        Ok(Self {
            count_and_flags: count,
            payload: ptr.map_addr(|address| {
                address
                    | (usize::from(arena) * ARENA)
                    | (usize::from(groups > 1) * MULTI)
                    | (usize::from(reserved) * RESERVED)
            }),
            arena_lifetime: PhantomData,
        })
    }
    /// Construct directly in the final allocation from exactly count
    /// positions in nonincreasing order.
    ///
    /// The replayable iterator is traversed once to validate and measure the
    /// exact encoded size, then again to write the final stream. No scratch
    /// buffer or intermediate encoded copy is used.
    ///
    /// # Errors
    /// Returns an error for incorrect length, increasing input, inconsistent
    /// replay size, or allocation failure.
    pub(in super::super) fn from_reversed_iter_direct(
        count: usize,
        input: impl Iterator<Item = u64> + Clone,
        lease: &AllocationLease<'arena>,
    ) -> Result<Self> {
        if count <= 2 {
            return Self::from_reversed_iter(
                count,
                input,
                &mut PositionEncodingScratch::default(),
                lease,
            );
        }
        if count >= INLINE {
            return Err(StorageError("position count exceeds resident bounds"));
        }

        let used = encoded_size(0, count, 0, input.clone())?;
        let groups = count
            .checked_add(RESTART_INTERVAL - 1)
            .ok_or(StorageError("position count exceeds resident bounds"))?
            / RESTART_INTERVAL;
        let mut result = Self::allocate(count, groups, used, used, false, lease)?;
        result.fill_suffix(0, count, 0, input, 0, used)?;
        result.publish_suffix(count, used);
        Ok(result)
    }

    /// Build exactly count nonincreasing positions, checking length and every gap.
    pub(in super::super) fn from_reversed_iter(
        count: usize,
        mut input: impl Iterator<Item = u64>,
        scratch: &mut PositionEncodingScratch,
        lease: &AllocationLease<'arena>,
    ) -> Result<Self> {
        if count == 0 {
            if input.next().is_some() {
                return Err(StorageError("position run exceeds its declared length"));
            }
            return Ok(Self::new());
        }
        if count >= INLINE {
            return Err(StorageError("position count exceeds resident bounds"));
        }
        if count <= 2 {
            let last = input
                .next()
                .ok_or(StorageError("position run ended early"))?;
            let first = if count == 1 {
                last
            } else {
                input
                    .next()
                    .ok_or(StorageError("position run ended early"))?
            };
            if input.next().is_some() {
                return Err(StorageError("position run exceeds its declared length"));
            }
            let gap = last
                .checked_sub(first)
                .ok_or(StorageError("positions are not sorted"))?;
            if first <= usize::MAX as u64 && (count == 1 || gap <= DELTA_MASK as u64) {
                return Ok(Self {
                    count_and_flags: INLINE | if count == 2 { PAIR | gap as usize } else { 0 },
                    payload: std::ptr::without_provenance_mut(first as usize),
                    arena_lifetime: PhantomData,
                });
            }
            let bytes = 8 + if count == 2 { varint_bytes(gap) } else { 0 };
            let result = Self::allocate(count, 1, bytes, bytes, false, lease)?;
            // SAFETY: the exact buffer has eight bytes for the seed and the
            // checked encoded width for an optional second-position gap.
            unsafe {
                result.data_ptr().cast::<u64>().write_unaligned(first);
                if count == 2 {
                    write_varint(result.data_ptr().add(8), gap);
                }
            }
            return Ok(result);
        }
        let begin = scratch.encode_reverse(0, count, 0, input)?;
        let used = scratch.stream.capacity() - begin;
        let mut result = Self::allocate(
            count,
            count.div_ceil(RESTART_INTERVAL),
            used,
            used,
            false,
            lease,
        )?;
        for (group, &offset) in scratch.restarts.iter().rev().enumerate() {
            result.set_group_offset(group, offset - begin);
        }
        // SAFETY: the scratch suffix is fully initialized; the new buffer is
        // disjoint, has exactly used stream bytes, and is not yet published.
        unsafe {
            std::ptr::copy_nonoverlapping(
                scratch.stream.as_ptr().add(begin),
                result.data_ptr(),
                used,
            );
        }
        Ok(result)
    }
    /// Build from a sorted run, without copying the positions to a temporary array.
    /// The iterator is consumed once, from its back.
    ///
    /// # Errors
    /// Returns an error for decreasing values, inconsistent run length, or allocation failure.
    pub(in super::super) fn from_sorted_iter(
        input: impl DoubleEndedIterator<Item = u64> + ExactSizeIterator,
        scratch: &mut PositionEncodingScratch,
        lease: &AllocationLease<'arena>,
    ) -> Result<Self> {
        Self::from_reversed_iter(input.len(), input.rev(), scratch, lease)
    }

    /// Build from a nondecreasing slice. Duplicate values are accepted.
    ///
    /// # Errors
    /// Returns an error for decreasing values or allocation failure.
    #[cfg(test)]
    pub(in super::super) fn from_sorted(
        positions: &[u64],
        scratch: &mut PositionEncodingScratch,
        lease: &AllocationLease<'arena>,
    ) -> Result<Self> {
        Self::from_sorted_iter(positions.iter().copied(), scratch, lease)
    }
    /// Append a sorted run whose values are at least the current last value.
    /// The iterator is cloned to measure, then replayed. An error leaves the previous list intact.
    ///
    /// # Errors
    /// Returns an error for decreasing values, inconsistent run length, or allocation failure.
    #[cfg(test)]
    fn append_sorted_iter(
        &mut self,
        input: impl DoubleEndedIterator<Item = u64> + ExactSizeIterator + Clone,
        scratch: &mut PositionEncodingScratch,
        lease: &AllocationLease<'arena>,
    ) -> Result<()> {
        self.append_reverse(input.len(), input.rev(), scratch, lease)
    }

    /// Build from sorted chains, merging overlapping runs and retaining duplicates.
    /// Disjoint runs concatenate without a merge heap.
    ///
    /// # Errors
    /// Returns an error for decreasing positions, count overflow, or allocation failure.
    #[cfg(test)]
    fn from_chains(
        sources: &[(&super::PositionChains, super::PositionChain)],
        scratch: &mut PositionEncodingScratch,
        lease: &AllocationLease<'arena>,
    ) -> Result<Self> {
        Self::from_reversed_chains(sources.iter().rev().copied(), scratch, lease)
    }

    /// Build from replayable chain sources supplied in reverse producer order.
    /// Spatially disjoint sources concatenate; interleaved sources merge while
    /// retaining duplicates. The sources need not be collected into a slice.
    ///
    /// # Errors
    /// Returns an error for decreasing chain values, count overflow, or allocation failure.
    pub(in super::super) fn from_reversed_chains<'s>(
        sources: impl Iterator<Item = (&'s super::PositionChains, super::PositionChain)> + Clone,
        scratch: &mut PositionEncodingScratch,
        lease: &AllocationLease<'arena>,
    ) -> Result<Self> {
        let mut previous = None;
        let mut ordered = true;
        let mut count = 0_usize;
        for (owner, chain) in sources.clone() {
            count = count
                .checked_add(chain.len())
                .ok_or(StorageError("position count exceeds resident bounds"))?;
            if !chain.is_empty() {
                ordered &= previous.is_none_or(|first| owner.last(chain) <= Some(first));
                previous = owner.first(chain);
            }
        }
        // PERF: Spatially ordered producers already supply disjoint runs. Encode
        // their reverse concatenation directly, without allocating run metadata.
        if ordered {
            return Self::from_reversed_iter(
                count,
                sources.flat_map(|(owner, chain)| owner.reversed(chain)),
                scratch,
                lease,
            );
        }
        let mut sources: Vec<_> = sources.filter(|(_, chain)| !chain.is_empty()).collect();
        sources.sort_unstable_by_key(|&(owner, chain)| std::cmp::Reverse(owner.first(chain)));
        let disjoint = sources
            .windows(2)
            .all(|runs| runs[1].0.last(runs[1].1) <= runs[0].0.first(runs[0].1));
        let runs: Vec<_> = sources
            .into_iter()
            .map(|(owner, chain)| owner.reversed(chain))
            .collect();
        if disjoint {
            Self::from_reversed_iter(count, runs.into_iter().flatten(), scratch, lease)
        } else {
            Self::from_reversed_iter(count, DescendingMerge::new(runs), scratch, lease)
        }
    }

    fn append_reverse(
        &mut self,
        count: usize,
        input: impl Iterator<Item = u64> + Clone,
        scratch: &mut PositionEncodingScratch,
        lease: &AllocationLease<'arena>,
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        let start = self.len();
        let end = start
            .checked_add(count)
            .filter(|&end| end < INLINE)
            .ok_or(StorageError("position count exceeds resident bounds"))?;
        if self.is_inline() {
            let mut previous = [0; 2];
            self.iter().decode_into(&mut previous[..start]);
            let result = Self::from_reversed_iter(
                end,
                input.chain(previous[..start].iter().rev().copied()),
                scratch,
                lease,
            )?;
            *self = result;
            return Ok(());
        }
        let previous = self.get(start - 1);
        // PERF: Measure the immutable run, then replay it into destination storage.
        // This visits input twice and avoids a temporary encoded allocation.
        let added = encoded_size(start, end, previous, input.clone())?;
        let old_used = self.stream_len();
        let used = old_used
            .checked_add(added)
            .ok_or(StorageError("position stream size overflow"))?;
        let need_groups = end.div_ceil(RESTART_INTERVAL);
        let groups = self.group_capacity();
        let capacity = self.stream_capacity();
        if need_groups <= groups && used <= capacity && (self.multi() || self.reserved()) {
            self.fill_suffix(start, end, previous, input, old_used, used)?;
            self.publish_suffix(end, used);
            return Ok(());
        }
        let groups = if need_groups <= groups {
            groups
        } else {
            need_groups.max(groups.saturating_mul(2))
        };
        let capacity = if used <= capacity {
            capacity
        } else {
            used.max(capacity.saturating_mul(2))
        };
        let mut result = Self::allocate(end, groups, capacity, old_used, true, lease)?;
        // SAFETY: both buffers are disjoint; the old stream is initialized and
        // the new capacity is at least its length. The old list remains readable.
        unsafe {
            std::ptr::copy_nonoverlapping(self.data_ptr(), result.data_ptr(), old_used);
        }
        for group in 0..start.div_ceil(RESTART_INTERVAL) {
            result.set_group_offset(group, self.group_offset(group));
        }
        result.fill_suffix(start, end, previous, input, old_used, used)?;
        result.publish_suffix(end, used);
        *self = result;
        Ok(())
    }
    fn fill_suffix(
        &mut self,
        start: usize,
        end: usize,
        previous: u64,
        input: impl Iterator<Item = u64>,
        old_used: usize,
        used: usize,
    ) -> Result<()> {
        let mut cursor = used;
        let data = self.data_ptr();
        reverse_codes(start, end, previous, input, |index, code| {
            let seed = index.is_multiple_of(RESTART_INTERVAL);
            let bytes = if seed { 8 } else { varint_bytes(code) };
            cursor = cursor
                .checked_sub(bytes)
                .filter(|&offset| offset >= old_used)
                .ok_or(StorageError("position producer changed during replay"))?;
            // SAFETY: checked bounds keep writes in the unpublished suffix.
            // Directory entries belong to new groups. The published prefix and
            // count remain intact if replay returns an error or panics.
            unsafe {
                let target = data.add(cursor);
                if seed {
                    self.set_group_offset(index / RESTART_INTERVAL, cursor);
                    target.cast::<u64>().write_unaligned(code);
                } else {
                    write_varint(target, code);
                }
            }
            Ok(())
        })?;
        if cursor != old_used {
            return Err(StorageError("position producer changed during replay"));
        }
        Ok(())
    }
    fn publish_suffix(&mut self, end: usize, used: usize) {
        // SAFETY: the stream and new directory entries have been initialized;
        // changing this prefix and count publishes them under exclusive access.
        unsafe {
            self.allocation_ptr().cast::<usize>().write(used);
        }
        self.count_and_flags = end;
    }
    /// Append a nondecreasing slice. An error leaves the old list intact.
    ///
    /// # Errors
    /// Returns an error for decreasing values or allocation failure.
    #[cfg(test)]
    fn append_sorted(
        &mut self,
        positions: &[u64],
        scratch: &mut PositionEncodingScratch,
        lease: &AllocationLease<'arena>,
    ) -> Result<()> {
        self.append_sorted_iter(positions.iter().copied(), scratch, lease)
    }

    /// Append an owned list whose values are at least the current last value.
    /// Empty destinations take its allocation directly. Decoding caches one
    /// restart group while measuring and replaying the compressed source.
    ///
    /// # Errors
    /// Returns an error for decreasing values, count overflow, or allocation failure.
    pub(in super::super) fn append(
        &mut self,
        other: Self,
        scratch: &mut PositionEncodingScratch,
        lease: &AllocationLease<'arena>,
    ) -> Result<()> {
        if self.is_empty() {
            *self = other;
            return Ok(());
        }
        let mut index = other.len();
        let source = &other;
        let mut cache = [0_u64; RESTART_INTERVAL];
        let mut cached_begin = usize::MAX;
        let input = (0..other.len()).map(move |_| {
            index -= 1;
            if source.is_inline() {
                return source.get(index);
            }
            let begin = index / RESTART_INTERVAL * RESTART_INTERVAL;
            if begin != cached_begin {
                let end = (begin + RESTART_INTERVAL).min(source.len());
                source
                    .cursor(begin..end)
                    .decode_into(&mut cache[..end - begin]);
                cached_begin = begin;
            }
            cache[index - begin]
        });
        self.append_reverse(other.len(), input, scratch, lease)
    }
    fn get(&self, index: usize) -> u64 {
        self.cursor(index..index + 1)
            .next()
            .expect("the requested list element exists")
    }
    /// Return the first index whose coordinate is at least `position`.
    /// Searches restart seeds, then decodes at most one 128-element group.
    pub(in super::super) fn lower_bound(&self, position: u64) -> usize {
        if self.is_empty() {
            return 0;
        }
        let mut low = 0;
        let mut high = self.len().div_ceil(RESTART_INTERVAL);
        while low < high {
            let middle = low + (high - low) / 2;
            if self.get(middle * RESTART_INTERVAL) < position {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        let begin = low.saturating_sub(1) * RESTART_INTERVAL;
        let end = (begin + RESTART_INTERVAL).min(self.len());
        self.cursor(begin..end)
            .position(|value| value >= position)
            .map_or(end, |offset| begin + offset)
    }
    /// Read an independent range of list-element indices.
    ///
    /// # Panics
    /// Panics if the range is outside the list.
    #[inline]
    pub(in super::super) fn cursor(&self, range: Range<usize>) -> PositionCursor<'_, 'arena> {
        assert!(range.start <= range.end && range.end <= self.len());
        let mut cursor = PositionCursor {
            positions: self,
            index: range.start,
            end: range.end,
            source: std::ptr::null(),
            value: 0,
        };
        if range.start < range.end
            && !self.is_inline()
            && !range.start.is_multiple_of(RESTART_INTERVAL)
        {
            // SAFETY: the validated range selects an initialized seed group;
            // every skipped varint was emitted by the encoder in that group.
            unsafe {
                cursor.source = self
                    .data_ptr()
                    .add(self.group_offset(range.start / RESTART_INTERVAL));
                cursor.value = cursor.source.cast::<u64>().read_unaligned();
                cursor.source = cursor.source.add(8);
                for _ in 1..range.start % RESTART_INTERVAL {
                    cursor.value += read_varint(&mut cursor.source);
                }
            }
        }
        cursor
    }
    /// Read all positions with independent decoder state.
    #[inline]
    pub(in super::super) fn iter(&self) -> PositionCursor<'_, 'arena> {
        self.cursor(0..self.len())
    }
}

struct DescendingMerge<I> {
    runs: Vec<I>,
    heads: BinaryHeap<(u64, usize)>,
}
impl<I: Iterator<Item = u64>> DescendingMerge<I> {
    fn new(mut runs: Vec<I>) -> Self {
        let heads = runs
            .iter_mut()
            .enumerate()
            .filter_map(|(index, run)| run.next().map(|position| (position, index)))
            .collect();
        Self { runs, heads }
    }
}
impl<I: Iterator<Item = u64>> Iterator for DescendingMerge<I> {
    type Item = u64;
    // PERF: Inline the iterator step into tight consumer loops to avoid a
    // function call for every position.
    #[inline]
    fn next(&mut self) -> Option<u64> {
        let (position, index) = self.heads.peek().copied()?;
        if let Some(next) = self.runs[index].next() {
            *self.heads.peek_mut().expect("the observed run head exists") = (next, index);
        } else {
            self.heads.pop();
        }
        Some(position)
    }
}

/// An independent sequential decoder over a list-element range.
pub(in super::super) struct PositionCursor<'list, 'arena> {
    positions: &'list SortedPositions<'arena>,
    index: usize,
    end: usize,
    source: *const u8,
    value: u64,
}
impl PositionCursor<'_, '_> {
    /// Decode at most output.len() positions and return the number written.
    #[inline]
    pub(in super::super) fn decode_into(&mut self, output: &mut [u64]) -> usize {
        let count = output.len().min(self.end - self.index);
        for slot in &mut output[..count] {
            *slot = self
                .next()
                .expect("the cursor has count remaining elements");
        }
        count
    }
}
impl Iterator for PositionCursor<'_, '_> {
    type Item = u64;
    // PERF: Inline the decoder step into tight consumer loops to avoid a
    // function call for every position.
    #[inline]
    fn next(&mut self) -> Option<u64> {
        if self.index == self.end {
            return None;
        }
        let value = if self.positions.is_inline() {
            self.positions.payload.addr() as u64
                + if self.index == 1 {
                    (self.positions.count_and_flags & DELTA_MASK) as u64
                } else {
                    0
                }
        } else {
            // SAFETY: range indices never exceed the list count. Its encoder
            // initialized every seed and terminated gap, and append preserves it.
            unsafe {
                if self.index.is_multiple_of(RESTART_INTERVAL) {
                    self.source = self
                        .positions
                        .data_ptr()
                        .add(self.positions.group_offset(self.index / RESTART_INTERVAL));
                    self.value = self.source.cast::<u64>().read_unaligned();
                    self.source = self.source.add(8);
                } else {
                    self.value += read_varint(&mut self.source);
                }
            }
            self.value
        };
        self.index += 1;
        Some(value)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end - self.index;
        (remaining, Some(remaining))
    }
}
impl ExactSizeIterator for PositionCursor<'_, '_> {}
impl std::iter::FusedIterator for PositionCursor<'_, '_> {}
impl Drop for SortedPositions<'_> {
    fn drop(&mut self) {
        if !self.is_inline() && self.payload.addr() & ARENA == 0 {
            let allocation = layout(
                self.group_capacity(),
                self.stream_capacity(),
                self.reserved(),
            )
            .expect("the allocation constructor validated this layout");
            // SAFETY: this list owns the buffer; the tag excludes arena storage.
            // Its original layout is stored in its prefix and is unchanged by moves.
            unsafe {
                dealloc(self.allocation_ptr(), allocation);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn verify(positions: &SortedPositions<'_>, expected: &[u64]) {
        assert_eq!(positions.iter().collect::<Vec<_>>(), expected);
        for position in expected
            .iter()
            .copied()
            .chain([0, 1, 1 << 32, 1 << 63, u64::MAX])
        {
            assert_eq!(
                positions.lower_bound(position),
                expected.partition_point(|&value| value < position)
            );
        }
        for batch in [1, 17, 127, 128, 129, 509] {
            let mut output = vec![0; expected.len()];
            for begin in (0..expected.len()).step_by(batch) {
                let end = (begin + batch).min(expected.len());
                assert_eq!(
                    positions
                        .cursor(begin..end)
                        .decode_into(&mut output[begin..end]),
                    end - begin
                );
            }
            assert_eq!(output, expected);
        }
    }
    #[test]
    fn direct_reversed_encoding_matches_sorted_oracle_and_checks_replay() {
        let arena = super::super::AllocationArena::new(1, 4096);
        let lease = arena.lease(0);
        for values in [
            vec![],
            vec![u64::MAX],
            vec![0, u64::MAX],
            vec![u64::MAX; 2],
            (0..129).map(|index| index as u64).collect(),
            (0..257)
                .map(|index| {
                    if index < 128 {
                        index as u64
                    } else if index < 256 {
                        (1_u64 << 63) + (index - 128) as u64
                    } else {
                        u64::MAX
                    }
                })
                .collect(),
        ] {
            let direct = SortedPositions::from_reversed_iter_direct(
                values.len(),
                values.iter().rev().copied(),
                &lease,
            )
            .unwrap();
            verify(&direct, &values);
        }

        let short = SortedPositions::from_reversed_iter_direct(129, (0_u64..128).rev(), &lease);
        assert!(short.is_err());
        let unsorted =
            SortedPositions::from_reversed_iter_direct(3, [3_u64, 1, 2].into_iter(), &lease);
        assert!(unsorted.is_err());

        #[derive(Clone)]
        struct ChangingIter {
            pass: std::sync::Arc<std::sync::atomic::AtomicUsize>,
            index: usize,
            mode: Option<usize>,
            second_mode: usize,
        }
        impl Iterator for ChangingIter {
            type Item = u64;
            fn next(&mut self) -> Option<Self::Item> {
                if self.index == 129 {
                    return None;
                }
                if self.mode.is_none() {
                    let pass = self.pass.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    self.mode = Some(if pass == 0 { 0 } else { self.second_mode });
                }
                let mode = self.mode.expect("the iterator mode is initialized");
                let index = self.index;
                self.index += 1;
                Some(match mode {
                    0 => 0,
                    1 => index as u64,
                    _ => u64::MAX - (index as u64) * (1_u64 << 56),
                })
            }
        }
        let changing = |second_mode| ChangingIter {
            pass: std::sync::Arc::default(),
            index: 0,
            mode: None,
            second_mode,
        };
        let result = SortedPositions::from_reversed_iter_direct(129, changing(1), &lease);
        assert!(result.is_err(), "second pass sorting must be checked");
        let result = SortedPositions::from_reversed_iter_direct(129, changing(2), &lease);
        assert!(result.is_err(), "second pass write bounds must be checked");
    }

    #[test]
    fn full_coordinates_duplicates_and_restart_ranges() {
        let arena = super::super::AllocationArena::new(1, 4096);
        let lease = arena.lease(0);
        let mut scratch = PositionEncodingScratch::default();
        for values in [
            vec![],
            vec![u64::MAX],
            vec![0, u64::MAX],
            vec![u64::MAX, u64::MAX],
            vec![1 << 32; 1027],
            (0..777).map(|i| (1 << 63) + i * 17).collect(),
            vec![0, (1 << 32) - 1, 1 << 32, (1 << 63) - 1, 1 << 63, u64::MAX],
        ] {
            let positions = SortedPositions::from_reversed_iter(
                values.len(),
                values.iter().rev().copied(),
                &mut scratch,
                &lease,
            )
            .unwrap();
            verify(&positions, &values);
            assert!(
                SortedPositions::from_reversed_iter(
                    values.len() + 1,
                    values.iter().rev().copied(),
                    &mut scratch,
                    &lease,
                )
                .is_err()
            );
        }
    }
    #[test]
    fn append_preserves_prefix_across_inline_restarts_and_growth() {
        let arena = super::super::AllocationArena::new(1, 4096);
        let lease = arena.lease(0);
        let mut scratch = PositionEncodingScratch::default();
        let mut values: Vec<_> = (0..1027).map(|i| (1 << 32) + i * 13).collect();
        values[1026] = u64::MAX;
        for split in [0, 1, 2, 127, 128, 129, 256, 511, 1027] {
            let mut positions =
                SortedPositions::from_sorted(&values[..split], &mut scratch, &lease).unwrap();
            positions
                .append_sorted(&values[split..], &mut scratch, &lease)
                .unwrap();
            verify(&positions, &values);
            let mut positions =
                SortedPositions::from_sorted(&values[..split], &mut scratch, &lease).unwrap();
            let suffix =
                SortedPositions::from_sorted(&values[split..], &mut scratch, &lease).unwrap();
            positions.append(suffix, &mut scratch, &lease).unwrap();
            verify(&positions, &values);
        }
    }
    #[test]
    fn releasing_lease_keeps_lists_alive_and_allows_parallel_readers() {
        let arena = super::super::AllocationArena::new(1, 4096);
        let mut scratch = PositionEncodingScratch::default();
        // Exercise arena and individually owned buffers across restart groups.
        for count in [180, 2048] {
            let values: Vec<_> = (0..count).map(|i| (1_u64 << 63) + i * 113).collect();
            let positions = {
                let lease = arena.lease(0);
                SortedPositions::from_sorted(&values, &mut scratch, &lease).unwrap()
            };
            assert!(!positions.is_inline());
            assert_eq!(positions.payload.addr() & ARENA != 0, count == 180);
            std::thread::scope(|scope| {
                for _ in 0..4 {
                    scope.spawn(|| verify(&positions, &values));
                }
            });
            let lease = arena.lease(0);
            let another = SortedPositions::from_sorted(&[2, 4, 8], &mut scratch, &lease).unwrap();
            verify(&another, &[2, 4, 8]);
            verify(&positions, &values);
        }
    }
    #[test]
    fn decreasing_seed_boundaries_and_append_fail_without_publication() {
        let arena = super::super::AllocationArena::new(1, 4096);
        let lease = arena.lease(0);
        let mut scratch = PositionEncodingScratch::default();
        let mut values: Vec<_> = (0..129).collect();
        values[128] = 0;
        assert!(SortedPositions::from_sorted(&values, &mut scratch, &lease).is_err());
        let prefix: Vec<_> = (0..128).collect();
        let mut positions = SortedPositions::from_sorted(&prefix, &mut scratch, &lease).unwrap();
        assert!(
            positions
                .append_sorted(&[1, 2, 3], &mut scratch, &lease)
                .is_err()
        );
        verify(&positions, &prefix);
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let input = (128_usize..145).map(|i| {
                if i == 135 {
                    panic!("producer panic");
                }
                i as u64
            });
            positions
                .append_sorted_iter(input, &mut scratch, &lease)
                .unwrap();
        }));
        assert!(panic.is_err());
        verify(&positions, &prefix);
        positions
            .append_sorted(&[128, 129, 130], &mut scratch, &lease)
            .unwrap();
        assert_eq!(positions.len(), 131);
        // Replay can fail after writing part of an unpublished suffix. Exercise
        // both an allocation-growing append and reuse of reserved capacity.
        for reserved in [false, true] {
            for producer_panics in [false, true] {
                let mut positions =
                    SortedPositions::from_sorted(&prefix, &mut scratch, &lease).unwrap();
                if reserved {
                    positions
                        .append_sorted(&[128, 129], &mut scratch, &lease)
                        .unwrap();
                }
                let old: Vec<_> = positions.iter().collect();
                let first = old.len() as u64;
                let calls = std::cell::Cell::new(0);
                let replay = (0_usize..17).map(|offset| {
                    calls.set(calls.get() + 1);
                    if calls.get() == 26 {
                        if producer_panics {
                            panic!("producer panic during replay");
                        }
                        return u64::MAX;
                    }
                    first + offset as u64
                });
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    positions.append_sorted_iter(replay, &mut scratch, &lease)
                }));
                if producer_panics {
                    assert!(result.is_err());
                } else {
                    assert!(result.unwrap().is_err());
                }
                verify(&positions, &old);
                positions
                    .append_sorted(&[first, first + 1], &mut scratch, &lease)
                    .unwrap();
                verify(&positions, &[old, vec![first, first + 1]].concat());
            }
        }
    }
    #[test]
    fn chain_sources_concatenate_or_merge_without_losing_duplicates() {
        let arena = super::super::AllocationArena::new(1, 4096);
        let lease = arena.lease(0);
        let mut scratch = PositionEncodingScratch::default();
        for runs in [
            vec![vec![], vec![0, 1], vec![1, 1 << 32, u64::MAX]],
            vec![
                vec![0, 10, 1 << 63],
                vec![1, 10, u64::MAX],
                vec![2, 1 << 32],
            ],
        ] {
            let mut chains = super::super::PositionChains::new();
            let mut handles = Vec::new();
            for run in &runs {
                let mut chain = super::super::PositionChain::default();
                for &position in run {
                    chains.push(&mut chain, position).unwrap();
                }
                handles.push(chain);
            }
            let sources: Vec<_> = handles.into_iter().map(|chain| (&chains, chain)).collect();
            let positions = SortedPositions::from_chains(&sources, &mut scratch, &lease).unwrap();
            let mut expected: Vec<_> = runs.into_iter().flatten().collect();
            expected.sort_unstable();
            verify(&positions, &expected);
        }
    }
}
