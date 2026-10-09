// SPDX-License-Identifier: BSD-2-Clause
// Copyright (c) 2025, 2026 Robert Clausecker <clausecker@zib.de>
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

//! Rust translation of Clausecker's BSD-2-Clause `radixsort_permuted.c`.
//! Reference f69e816c3cd79d312cd67aea5b9cf1c338c1b371, July 2026 paper:
//! <https://arxiv.org/abs/2607.05302>; full upstream license retained above.
//! Sort complete keys; retain payloads in stable incoming order.
//! Full keys use twelve-byte records; bounded keys can use eight-byte records.
//! Both use the original 512-element block permutation.
//! Scatter scratch is 3 MiB for full records and 2 MiB for compact records;
//! metadata adds nine bytes per input block.
//! The fixed block size makes metadata grow with n/512; the paper's square-root
//! overhead bound does not describe this local parameterization.
//! Original source and license at the fixed reference revision:
//! <https://github.com/clausecker/radsort/blob/f69e816c3cd79d312cd67aea5b9cf1c338c1b371/radixsort_permuted.c>
//! <https://github.com/clausecker/radsort/blob/f69e816c3cd79d312cd67aea5b9cf1c338c1b371/COPYING>
use std::{marker::PhantomData, ptr};

/// Bounded records keep local block-permutation indices within u32.
pub(in super::super) const MAX_RECORDS: usize = 1 << 28;
const RADIX: usize = 256;
const BLOCK: usize = 512;
const SCRATCH: usize = 2 * RADIX;

pub(in super::super) trait RadixRecord: Copy + Default + Send + Sync {
    fn key(self) -> u64;
}

/// A full-width key and a bounded payload in twelve bytes.
///
/// Splitting the key into two words avoids the padding of `(u64, u32)`.
/// The payload is opaque to sorting; equal keys retain incoming payload order.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(in super::super) struct KeyedValue {
    key_low: u32,
    key_high: u32,
    value: u32,
}
impl KeyedValue {
    #[inline]
    pub(in super::super) const fn new(key: u64, value: u32) -> Self {
        Self {
            key_high: (key >> 32) as u32,
            key_low: key as u32,
            value,
        }
    }
    #[inline]
    pub(in super::super) const fn key(self) -> u64 {
        (self.key_high as u64) << 32 | self.key_low as u64
    }
    #[inline]
    pub(in super::super) const fn value(self) -> u32 {
        self.value
    }
}
const _: () = assert!(std::mem::size_of::<KeyedValue>() == 12);

/// A pair key whose token IDs each fit in sixteen bits, plus its wave offset.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(in super::super) struct CompactKeyedValue {
    key: u32,
    value: u32,
}
impl CompactKeyedValue {
    #[inline]
    pub(in super::super) const fn new(key: u64, value: u32) -> Self {
        let left = (key >> 32) as u32;
        let right = key as u32;
        debug_assert!(left <= u16::MAX as u32 && right <= u16::MAX as u32);
        Self {
            key: (left << 16) | right,
            value,
        }
    }
    #[inline]
    const fn key(self) -> u64 {
        (((self.key >> 16) as u64) << 32) | ((self.key & 0xffff) as u64)
    }
    #[inline]
    pub(in super::super) const fn value(self) -> u32 {
        self.value
    }
}
impl RadixRecord for KeyedValue {
    #[inline]
    fn key(self) -> u64 {
        self.key()
    }
}
impl RadixRecord for CompactKeyedValue {
    #[inline]
    fn key(self) -> u64 {
        self.key()
    }
}
const _: () = assert!(std::mem::size_of::<CompactKeyedValue>() == 8);

#[derive(Clone, Copy)]
struct Partial {
    index: usize,
    length: usize,
}
#[derive(Clone, Copy)]
struct Bucket<R> {
    next: *mut R,
    end: *mut R,
}
struct Sorter<'a, R> {
    records: *mut R,
    length: usize,
    _borrow: PhantomData<&'a mut [R]>,
    _scratch: Vec<R>,
    scratch_base: *mut R,
    perm: Vec<u32>,
    perm2: Vec<u32>,
    usage: Vec<u8>,
    partials: [Partial; RADIX],
    fill: usize,
}
impl<'a, R: RadixRecord> Sorter<'a, R> {
    fn new(records: &'a mut [R]) -> Self {
        let full = records.len() / BLOCK;
        let blocks = full + SCRATCH;
        assert!(blocks <= u32::MAX as usize);
        let fill = RADIX + full + 1;
        let mut perm = vec![0; blocks];
        for (i, p) in perm.iter_mut().enumerate() {
            *p = if i < RADIX {
                i
            } else if i < fill - 1 {
                i + RADIX
            } else {
                i - (fill - 1) + RADIX
            } as u32;
        }
        let mut scratch = vec![R::default(); SCRATCH * BLOCK];
        let tail = records.len() % BLOCK;
        scratch[RADIX * BLOCK..RADIX * BLOCK + tail].copy_from_slice(&records[full * BLOCK..]);
        let mut partials = [Partial {
            index: blocks,
            length: 0,
        }; RADIX];
        partials[0] = Partial {
            index: fill - 1,
            length: tail,
        };
        let length = records.len();
        let records = records.as_mut_ptr();
        let scratch_base = scratch.as_mut_ptr();
        Self {
            records,
            length,
            _borrow: PhantomData,
            _scratch: scratch,
            scratch_base,
            perm,
            perm2: vec![0; blocks],
            usage: vec![0; blocks],
            partials,
            fill,
        }
    }
    fn block(&mut self, physical: usize) -> *mut R {
        assert!(physical < self.perm.len());
        // SAFETY: physical IDs below SCRATCH identify full scratch blocks.
        // Other IDs identify exactly floor(n/BLOCK) full input blocks.
        // Vec allocations are never resized while these pointers are in use.
        unsafe {
            if physical < SCRATCH {
                self.scratch_base.add(physical * BLOCK)
            } else {
                self.records.add((physical - SCRATCH) * BLOCK)
            }
        }
    }
    fn length(&self, logical: usize, partial: &mut usize) -> usize {
        if *partial < RADIX && self.partials[*partial].index == logical {
            let length = self.partials[*partial].length;
            *partial += 1;
            length
        } else {
            BLOCK
        }
    }
    #[cfg(debug_assertions)]
    fn validate(&self) {
        let mut seen = vec![false; self.perm.len()];
        for &physical in &self.perm {
            assert!(!seen[physical as usize]);
            seen[physical as usize] = true;
        }
        assert!(self.fill <= self.perm.len());
        assert!(self.partials.windows(2).all(|p| p[0].index < p[1].index));
        assert!(
            self.partials
                .iter()
                .all(|p| p.index >= RADIX && p.index < self.fill && p.length < BLOCK)
        );
        let mut partial = 0;
        let total: usize = (RADIX..self.fill)
            .map(|i| self.length(i, &mut partial))
            .sum();
        assert_eq!(total, self.length);
    }
    fn step(&mut self, shift: u32) {
        let mut buckets = [Bucket::<R> {
            next: ptr::null_mut(),
            end: ptr::null_mut(),
        }; RADIX];
        let mut counts = [1_usize; RADIX];
        for (i, bucket) in buckets.iter_mut().enumerate() {
            let out = self.block(self.perm[i] as usize);
            // SAFETY: block() returns exactly BLOCK valid elements.
            *bucket = Bucket {
                next: out,
                end: unsafe { out.add(BLOCK) },
            };
            self.usage[i] = i as u8;
        }
        let mut output = RADIX;
        let mut partial = 0;
        for input in RADIX..self.fill {
            let source = self.block(self.perm[input] as usize);
            let length = self.length(input, &mut partial);
            for j in 0..length {
                // SAFETY: source is a valid block, length <= BLOCK. The stable
                // logical input traversal consumes each element exactly once.
                let value = unsafe { source.add(j).read() };
                let b = ((value.key() >> shift) & 255) as usize;
                // SAFETY: bucket.next points at an unused cell in its current
                // output block. It advances at most to end before reallocation.
                // Input blocks are recycled only after consumption: the RADIX
                // block head start establishes output <= input. If equality
                // occurs, allocation follows the last consumed source element;
                // the newly allocated block is written on a subsequent element.
                // More precisely, after J consumed values the newly reserved
                // logical block is R + sum(floor(count[b]/B)) - 1 <=
                // R + floor(J/B) - 1 <= current input. Equality requires
                // J to end a full input block, so no unread value is overwritten.
                // This adapts paper Lemma 1 to the reference C allocation timing.
                unsafe {
                    buckets[b].next.write(value);
                    buckets[b].next = buckets[b].next.add(1);
                }
                if buckets[b].next == buckets[b].end {
                    debug_assert!(output <= input);
                    let out = self.block(self.perm[output] as usize);
                    buckets[b] = Bucket {
                        next: out,
                        end: unsafe { out.add(BLOCK) },
                    };
                    self.usage[output] = b as u8;
                    counts[b] += 1;
                    output += 1;
                }
            }
        }
        self.fill = output;
        let mut starts = [RADIX; RADIX];
        for b in 1..RADIX {
            starts[b] = starts[b - 1] + counts[b - 1];
        }
        for i in 0..self.fill {
            let b = self.usage[i] as usize;
            let j = starts[b];
            starts[b] += 1;
            self.perm2[j] = self.perm[i];
            let base = self.block(self.perm[i] as usize);
            // SAFETY: next/end belong to the same BLOCK-sized allocation.
            if unsafe { base.add(BLOCK) } == buckets[b].end {
                self.partials[b] = Partial {
                    index: j,
                    length: unsafe { buckets[b].next.offset_from(base) as usize },
                };
            }
        }
        assert!(self.fill + RADIX <= self.perm.len());
        self.perm2[..RADIX].copy_from_slice(&self.perm[self.fill..self.fill + RADIX]);
        self.perm2[self.fill + RADIX..].copy_from_slice(&self.perm[self.fill + RADIX..]);
        self.fill += RADIX;
        std::mem::swap(&mut self.perm, &mut self.perm2);
        #[cfg(debug_assertions)]
        self.validate();
    }
    fn compact(&mut self) {
        for i in 0..self.perm.len() {
            self.perm2[self.perm[i] as usize] = i as u32;
        }
        let mut start = 0;
        let mut partial = 0;
        let mut free_logical = 0;
        let mut free_physical = self.perm[0] as usize;
        for destination in SCRATCH..self.perm.len() {
            let input = destination - RADIX;
            let mut output = self.perm2[destination] as usize;
            let source = self.perm[input] as usize;
            if output > input && output < self.fill {
                debug_assert!(free_logical < RADIX || free_logical >= self.fill);
                let from = self.block(destination);
                let to = self.block(free_physical);
                // SAFETY: both physical IDs are valid full blocks. Free block
                // receives live output which would otherwise be overwritten.
                unsafe {
                    ptr::copy(from, to, BLOCK);
                }
                self.perm[free_logical] = destination as u32;
                self.perm[output] = free_physical as u32;
                self.perm2[free_physical] = output as u32;
                self.perm2[destination] = free_logical as u32;
                free_logical = output;
                output = self.perm2[destination] as usize;
            }
            let length = self.length(input, &mut partial);
            let from = self.block(source);
            assert!(start + length <= self.length);
            // Completed partials can only shorten preceding output: start <=
            // (destination - SCRATCH) * BLOCK, and start + length never exceeds
            // this destination block's end. Any still-live destination block was
            // evacuated above; earlier physical destinations are already consumed.
            // SAFETY: length valid source elements are moved into the next
            // logical output interval. ptr::copy permits overlap as memmove.
            unsafe {
                ptr::copy(from, self.records.add(start), length);
            }
            self.perm[input] = destination as u32;
            self.perm[output] = source as u32;
            self.perm2[source] = output as u32;
            self.perm2[destination] = input as u32;
            start += length;
            if input != output {
                free_physical = source;
                free_logical = output;
            }
        }
        for input in self.perm.len() - RADIX..self.fill {
            let source = self.perm[input] as usize;
            assert!(source < SCRATCH);
            let length = self.length(input, &mut partial);
            let from = self.block(source);
            assert!(start + length <= self.length);
            unsafe {
                ptr::copy(from, self.records.add(start), length);
            }
            start += length;
        }
        assert_eq!(start, self.length);
    }
}
/// Sort records stably by their complete key.
///
/// Mutates records in place. Constant key bytes need no scatter pass. Block
/// scratch is 512 blocks of the selected record type plus nine bytes per 512
/// input records; small inputs use a
/// simpler scatter when its allocation is smaller. Scratch sizing follows the
/// selected record layout so compact records retain their memory advantage.
pub(in super::super) fn sort_by_key<R: RadixRecord>(records: &mut [R]) {
    assert!(
        records.len() <= MAX_RECORDS,
        "radix input exceeds bounded chunk size"
    );
    let Some(&first) = records.first() else {
        return;
    };
    let varying = records
        .iter()
        .fold(0, |bits, &r| bits | (r.key() ^ first.key()));
    if std::mem::size_of_val(records)
        <= SCRATCH * BLOCK * std::mem::size_of::<R>() + (records.len() / BLOCK + SCRATCH) * 9
    {
        sort_classic(records, varying)
    } else {
        sort_digits(records, varying)
    }
}
fn sort_classic<R: RadixRecord>(records: &mut [R], varying: u64) {
    if records.len() < 2 || varying == 0 {
        return;
    }
    let mut scratch = vec![R::default(); records.len()];
    let mut flipped = false;
    {
        let mut input = &mut records[..];
        let mut output = scratch.as_mut_slice();
        for shift in (0..64).step_by(8) {
            if (varying >> shift) & 255 == 0 {
                continue;
            }
            let mut counts = [0usize; RADIX];
            for &r in input.iter() {
                counts[((r.key() >> shift) & 255) as usize] += 1;
            }
            let mut offsets = [0usize; RADIX];
            let mut sum = 0;
            for (count, offset) in counts.iter().zip(offsets.iter_mut()) {
                *offset = sum;
                sum += count;
            }
            for &r in input.iter() {
                let b = ((r.key() >> shift) & 255) as usize;
                output[offsets[b]] = r;
                offsets[b] += 1;
            }
            std::mem::swap(&mut input, &mut output);
            flipped = !flipped;
        }
    }
    if flipped {
        records.copy_from_slice(&scratch);
    }
}
fn sort_digits<R: RadixRecord>(records: &mut [R], varying: u64) {
    if records.len() < 2 || varying == 0 {
        return;
    }
    let mut sorter = Sorter::new(records);
    for shift in (0..64).step_by(8) {
        if (varying >> shift) & 255 != 0 {
            sorter.step(shift);
        }
    }
    sorter.compact();
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_keys_keep_payload_order_across_dispatch_and_block_boundaries() {
        let extremes = [
            (u64::MAX, u32::MAX),
            (3, 0),
            ((1_u64 << 32) | 3, 1),
            ((1_u64 << 32) | 3, u32::MAX),
            (1_u64 << 63, 7),
        ];
        let mut records: Vec<_> = extremes
            .into_iter()
            .map(|(key, value)| {
                let record = KeyedValue::new(key, value);
                assert_eq!(record.key(), key);
                assert_eq!(record.value(), value);
                record
            })
            .collect();
        sort_by_key(&mut records);
        assert_eq!(
            records
                .iter()
                .map(|record| (record.key(), record.value()))
                .collect::<Vec<_>>(),
            [
                extremes[1],
                extremes[2],
                extremes[3],
                extremes[4],
                extremes[0]
            ]
        );
        for count in [
            0, 1, 511, 512, 513, 262_912, 262_913, 400_383, 400_384, 400_385,
        ] {
            let mut records: Vec<KeyedValue> = (0..count)
                .map(|i| {
                    let key = ((i as u64).wrapping_mul(0x9e3779b97f4a7c15) % 997)
                        ^ ((i as u64 % 11) << 60);
                    KeyedValue::new(key, i as u32)
                })
                .collect();
            let mut expected = records.clone();
            expected.sort_by_key(|r| r.key());
            sort_by_key(&mut records);
            assert_eq!(records, expected, "{count}");
        }
    }

    #[test]
    fn compact_sort_matches_stable_order_at_small_and_dispatch_boundaries() {
        for (left, right) in [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (u16::MAX as u32, u16::MAX as u32),
        ] {
            let key = (u64::from(left) << 32) | u64::from(right);
            let record = CompactKeyedValue::new(key, 17);
            assert_eq!(record.key(), key);
            assert_eq!(record.value(), 17);
        }
        for id in [u16::MAX as u32 + 1, u32::MAX] {
            let key = (u64::from(id) << 32) | 1;
            let record = KeyedValue::new(key, 23);
            assert_eq!(record.key(), key);
            assert_eq!(record.value(), 23);
        }
        for count in [0_usize, 1, 511, 512, 513, 263_298, 263_299] {
            // Exercise one, two and three varying bytes with repeated full keys.
            for left_limit in [1, 251, 65_536] {
                let mut records: Vec<_> = (0..count)
                    .map(|index| {
                        let sample = index % 997;
                        let left = sample * 73 % left_limit;
                        let right = sample * 37 % 251;
                        CompactKeyedValue::new(((left as u64) << 32) | right as u64, index as u32)
                    })
                    .collect();
                let max_key = (u64::from(u16::MAX) << 32) | u64::from(u16::MAX);
                if count > BLOCK * 3 {
                    for (index, record) in records.iter_mut().enumerate().skip(count - BLOCK * 3) {
                        *record = CompactKeyedValue::new(max_key, index as u32);
                    }
                }
                let mut expected = records.clone();
                expected.sort_by_key(|record| record.key());
                sort_by_key(&mut records);
                assert_eq!(records, expected, "count={count}, left_limit={left_limit}");
            }
        }
    }
}
