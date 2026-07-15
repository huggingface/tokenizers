use super::*;
use crate::{bucket_idx::BucketIdx, stats::BucketStats};
use bitvec::{slice::BitSlice, vec::BitVec};
use log::warn;
use rayon::prelude::*;
use std::{
    collections::BinaryHeap,
    iter::zip,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
};

impl<Key: KeyT + ?Sized, BF: BucketFn, F: Packed, Hx: KeyHasher<Key>, const SINGLE_PART: bool, const REMAP: bool>
    PtrHash<Key, BF, F, Hx, Vec<u8>, SINGLE_PART, REMAP>
{
    pub(super) fn build_shard(
        &self,
        shard: usize,
        hashes: &[Hx::H],
        part_starts: &[u32],
        pilots: &mut [u8],
        taken: &mut [BitVec],
    ) -> Option<BucketStats> {
        let pilots_per_part = pilots.par_chunks_exact_mut(self.buckets);

        let iter = pilots_per_part.zip(taken).enumerate();

        // let total_evictions = AtomicUsize::new(0);
        let parts_done = AtomicUsize::new(shard * self.parts_per_shard);
        let stats = Mutex::new(BucketStats::new());

        let ok = iter.try_for_each(|(part_in_shard, (pilots, taken))| {
            let part = shard * self.parts_per_shard + part_in_shard;
            let hashes = &hashes
                [part_starts[part_in_shard] as usize..part_starts[part_in_shard + 1] as usize];
            let _cnt = self.build_part(part, hashes, pilots, taken, &stats)?;
            let _parts_done = parts_done.fetch_add(1, Ordering::Relaxed);
            // total_evictions.fetch_add(cnt, Ordering::Relaxed);

            // if self.params.print_stats {
            //     eprint!(
            //         "parts done: {parts_done:>6}/{:>6} ({:>4.1}%)\r",
            //         self.parts,
            //         100. * parts_done as f32 / self.parts as f32
            //     );
            // }

            Some(())
        });

        if ok.is_none() {
            return None;
        }

        assert_eq!(
            parts_done.load(Ordering::Relaxed),
            (shard + 1) * self.parts_per_shard
        );

        // let total_evictions: usize = total_evictions.load(Ordering::Relaxed);
        // let sum_pilots = pilots.iter().map(|&k| k as Pilot).sum::<Pilot>();

        // // Clear the last \r line.
        // if self.params.print_stats {
        //     eprint!("\x1b[K");
        //     eprintln!(
        //         "  displ./bkt: {:>14.3}",
        //         total_evictions as f32 / (self.buckets * self.parts_per_shard) as f32
        //     );
        //     eprintln!(
        //         "   avg pilot: {:>14.3}",
        //         sum_pilots as f32 / (self.buckets * self.parts_per_shard) as f32
        //     );
        // }

        // if self.params.print_stats {
        //     stats.lock().unwrap().print();
        // }

        Some(stats.into_inner().unwrap())
    }

    fn build_part(
        &self,
        part: usize,
        hashes: &[Hx::H],
        pilots: &mut [u8],
        taken: &mut BitSlice,
        _stats: &Mutex<BucketStats>,
    ) -> Option<usize> {
        let (starts, bucket_order) = self.sort_buckets(part, hashes);

        let kmax = 256;

        let mut slots = vec![BucketIdx::NONE; self.slots];
        let bucket_len = |b: BucketIdx| (starts[b + 1] - starts[b]) as usize;

        let max_bucket_len = bucket_len(bucket_order[0]);

        // First process larger buckets.
        // TODO: Use bucket queue instead?
        // NOTE: I tried 'rattle-kicking' where we prefer evicting buckets with a small pilot,
        //       but in practice this ends up slower, even though it saves ~15% of evictions.
        let mut stack = BinaryHeap::new();

        let slots_for_bucket = |b: BucketIdx, p: Pilot| unsafe {
            let hp = self.hash_pilot(p);
            hashes
                .get_unchecked(starts[b] as usize..starts[b + 1] as usize)
                .iter()
                .map(move |&hx| self.slot_in_part_hp(hx, hp))
        };
        let mut duplicate_slots = {
            let mut slots_tmp = vec![0; max_bucket_len];
            move |b: BucketIdx, p: Pilot| {
                slots_tmp.clear();
                slots_tmp.extend(slots_for_bucket(b, p));
                slots_tmp.sort_unstable();
                slots_tmp.iter().tuple_windows().any(|(a, b)| a == b)
            }
        };

        let mut recent = [BucketIdx::NONE; 16];
        let mut total_evictions = 0;

        let mut rng = fastrand::Rng::new();

        // let mut eviction_counts: Vec<usize> = vec![];

        for (i, &new_b) in bucket_order.iter().enumerate() {
            let new_bucket = &hashes[starts[new_b] as usize..starts[new_b + 1] as usize];
            if new_bucket.is_empty() {
                pilots[new_b] = 0;
                continue;
            }
            let new_b_len = new_bucket.len();

            let mut evictions = 0usize;

            stack.push((new_b_len, new_b));
            recent.fill(BucketIdx::NONE);
            let mut recent_idx = 0;
            recent[0] = new_b;

            'b: while let Some((_b_len, b)) = stack.pop() {
                if evictions > self.slots && evictions.is_power_of_two() {
                    // log = true;
                    let num_taken_slots = taken.count_ones();
                    // if self.params.print_stats {
                    //     eprintln!(
                    //         "part {part:>6} alpha {:>5.2}% bucket size {} ({}/{}, {:>5.2}%) slots filled {}/{} ({:>5.2}%) chain: {evictions:>9}",
                    //         100. * hashes.len()  as f32 / slots.len() as f32,
                    //         new_b_len,
                    //         i, self.buckets,
                    //         100. * i as f32 / self.buckets as f32,
                    //         num_taken_slots,
                    //         taken.len(),
                    //         100. * num_taken_slots as f32 / taken.len() as f32,
                    //     );
                    // }
                    if evictions >= 10 * self.slots {
                        warn!(
                            "\
Too many evictions. Aborting!
When the current bucket has size >=2, try decreasing lambda to use fewer elements per buckets.
When the current bucket has size 1 (or maybe 2), try decreasing alpha to have more empty slots for the last few buckets.

Current part: {part:>6} with load factor alpha={:>5.2}%
Current bucket: size {} ({}/{}, {:>5.2}%)
Slots filled so far: {}/{} ({:>5.2}%)
Eviction chain length: {evictions:>9}
",
                            100. * hashes.len()  as f32 / slots.len() as f32,
                            new_b_len,
                            i, self.buckets,
                            100. * i as f32 / self.buckets as f32,
                            num_taken_slots,
                            taken.len(),
                            100. * num_taken_slots as f32 / taken.len() as f32,
                        );
                        return None;
                    }
                }

                // 1a) Check for a solution without collisions.

                let bucket =
                    unsafe { hashes.get_unchecked(starts[b] as usize..starts[b + 1] as usize) };
                let b_slots =
                    |hp: PilotHash| bucket.iter().map(move |&hx| self.slot_in_part_hp(hx, hp));

                // 1b) Hot-path for when there are no collisions, which is most of the buckets.
                if let Some((p, hp)) = self.find_pilot(kmax, bucket, taken) {
                    // HOT: Many branch misses here.
                    pilots[b] = p as u8;
                    for p in b_slots(hp) {
                        unsafe {
                            // Taken is already filled by find_pilot.
                            // HOT: This is a hot instruction; takes as much time as finding the pilot.
                            *slots.get_unchecked_mut(p) = b;
                        }
                    }
                    continue 'b;
                }

                // 2) Search for a pilot with minimal number of collisions.

                // Start at a random pilot to prevent eviction cycles.
                let p0 = rng.u8(..) as u64;
                // (worst colliding bucket size, p)
                let mut best = (usize::MAX, u64::MAX);

                'p: for delta in 0u64..kmax {
                    // HOT: This code is slow and full of branch-misses.
                    // But also, it's only 20% of build_part() time, since the
                    // hot-path above covers most.
                    let p = (p0 + delta) % kmax;
                    let hp = self.hash_pilot(p);
                    let mut collision_score = 0;
                    for p in b_slots(hp) {
                        let s = unsafe { *slots.get_unchecked(p) };
                        // HOT: many branches
                        let new_score = if s.is_none() {
                            continue;
                        } else if recent.contains(&s) {
                            continue 'p;
                        } else {
                            // HOT: cache misses.
                            bucket_len(s).pow(2)
                        };
                        collision_score += new_score;
                        if collision_score >= best.0 {
                            continue 'p;
                        }
                    }

                    // This check takes 2% of time even though it almost
                    // always passes. Can we delay it to filling of the
                    // slots table, and backtrack if needed.
                    if !duplicate_slots(b, p) {
                        best = (collision_score, p);
                        // Since we already checked for a collision-free solution,
                        // the next best is a single collision of size b_len.
                        if collision_score == new_b_len * new_b_len {
                            break;
                        }
                    }
                }

                if best == (usize::MAX, u64::MAX) {
                    let slots = b_slots(0);
                    let len = bucket.len();
                    let num_slots = self.slots;
                    eprintln!(
                        "part {part}: bucket of size {len} with {num_slots} slots: Indistinguishable hashes in bucket!"
                    );
                    for (hx, slot) in zip(bucket, slots) {
                        eprintln!("{:x?} -> slot {slot}", hx);
                    }
                    eprintln!(
                        "part {part}: bucket of size {len} with {num_slots} slots: Indistinguishable hashes in bucket!"
                    );
                    return None;
                }

                let (_collision_score, p) = best;
                // if self.params.print_stats {
                //     eprintln!(
                //         "{evictions:>7} | pilots[{:>7}] = {:>3} len: {} stack: {} score: {:>3}",
                //         b.0,
                //         p,
                //         bucket_len(b),
                //         stack.len(),
                //         _collision_score
                //     );
                // }
                pilots[b] = p as u8;
                let hp = self.hash_pilot(p);

                // Drop the collisions and set the new pilot.
                for slot in b_slots(hp) {
                    // THIS IS A HOT INSTRUCTION.
                    let b2 = slots[slot];
                    if b2.is_some() {
                        assert!(b2 != b);
                        // DROP BUCKET b
                        // if self.params.print_stats {
                        //     eprintln!(
                        //         "{evictions:>7} | Push {:>7} len: {}",
                        //         b2.0,
                        //         bucket_len(b2)
                        //     );
                        // }
                        stack.push((bucket_len(b2), b2));
                        evictions += 1;
                        for p2 in slots_for_bucket(b2, pilots[b2] as Pilot) {
                            unsafe {
                                *slots.get_unchecked_mut(p2) = BucketIdx::NONE;
                                taken.set_unchecked(p2, false);
                            }
                        }
                    }
                    unsafe {
                        *slots.get_unchecked_mut(slot) = b;
                        taken.set_unchecked(slot, true);
                    }
                }

                recent_idx += 1;
                recent_idx %= recent.len();
                recent[recent_idx] = b;
            }
            total_evictions += evictions;
            // if self.params.print_stats {
            // eviction_counts.push(evictions);
            // }
        }

        // if self.params.print_stats {
        //     let mut stats = stats.lock().unwrap();
        //     for (i, &b) in bucket_order.iter().enumerate() {
        //         stats.add(
        //             i,
        //             bucket_order.len(),
        //             bucket_len(b),
        //             pilots[b] as Pilot,
        //             *eviction_counts.get(i).unwrap_or(&0),
        //         );
        //     }
        // }

        Some(total_evictions)
    }

    fn find_pilot(
        &self,
        kmax: u64,
        bucket: &[Hx::H],
        taken: &mut BitSlice,
    ) -> Option<(Pilot, PilotHash)> {
        // This gives ~10% speedup.
        match bucket.len() {
            1 => self.find_pilot_array::<1>(kmax, bucket.try_into().unwrap(), taken),
            2 => self.find_pilot_array::<2>(kmax, bucket.try_into().unwrap(), taken),
            3 => self.find_pilot_array::<3>(kmax, bucket.try_into().unwrap(), taken),
            4 => self.find_pilot_array::<4>(kmax, bucket.try_into().unwrap(), taken),
            5 => self.find_pilot_array::<5>(kmax, bucket.try_into().unwrap(), taken),
            6 => self.find_pilot_array::<6>(kmax, bucket.try_into().unwrap(), taken),
            7 => self.find_pilot_array::<7>(kmax, bucket.try_into().unwrap(), taken),
            8 => self.find_pilot_array::<8>(kmax, bucket.try_into().unwrap(), taken),
            _ => self.find_pilot_slice(kmax, bucket, taken),
        }
    }
    fn find_pilot_array<const L: usize>(
        &self,
        kmax: u64,
        bucket: &[Hx::H; L],
        taken: &mut BitSlice,
    ) -> Option<(Pilot, PilotHash)> {
        self.find_pilot_slice(kmax, bucket, taken)
    }

    // Note: Prefetching on `taken` is not needed because we use parts that fit in L1 cache anyway.
    //
    // Note: Tried looping over multiple pilots in parallel, but the additional
    // lookups this does aren't worth it.
    #[inline(always)]
    fn find_pilot_slice(
        &self,
        kmax: u64,
        bucket: &[Hx::H],
        taken: &mut BitSlice,
    ) -> Option<(Pilot, PilotHash)> {
        let r = bucket.len() / 4 * 4;
        'p: for p in 0u64..kmax {
            let hp = self.hash_pilot(p);
            // True when the slot for hx is already taken.
            let check = |hx| unsafe { *taken.get_unchecked(self.slot_in_part_hp(hx, hp)) };

            // Process chunks of 4 bucket elements at a time.
            // This reduces branch-misses (of all of build_part) 3-fold, giving 20% speedup.
            for i in (0..r).step_by(4) {
                // Check all 4 elements of the chunk without early break.
                // NOTE: It's hard to SIMD vectorize the `slot` computation
                // here because it uses 64x64->128bit multiplies.
                let checks: [bool; 4] = unsafe {
                    [
                        check(*bucket.get_unchecked(i)),
                        check(*bucket.get_unchecked(i + 1)),
                        check(*bucket.get_unchecked(i + 2)),
                        check(*bucket.get_unchecked(i + 3)),
                    ]
                };
                if checks.iter().any(|&bad| bad) {
                    continue 'p;
                }
            }
            // Check remaining elements.
            let mut bad = false;
            for &hx in &bucket[r..] {
                bad |= check(hx);
            }
            if bad {
                continue 'p;
            }

            if self.try_take_pilot(bucket, hp, taken) {
                return Some((p, hp));
            }
        }
        None
    }

    /// Fill `taken` with the slots for `hp`, but backtrack as soon as a
    /// collision within the bucket is found.
    ///
    /// Returns true on success.
    fn try_take_pilot(&self, bucket: &[Hx::H], hp: PilotHash, taken: &mut BitSlice) -> bool {
        // This bucket does not collide with previous buckets, but it may still collide with itself.
        for (i, &hx) in bucket.iter().enumerate() {
            let slot = self.slot_in_part_hp(hx, hp);
            if unsafe { *taken.get_unchecked(slot) } {
                // Collision within the bucket. Clean already set entries.
                for &hx in unsafe { bucket.get_unchecked(..i) } {
                    unsafe { taken.set_unchecked(self.slot_in_part_hp(hx, hp), false) };
                }
                return false;
            }
            unsafe { taken.set_unchecked(slot, true) };
        }
        true
    }
}
