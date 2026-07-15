use super::*;
use crate::bucket_idx::BucketIdx;
use rdst::RadixSort;
use std::time::Instant;

impl<Key: KeyT + ?Sized, BF: BucketFn, F: Packed, Hx: KeyHasher<Key>, const SINGLE_PART: bool, const REMAP: bool>
    PtrHash<Key, BF, F, Hx, Vec<u8>, SINGLE_PART, REMAP>
{
    /// Returns:
    /// 1. Hashes
    /// 2. Start indices of each bucket.
    /// 3. Order of the buckets within each part.
    ///
    /// This returns None if duplicate hashes are found.
    #[must_use]
    pub(super) fn sort_parts(
        &self,
        shard: usize,
        mut hashes: Vec<Hx::H>,
    ) -> Option<(Vec<Hx::H>, Vec<u32>)> {
        // For FastReduce methods, we can just sort by hash directly
        // instead of sorting by bucket id: For FR32L, first partition by those
        // <self.p1 and those >=self.p1, and then sort each group using the low
        // 32 bits.
        // NOTE: This does not work for other reduction methods.

        let start = Instant::now();
        // 2. Radix sort hashes.
        // HOT: This takes half the time for 128bit hashes.
        // TODO: Just append each hash to its part directly, where each part has
        //       space for exactly its number of slots.
        //
        // TODO: Write robinhood sort that inserts in the right place directly.
        // A) Sort L1 sized ranges.
        // B) Splat the front of each range to the next part of the target interval.
        hashes.radix_sort_unstable();
        let start = log_duration("┌ radix sort", start);

        // 3. Check duplicates.
        let distinct = hashes.par_windows(2).all(|w| w[0] != w[1]);
        let start = log_duration("├ check dups", start);
        if !distinct {
            eprintln!("Hashes are not distinct!");
            return None;
        }

        // 4. Find the start of each part using binary search.
        if !hashes.is_empty() {
            assert!(shard * self.parts_per_shard <= self.part(hashes[0]));
            assert!(self.part(*hashes.last().unwrap()) < (shard + 1) * self.parts_per_shard);
        }
        let mut part_starts = vec![0u32; self.parts_per_shard + 1];
        for part_in_shard in 1..=self.parts_per_shard {
            part_starts[part_in_shard] = hashes
                .binary_search_by(|h| {
                    if self.part(*h) < shard * self.parts_per_shard + part_in_shard {
                        std::cmp::Ordering::Less
                    } else {
                        std::cmp::Ordering::Greater
                    }
                })
                .unwrap_err() as u32;
        }

        // Check max part len.
        let mut max_part_len = 0;
        for (start, end) in part_starts.iter().tuple_windows() {
            let len = (end - start) as usize;
            max_part_len = max_part_len.max(len);
        }
        let exp = self.n / self.parts;
        let stddev = exp.isqrt();

        // https://math.stackexchange.com/a/89147/91741:
        // expected max of N (here #parts) samples of a random variable is
        // exp + sigma * sqrt(2 * ln N).
        let exp_max = exp + stddev * ((self.parts as f32).ln() * 2.).sqrt() as usize;
        trace!("exp key/part: {exp:>10} stddev {stddev:>10}");
        trace!(
            "exp max k/pt: {exp_max:>10}        {:>10} {:>8.2}",
            exp_max - exp,
            (exp_max - exp) as f32 / stddev as f32
        );
        trace!(
            "    max k/pt: {max_part_len:>10}        {:>10} {:>8.2}",
            max_part_len - exp,
            (max_part_len - exp) as f32 / stddev as f32
        );
        trace!(
            "    slots/pt: {:>10}        {:>10} {:>8.2}",
            self.slots,
            self.slots - exp,
            (self.slots - exp) as f32 / stddev as f32
        );
        trace!("exp    alpha: {:>13.2}%", 100. * self.params.alpha);
        trace!(
            "max    alpha: {:>13.2}%",
            100. * max_part_len as f32 / self.slots as f32
        );

        if max_part_len as usize > self.slots {
            trace!(
                    "Shard {shard}: Part has more elements than slots! elements {max_part_len} > {} slots",
                    self.slots
                );
            return None;
        }

        log_duration("├part starts", start);

        Some((hashes, part_starts))
    }

    // Sort the buckets in the given part and corresponding range of hashes.
    pub(super) fn sort_buckets(&self, part: usize, hashes: &[Hx::H]) -> (Vec<u32>, Vec<BucketIdx>) {
        // Where each bucket starts in hashes.
        let mut bucket_starts = Vec::with_capacity(self.buckets + 1);

        // The order of buckets, from large to small.
        let mut order: Vec<BucketIdx> = vec![BucketIdx::NONE; self.buckets];

        // The number of buckets of each length.
        let mut bucket_len_cnt = vec![0; 32];

        let mut end = 0;
        bucket_starts.push(end as u32);

        // Loop over buckets in part, setting start positions and counting # buckets of each size.
        for b in 0..self.buckets {
            let start = end;
            // NOTE: Many branch misses here.
            while end < hashes.len() && self.bucket(hashes[end]) == part * self.buckets + b {
                end += 1;
            }

            let l = end - start;
            if l >= bucket_len_cnt.len() {
                bucket_len_cnt.resize(l + 1, 0);
            }
            bucket_len_cnt[l] += 1;
            bucket_starts.push(end as u32);
        }

        assert_eq!(end, hashes.len());

        let max_bucket_size = bucket_len_cnt.len() - 1;
        // This assert is disabled, because it only holds when using uniform buckets.
        if false {
            let expected_bucket_size = self.slots as f32 / self.buckets as f32;
            assert!(max_bucket_size <= (20. * expected_bucket_size) as usize, "Part {part}: Bucket size {max_bucket_size} is too much larger than the expected size of {expected_bucket_size}." );
        }

        // Compute start positions of each range of buckets of equal size.
        let mut acc = 0;
        for i in (0..=max_bucket_size).rev() {
            let tmp = bucket_len_cnt[i];
            bucket_len_cnt[i] = acc;
            acc += tmp;
        }

        // Write buckets to their right location.
        for b in BucketIdx::range(self.buckets) {
            let l = (bucket_starts[b + 1] - bucket_starts[b]) as usize;
            order[bucket_len_cnt[l]] = b;
            bucket_len_cnt[l] += 1;
        }

        // for i in 0..max_bucket_size {
        //     if bucket_len_cnt[i] > 0 {
        //         eprintln!("Bucket size {:>2}: {:>6}", i, bucket_len_cnt[i]);
        //     }
        // }

        (bucket_starts, order)
    }
}
