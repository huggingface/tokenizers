//! Multipass merging, for words below the gate.
//!
//! Each pass rewrites the word in place, merging the lowest-ranked pair and recording the lowest
//! pair of the result, which becomes the next pass's target. Read and write cursors share one
//! buffer, so a pass shortens it by one per merge applied.
//!
//! A pass merges *every* occurrence of that pair only when the pair is `SAFE`: batching is exact
//! only if the product cannot reach a merge cheaper than the one being applied, since such a merge
//! would be due before the pair's remaining occurrences. When it is not safe the pass stops after
//! the first occurrence, which costs a pass per occurrence but is what BPE actually does.
use crate::models::bpe::bpe_build_tables::{ID_MASK, SAFE};
use crate::models::bpe::bpe_model::PipelineBPE;
use std::cmp;

impl PipelineBPE {
    /// `M` is false only for the first written symbol, which has no left neighbour and therefore no
    /// pair to rank. `BATCH` merges every occurrence of `global_min`; without it only the first
    /// merges, which `merged` tracks.
    ///
    /// `&mut [u32]` rather than `&mut Vec<u32>` so the length is a local and the reads can have
    /// their bounds checks removed..
    #[inline(always)]
    fn advance_one<const M: bool, const BATCH: bool>(
        &self,
        to_merge: &mut [u32],
        mut read_id: usize,
        global_min: u64,
        mut write_id: usize,
        mut running_min: u64,
        mut merged: bool,
    ) -> (u64, usize, usize, bool) {
        let (ia, ib) = (to_merge[read_id], to_merge[read_id + 1]);
        let value = self.tables.get_value(&ia, &ib);
        let id = (value & ID_MASK) as u32;
        // only merge pairs that have the min rank, and only the first of them unless BATCH
        let written = if value == global_min && (BATCH || !merged) {
            read_id += 1;
            merged = true;
            id
        } else {
            ia
        };
        to_merge[write_id] = written;
        if M {
            let merge_rank = self.tables.get_value(&to_merge[write_id - 1], &written);
            running_min = std::cmp::min(running_min, merge_rank);
        }
        write_id += 1;
        read_id += 1;
        (running_min, read_id, write_id, merged)
    }

    /// One sweep of the live buffer, returning this pass's lowest pair and the new length.
    fn one_pass<const BATCH: bool>(
        &self,
        to_merge: &mut [u32],
        len: usize,
        global_min: u64,
    ) -> (u64, usize) {
        // Both cursors restart every pass: a pass is a full sweep of the live buffer.
        let mut read_id = 0usize;
        let mut write_id = 0usize;
        let mut running_min = u64::MAX;
        let mut merged = false;
        (running_min, read_id, write_id, merged) = self.advance_one::<false, BATCH>(
            to_merge,
            read_id,
            global_min,
            write_id,
            running_min,
            merged,
        );
        while read_id + 1 < len {
            (running_min, read_id, write_id, merged) = self.advance_one::<true, BATCH>(
                to_merge,
                read_id,
                global_min,
                write_id,
                running_min,
                merged,
            );
        }
        // `advance_one` consumes a pair per call, so when the sweep ends on the final symbol it
        // has no right neighbour and was never written. Copy it, and rank it against its left
        // neighbour so this pass's minimum accounts for the last pair too.
        if read_id < len {
            to_merge[write_id] = to_merge[read_id];
            let merge_rank = self
                .tables
                .get_value(&to_merge[write_id - 1], &to_merge[write_id]);
            running_min = cmp::min(running_min, merge_rank);
            write_id += 1;
        }
        (running_min, write_id)
    }

    /// Merges the lowest-ranked pair, then repeats with the next lowest, until no pair merges. Read
    /// and write cursors share one buffer: a pass rewrites `to_merge` in place and shortens it, so
    /// `len` shrinks by one per merge applied.
    pub(super) fn multipass_merge(&self, to_merge: &mut Vec<u32>, mut global_min: u64) {
        // `global_min` is the value of the pair to merge, and a missing pair is `u64::MAX`. If the
        // word has no merge at all then every non-merging pair also compares equal to `u64::MAX`,
        // so without this guard `advance_one` would "merge" all of them into id 0.
        if to_merge.len() < 2 || global_min == u64::MAX {
            return;
        }
        let mut len = to_merge.len();
        loop {
            // One test per pass, not per merge. The guard above means `global_min` is a real merge
            // here, so its flag bits are the ones the table stored rather than a sentinel's.
            let (running_min, written) = if !self.tables.any_unsafe || global_min & SAFE != 0 {
                self.one_pass::<true>(to_merge, len, global_min)
            } else {
                self.one_pass::<false>(to_merge, len, global_min)
            };
            len = written;
            if running_min == u64::MAX {
                break; // no pair in the rewritten buffer merges: done
            }
            global_min = running_min;
        }
        to_merge.truncate(len);
    }
}
