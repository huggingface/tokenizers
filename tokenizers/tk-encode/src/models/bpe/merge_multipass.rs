//! Multipass merging, for short words (pre tokens)
//!
//! For short words, running BPE naively can be faster than using a more complex data structure.
//!
//! We iteratively sweep the pre token's pairs of symbols to find the pair with the lowest merge rank,
//! merge it in-place, and repeat until there is no legal merge left.
//!
//! # Example
//!
//! Merging the word: "hello".
//! The internal ids are h=0, e=1, l=2, o=3, ll=4, he=5, llo=6, hello=7
//! The model has 4 merges, we store them in a lookup table as follows:
//!
//! | key (pair) | value                 | rank              | SAFE | merged symbol id |
//! |------------|-----------------------|-------------------|------|------------------|
//! | (l,l)      | `0x00000000_40000004` | 0                 | yes  | 4 (ll)           |
//! | (h,e)      | `0x00000001_40000005` | 1                 | yes  | 5 (he)           |
//! | (ll,o)     | `0x00000002_40000006` | 2                 | yes  | 6 (llo)          |
//! | (he,llo)   | `0x00000003_40000007` | 3                 | yes  | 7 (hello)        |
//! | any other  | `0xFFFFFFFF_FFFFFFFF` | not a legal merge |      |                  |
//!
//! The values are `u64` packed as follows:
//!
//!  ```text
//! bit 63                              32    31     30   29                           0
//!    ┌──────────────────────────────────┬──────┬────┬──────────────────────────────┐
//!    │            rank : u32            │unused│SAFE│      product id : 30 bits    │
//!    │      (merge priority, 0 = best)  │      │    │   (internal id of the token  │
//!    │                                  │      │    │      this pair merges into)  │
//!    └──────────────────────────────────┴──────┴────┴──────────────────────────────┘
//! ```
//!
//! Then we repeatedly merge symbols with "passes", until there is no legal merge left.
//!
//! A pass builds the new word in the same array that holds the old one, using two cursors that both start at index 0.
//! The read cursor marks the start of what is left of the old word.
//! The write cursor marks the end of the new word built so far.
//! Every step writes exactly one symbol: a copy (a no merge for a pair just results in copying the id that was just read) moves both cursors by one, and a merge writes one symbol but consumes two, so the read cursor moves by 2.
//! The pass needs no second array and no allocation, and each merge makes the new word one symbol shorter.
//!
//! ## Illustrated
//!
//! ```text
//!   ┌───┬───┬───┬───┬───┐      (h,e) = 0x00000001_40000005
//!   │ h │ e │ l │ l │ o │      (e,l) = u64::MAX
//!   └───┴───┴───┴───┴───┘      (l,l) = 0x00000000_40000004  <- lowest: pass 1's target
//!                              (l,o) = u64::MAX
//! ```
//!
//! Pass 1 then sweeps the array:
//!
//! ```text
//!   ┌───┬───┬───┬───┬───┐
//!   │ h │ e │ l │ l │ o │
//!   └───┴───┴───┴───┴───┘
//!     ^w                    value(h,e) != target: copy h, both cursors move by one.
//!     ^r                    First write: nothing to its left to rank yet
//!
//!   ┌───┬───┬───┬───┬───┐
//!   │ h │ e │ l │ l │ o │
//!   └───┴───┴───┴───┴───┘
//!         ^w                value(e,l) != target: copy e,
//!         ^r                and rank the newly written pair (h,e): rank 1
//!
//!   ┌───┬───┬───┬───┬───┐
//!   │ h │ e │ l │ l │ o │
//!   └───┴───┴───┴───┴───┘
//!             ^w            value(l,l) == target: write its product id, ll, and skip
//!             ^r            both l; rank the newly written pair (e,ll): not a merge
//!
//!   ┌───┬───┬────┬───┬───┐
//!   │ h │ e │ ll │ l │ o │
//!   └───┴───┴────┴───┴───┘
//!                  ^w       read is now ahead of write; the leftover l was already
//!                      ^r   read, so the next write may overwrite it
//!
//!   ┌───┬───┬────┬───┬───┐
//!   │ h │ e │ ll │ o │ ▒ │
//!   └───┴───┴────┴───┴───┘
//!                     ^^^   the last symbol has no pair left: copy it as is,
//!                           and rank (ll,o): rank 2. 5 symbols in, 4 out, so the
//!                           5th slot is now dead: it sits past the new length and
//!                           still holds the stale o the copy read from.
//!                           The lowest pair ranked during the sweep was (h,e),
//!                           so (h,e) is pass 2's target
//! ```
//!
//! Each later pass repeats this, merging the target the previous pass found:
//!
//! ```text
//!   pass 2  target (h,e):    [ h  │ e   │ ll │ o │ ▒ ]  ->  [ he    │ ll  │ o │ ▒ │ ▒ ]  lowest written pair: (ll,o)
//!   pass 3  target (ll,o):   [ he │ ll  │ o  │ ▒ │ ▒ ]  ->  [ he    │ llo │ ▒ │ ▒ │ ▒ ]  lowest written pair: (he,llo)
//!   pass 4  target (he,llo): [ he │ llo │ ▒  │ ▒ │ ▒ ]  ->  [ hello │ ▒   │ ▒ │ ▒ │ ▒ ]  no pair left to rank: done
//!
//!   ▒ = dead slot: past the pass's new length, so the next pass never reads it.
//!       The array itself never shrinks, only the length does, and the trailing
//!       dead slots are dropped once by the final truncate.
//! ```
//!
//! Recording the lowest-ranked pair happens while writing the symbols: after each write, we take
//! the value of the last two written symbols and keep the minimum. When both were copies, that
//! pair also existed in the old word, and the sweep just looked up its value to decide against
//! merging it, so the value is reused rather than looked up again. Only pairs involving a merge's
//! product are new, and looking those up at write time ranks them in the same pass that creates
//! them. When a pass ends with a minimum of `u64::MAX`, no pair in the word merges anymore, and
//! the word is done.
//!
//! # Batching and the `SAFE` bit
//!
//! The target merge can occur several times in the word (for example "hello lots", the pair "lo" appears twice). When its merge is `SAFE`, one pass merges every occurrence.
//! The merge is safe to batch merge only if the produced id (merged symbol) does not take part in other merges with lower rank (higher priority).
//! This is enforced when building the lookup table and encoded in the `SAFE` bit.
//!
use crate::models::bpe::bpe_build_tables::{BpeTables, ID_MASK, SAFE_MASK};
use std::cmp;

const NOT_LEGAL: u64 = u64::MAX;

/// Iteratively merges a word in place until it has no legal merge left
pub(super) fn merge_multipass(tables: &BpeTables, symbols: &mut Vec<u32>, mut target_merge: u64) {
    let mut len = symbols.len();
    if len < 2 || target_merge == NOT_LEGAL {
        return;
    }
    loop {
        let MergeOnceOutput {
            next_merge,
            merged_length,
        } = merge_once(
            tables,
            symbols,
            len,
            target_merge,
            batch_merging_is_safe(tables, target_merge),
        );
        len = merged_length;
        if next_merge == NOT_LEGAL {
            break;
        }
        target_merge = next_merge;
    }
    symbols.truncate(len);
}

/// Whether one pass may merge every occurrence of the target, or only the first occurrence.
#[inline(always)]
fn batch_merging_is_safe(tables: &BpeTables, target_merge: u64) -> bool {
    // A NOT_LEGAL merge has the SAFE bit set, it would incorrectly return true here
    // The caller is responsible for checking NOT_LEGAL does not reach this
    debug_assert!(target_merge != NOT_LEGAL);
    !tables.any_unsafe || (target_merge & SAFE_MASK != 0)
}

/// One pass's cursors and running result.
///
/// The read cursor marks the start of what is left of the old word, the write cursor
/// the end of the new word built so far (see the module docs).
struct MergeState {
    read_cursor: usize,
    write_cursor: usize,
    target_merge: u64,
    batched: bool,
    was_merged: bool,
    next_merge: u64,
}

impl MergeState {
    fn new(target_merge: u64, batched: bool) -> Self {
        Self {
            read_cursor: 0,
            write_cursor: 0,
            target_merge,
            batched,
            was_merged: false,
            next_merge: NOT_LEGAL,
        }
    }

    /// Looks up the pair at the read cursor and writes one symbol: the pair's product id when
    /// its value equals the target and the pair may still merge, the left symbol otherwise. A
    /// merge consumes both symbols of the pair, a copy only the left one.
    ///
    /// The returned value is the next call's `cached_pair_value`, and it is what keeps a step
    /// down to a single table lookup. A copy has already paid for the lookup of
    /// (`left_symbol`, `right_symbol`), and that is exactly the pair the next write has to
    /// rank, so handing the value forward wins that second lookup back. A merge returns `None`
    /// instead: it writes a product id that no pair has been looked up against yet, so the
    /// next write has to pay for its own lookup.
    #[inline(always)]
    fn step(
        &mut self,
        tables: &BpeTables,
        symbols: &mut [u32],
        cached_pair_value: Option<u64>,
    ) -> Option<u64> {
        let (left_symbol, right_symbol) =
            (symbols[self.read_cursor], symbols[self.read_cursor + 1]);
        let pair_value = tables.get_value(&left_symbol, &right_symbol);
        let should_merge = pair_value == self.target_merge && (self.batched || !self.was_merged);
        if should_merge {
            self.was_merged = true;
            self.read_cursor += 2;
            let merged_symbol = (pair_value & ID_MASK) as u32;
            self.write(tables, symbols, merged_symbol, None);
            None
        } else {
            self.read_cursor += 1;
            self.write(tables, symbols, left_symbol, cached_pair_value);
            Some(pair_value)
        }
    }

    /// Writes one symbol at the write cursor and ranks the pair it forms with the previously
    /// written symbol as a candidate for the next pass's target: `next_merge` keeps the lowest
    /// value seen. `Some` reuses the value the caller already looked up, `None` pays for a
    /// lookup here. The first written symbol has no left neighbour and nothing to rank.
    #[inline(always)]
    fn write(
        &mut self,
        tables: &BpeTables,
        symbols: &mut [u32],
        symbol: u32,
        cached_pair_value: Option<u64>,
    ) {
        symbols[self.write_cursor] = symbol;
        if self.write_cursor > 0 {
            let rank = cached_pair_value
                .unwrap_or_else(|| tables.get_value(&symbols[self.write_cursor - 1], &symbol));
            self.next_merge = cmp::min(self.next_merge, rank);
        }
        self.write_cursor += 1;
    }
}

struct MergeOnceOutput {
    next_merge: u64,
    merged_length: usize,
}

/// One pass: walk the first `len` elements of `symbols` and merge occurrences of `target_merge`.
///
/// Returns the next pass's target (`u64::MAX` when no pair in the rewritten word merges), and the number of symbols written.
/// Symbols past `len` are leftovers of earlier passes and should be truncated.
fn merge_once(
    tables: &BpeTables,
    symbols: &mut [u32],
    len: usize,
    target_merge: u64,
    batched: bool,
) -> MergeOnceOutput {
    // Resliced so the loop bound and the slice length are the same value. Without this the
    // compiler cannot connect `len` to the length of `symbols` and keeps a bounds check on
    // every read and write of the sweep.
    let symbols = &mut symbols[..len];
    let mut state = MergeState::new(target_merge, batched);
    let mut cached_pair_value = None;
    while state.read_cursor + 1 < len {
        cached_pair_value = state.step(tables, symbols, cached_pair_value);
    }
    if state.read_cursor < len {
        // The sweep's final symbol has no right neighbour to pair with, so it is copied as is.
        let last_symbol = symbols[state.read_cursor];
        state.write(tables, symbols, last_symbol, cached_pair_value);
    }
    MergeOnceOutput {
        next_merge: state.next_merge,
        merged_length: state.write_cursor,
    }
}
