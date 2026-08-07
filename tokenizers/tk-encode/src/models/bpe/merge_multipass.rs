//! Multipass merging, for short words (pretokens)
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
//! The values are `u64`s packed as documented in `tables`: the rank in the high half, the SAFE
//! bit, and the internal id of the token the pair merges into in the low 30 bits.
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
use crate::models::bpe::tables::{BpeTables, ID_MASK};


/// Iteratively merges a word in place until it has no legal merge left.
///
/// `ranks[i]` is the value of the pair `(symbols[i], symbols[i + 1])`, seeded by
/// `convert_multipass` and carried across passes. That is the whole point: a pass used to re-look-up
/// every pair it walked over, so the passes summed to O(n^2) table lookups -- measured at 41.9 per
/// merged word. Only the pairs touching a merge's product actually change, so a pass now copies the
/// ranks it did not invalidate and pays `get_value` twice per merge instead of once per symbol.
/// Finding the next target is then a scan of `ranks`, with no lookups at all.
/// Wave merging is not an option here, and it is worth saying why: applying every *local* minimum in
/// one pass, rather than the one *global* minimum, is not byte-exact. Measured on 4 MB of english it
/// produced 1,473,346 tokens against the correct 944,838 -- a different tokenisation -- and was
/// slower besides (316 vs 525 MB/s). That closes chunk-wide bit-sliced rank comparison as a
/// direction: comparing all positions at once only helps if all the winners can be applied at once.
///
/// 24 symbols: `GATE_ASCII` is 24 bytes and a word never has more symbols than bytes.
///
/// Widening this to 64 (on a `u64` live mask) so the gate could move was tried, on the reasoning that
/// the gate existed because multipass searched a compacting array. It does not pay: routing long
/// words here instead of the hot/cold queue measured chinese 101 -> 93 MB/s and russian 152 -> 145 at
/// `GATE_MULTI = 64`. Multipass is O(n) search x O(n) merges against the queue's O(n log n), and by
/// n = 12 the queue is already ahead -- so the gate is not a leftover, it is the crossover. Keeping
/// the bound at 24 also keeps the three stack arrays to a 96-byte fill per word rather than 768.
const MAX_MP: usize = 24;


pub fn merge_multipass(
    tables: &BpeTables,
    symbols: &mut Vec<u32>,
    ranks: &mut Vec<u32>,
    prods: &mut Vec<u32>,
    _first_merge: u64,
) {
    let n = symbols.len();
    if n < 2 || n > MAX_MP {
        return merge_multipass_vec(tables, symbols, ranks, prods);
    }
    debug_assert_eq!(ranks.len(), n - 1, "one rank per adjacent pair");

    // Symbols keep their ORIGINAL positions for the whole merge; nothing is ever compacted.
    //
    // A merge used to splice: write the product, then `copy_within` symbols, ranks and products to
    // close the gap -- three memmoves per merge, on every merge. Instead the word carries a `live`
    // bitmap of which slots still hold a symbol, and a merge just clears one bit. Finding the symbol
    // to the left or right of a position is then two bit ops rather than an index that shifted.
    //
    // The bound is what makes this work: it puts the live set in one `u64`, so "previous live" and
    // "next live" are `leading_zeros` / `trailing_zeros`. Dead pair slots hold `u32::MAX`, which is
    // also "does not merge", so the minimum search skips them for free and needs no mask of its own.
    // The search runs to `n`, the word's real length -- padding it out to the bound was measured 3%
    // slower, because the bound is 24 and the size is ~9.
    //
    // And it runs on the caller's buffers. Copying into `[u32; MAX_MP]` stack arrays first, which is
    // what this did, compiled to a 288-byte `stp q` fill of all three arrays plus THREE out-of-line
    // `memcpy` calls through the PLT per word, for at most 96 bytes each -- the bound only ever
    // constrained `n`, never where the symbols had to live.
    // Padding these out to `MAX_MP` so their length is a compile-time constant was tried, to let the
    // bounds checks fold away and to hand the search a fixed-size reduction: 814 instructions instead
    // of 640, and slower everywhere (english 3.40 -> 3.62 ns/B), because LLVM does not vectorise an
    // argmin -- the index half of the reduction defeats it -- so the scan simply walked 23 padded
    // lanes instead of 7.
    let sym = &mut symbols[..];
    let rank = &mut ranks[..];
    let prod = &mut prods[..];

    // Bit i set means slot i still holds a symbol. `n <= 24`, so this never overflows.
    let mut live: u64 = (1u64 << n) - 1;

    #[inline(always)]
    fn next_live(live: u64, from: usize) -> Option<usize> {
        if from + 1 >= 64 {
            return None;
        }
        let above = live >> (from + 1);
        (above != 0).then(|| from + 1 + above.trailing_zeros() as usize)
    }
    #[inline(always)]
    fn prev_live(live: u64, before: usize) -> Option<usize> {
        let below = live & ((1u64 << before) - 1);
        (below != 0).then(|| 63 - below.leading_zeros() as usize)
    }

    // The loop below indexes the caller's slices, so LLVM cannot see the bound and emits a
    // bounds-check panic per access: 26 of them and 81 branches in a 640-instruction function. Every
    // index is provably in range -- the scan gives `at < n - 1`, and `live` only ever has bits below
    // `n` set -- so `get_unchecked` is sound here, and it takes the function to 555 instructions, 67
    // branches and 16 panics. It is still not worth the `unsafe`: measured 1.0031x over tokbench's 29
    // gpt2 cells, inside a +-0.8% noise floor. Whatever makes an iteration cost ~66 cycles -- for a
    // scan over <= 8 `u32`, two bit ops, and two lookups that are demonstrably hot, since doubling
    // `get_value` in place costs +2.1 ns of ~19 -- it is neither the bounds checks nor memory.
    loop {
        // Lowest rank, leftmost on a tie, over the word's real length. Dead slots are `u32::MAX`.
        //
        // This scan is already branchless: LLVM emits `cmp` + two `csel` (best, and the index), so
        // the only branch is the perfectly-predicted loop-back. Packing the index into the low bits
        // to turn the argmin into a plain min-reduction -- on the theory that the branch was the
        // cost and that the index half was what blocked vectorisation -- measured **0.9836x** over
        // tokbench's 29 gpt2 cells (english 3.26 -> 3.44), byte-exact. It backfired precisely
        // because it succeeded: LLVM then vectorises the reduction, and the vector prologue (lane
        // index vectors, four accumulators, a scalar epilogue) costs far more than the ~9 elements
        // it processes. Do not retry either half of that idea without also pinning the trip count.
        let mut best = u32::MAX;
        let mut at = 0usize;
        for (i, &r) in rank[..n - 1].iter().enumerate() {
            if r < best {
                best = r;
                at = i;
            }
        }
        if best == u32::MAX {
            break;
        }

        // The pair is (at, next_live(at)). The product lands in `at`; the right symbol dies.
        let Some(right) = next_live(live, at) else { break };
        sym[at] = prod[at];
        live &= !(1u64 << right);
        // No pair starts at the last symbol, so there is no rank slot for it; in the padded array
        // this wrote `u32::MAX` over `u32::MAX`.
        if right + 1 < n {
            rank[right] = u32::MAX; // its pair is gone with it
        }
        rank[at] = u32::MAX; // recomputed below if a right neighbour remains

        // Only the pairs either side of the new symbol changed.
        if let Some(pv) = prev_live(live, at) {
            let v = tables.get_value(&sym[pv], &sym[at]);
            rank[pv] = (v >> 32) as u32;
            prod[pv] = (v & ID_MASK) as u32;
        }
        if let Some(nx) = next_live(live, at) {
            let v = tables.get_value(&sym[at], &sym[nx]);
            rank[at] = (v >> 32) as u32;
            prod[at] = (v & ID_MASK) as u32;
        }
    }

    // Compact the survivors to the front, in position order. The write index never passes the read
    // index, so this is safe in place and needs no second buffer.
    let mut kept = 0usize;
    let mut m = live;
    while m != 0 {
        let i = m.trailing_zeros() as usize;
        sym[kept] = sym[i];
        kept += 1;
        m &= m - 1;
    }
    symbols.truncate(kept);
    ranks.clear();
    prods.clear();
}

/// The compacting version, for the rare word longer than [`MAX_MP`] that still routes here.
fn merge_multipass_vec(
    tables: &BpeTables,
    symbols: &mut Vec<u32>,
    ranks: &mut Vec<u32>,
    prods: &mut Vec<u32>,
) {
    let mut len = symbols.len();
    while len >= 2 {
        let mut best = u32::MAX;
        let mut at = 0usize;
        for (i, &rank) in ranks[..len - 1].iter().enumerate() {
            if rank < best {
                best = rank;
                at = i;
            }
        }
        if best == u32::MAX {
            break;
        }
        symbols[at] = prods[at];
        symbols.copy_within(at + 2..len, at + 1);
        len -= 1;
        if len >= 2 {
            ranks.copy_within(at + 1..len, at);
            prods.copy_within(at + 1..len, at);
        }
        if at > 0 {
            let v = tables.get_value(&symbols[at - 1], &symbols[at]);
            ranks[at - 1] = (v >> 32) as u32;
            prods[at - 1] = (v & ID_MASK) as u32;
        }
        if at + 1 < len {
            let v = tables.get_value(&symbols[at], &symbols[at + 1]);
            ranks[at] = (v >> 32) as u32;
            prods[at] = (v & ID_MASK) as u32;
        }
    }
    symbols.truncate(len);
    ranks.truncate(len.saturating_sub(1));
    prods.truncate(len.saturating_sub(1));
}
