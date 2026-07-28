//! Remembers the token ids a word encodes to, so BPE only has to merge it once.
//!
//! Text repeats itself: a few thousand distinct words usually cover nearly every
//! word of a document, and merging is the most expensive thing the BPE model
//! does. So the second time a word shows up, the answer should come from a table
//! instead of from the merge loop. Both sides of that table are on the hot path:
//! a hit has to be far cheaper than merging, and a miss has to cost close to
//! nothing, because every word pays for one the first time it is seen.
//!
//! There are four pieces. The table is `N` fixed-size slots, `N` being the
//! requested capacity rounded up to a power of two; beside it one byte per slot
//! records how much that slot is being used; and two arenas hold what a slot is
//! too small for:
//!
//! ```text
//!   slots     [Slot; N]      32 bytes each, one word per slot
//!   freqs     [u8; N]        how used each slot is — and 0 means "never written"
//!   key_bytes Slab<u8>       the bytes of words longer than 15
//!   ids       Slab<u32>      id runs longer than 3
//! ```
//!
//! One `WordCache` lives in one scratch buffer, so it is owned by a single
//! thread: no sharing, no locking, no atomics.
//!
//! # Looking a word up
//!
//! A word's hash picks its *home* slot. If another word is already sitting
//! there, the search steps forward one slot at a time — this is open addressing
//! with linear probing — for at most [`WINDOW`] slots:
//!
//! ```text
//!   get(b"tion")      home = hash(b"tion") & (N - 1) = 4
//!
//!     slot 3  │ "port"  │
//!     slot 4  │ "ation" │ ← home, but another word got here first: step forward
//!     slot 5  │ "tion"  │ ← the key matches: hit, hand back its ids
//!     slot 6  │  empty  │
//!     slot 7  │ "the"   │
//!
//!   get(b"xyz")       home = 4 → not "ation", not "tion", slot 6 is empty → miss
//! ```
//!
//! A walk may stop at that empty slot instead of going the full [`WINDOW`],
//! because no slot ever goes back to being empty once it has been used: an empty
//! slot means nothing was ever stored beyond it.
//!
//! # What a slot holds
//!
//! A slot is 32 bytes: a 16-byte `key`, three `u32`s of `payload`, and four bytes
//! of bookkeeping, one of them spare. It takes one of three shapes, depending on
//! how long the word is and how many ids it encoded to. The first is the common
//! case for English or code — a short word encoding to one or two tokens — and it
//! touches nothing but the slot itself:
//!
//! ```text
//!  (1) at most 15 bytes, at most 3 ids — the slot holds everything
//!
//!     b"tion" → [5378, 262]
//!    key     │ t  i  o  n  00 … 00 │ 4 │   the word, length in the top byte
//!    payload │ 5378 │ 262 │  ·  │          the ids, right here
//!    len     2                             2 of the 3 payload words are ids
//!    key_len 0                             none of it is in key_bytes
//!
//!  (2) at most 15 bytes, more than 3 ids — the ids move to the arena
//!
//!     b"电脑" (6 bytes) → 8 ids
//!    key     │ e7 94 b5 e8 84 91 … │ 6 │   still the word itself
//!    payload │  ·  │  40  │  8  │          8 ids, in the ids arena at 40
//!    len     SPILLED                       payload is coordinates, not ids
//!    key_len 0
//!
//!  (3) over 15 bytes — the bytes move too, and the key becomes a hash
//!
//!     b"unbelievabilities" (17 bytes) → 6 ids
//!    key     │ hash(word) │ LONG_TAG │   no room for 17 bytes
//!    payload │  12  │  40  │  6  │       bytes at 12, ids at 40
//!    len     SPILLED
//!    key_len 17                          so a hit has to check those bytes
//! ```
//!
//! Shape (3) is the only one where a matching `key` is not proof: two different
//! long words can hash alike, so a hit there also compares the bytes in
//! `key_bytes`. Shapes (1) and (2) carry the word itself in the key, so an equal
//! key *is* an equal word — one register-wide comparison, no memory to chase.
//!
//! # The life of an entry
//!
//! An entry is born on a miss, keeps its slot for as long as it is being used,
//! and is eventually overwritten by another word that hashes into the same
//! window. All of that is decided by `freqs` — here are four of one window's
//! sixteen bytes over time:
//!
//! ```text
//!                                                    A    B    C    D
//!    1. A, B, C inserted, each starting at            1    1    1    0
//!       NEWBORN — on probation, but not free
//!
//!    2. B is looked up three times, C once            1    4    2    0
//!
//!    3. D takes the last free slot, then a hit        1    4    2    2
//!
//!    4. E arrives and the window is full:
//!
//!         A is the coldest — stored, never used       1    4    2    2
//!         every frequency halves, floored at 1        1    2    1    1
//!         A's slot goes to E, back at NEWBORN         1    2    1    1
//!                                                     ▲ E's now; A's arena runs,
//!                                                       if it had any, come back
//! ```
//!
//! Halving is what keeps this from being "whoever got there first wins": B was
//! hot once, but unless it keeps earning hits it will be down with the rest by
//! the time the next word needs a slot. C and D are one unlucky moment from
//! eviction and one hit from safety.
//!
//! The floor at [`NEWBORN`] is what lets the same byte also mean *free*: no live
//! entry can ever fall to 0, so a 0 says nothing was ever stored in that slot, and
//! one scan of sixteen bytes answers both "is there a free slot here?" and "which
//! entry is coldest?".
//!
//! Overwriting the victim where it lies is what keeps the whole structure small.
//! Removing an entry from an open-addressed table normally needs a tombstone or a
//! backward shift, because a probe walk stops at the first empty slot and a hole
//! in the middle of a chain would cut the walk short. Here nothing is ever
//! removed: a slot goes from empty to occupied once and after that only ever
//! changes owner. That is what lets a lookup stop at the first empty slot, and it
//! is why the window has to be bounded — in a saturated table, that bound is all
//! that stops a walk from running over the whole table.
//!
//! ## Why the frequencies are not in the slot
//!
//! There is room for them — the slot has a spare byte where one used to sit. But
//! then a window's sixteen frequencies would be sixteen bytes scattered across
//! eight cache lines of slot table, read one slot at a time. In their own array
//! they are sixteen *consecutive* bytes, and picking a slot compiles to a 16-byte
//! load, a vector minimum, and — when the window is full — a vector shift and a
//! vector maximum to age the whole window at once.
//!
//! That is a trade, not a free win: a hit now also writes a byte to a line it
//! would otherwise never touch, which buys nothing while the table still has room.
//! Measured over 24 model × corpus × capacity combinations, against the same cache
//! with `freq` back in the slot (`examples/cache_freq_bench.rs` — it runs both in
//! one binary, because cross-binary timings here swing ±30% on code layout alone):
//!
//! ```text
//!   evictions per 1000 lookups        time per word
//!            0 -   2                      ×1.02      ← paying, getting nothing
//!            5 - 100                      ×1.01
//!          136 - 160                      ×0.96
//!          230 - 240                      ×0.91
//!          376 - 442                      ×0.92
//! ```
//!
//! So the crossover is around 50 evictions per 1000 lookups. A cache comfortably
//! larger than the corpus in front of it stays below that and pays the 2% — at the
//! default capacity most of our fixtures never evict at all. A cache that is
//! actually working sits above it: a table smaller than the vocabulary it sees, or
//! a long-lived process whose traffic keeps changing shape.
//!
//! # Where an evicted entry's space goes
//!
//! Whatever A had in the arenas is handed back when A is overwritten, and the
//! next word of the same shape takes it. Runs are never moved and never
//! rounded up, because there is a free list for every length a run can have
//! (see [`Slab`], and [`MAX_LENGTH`] for why that is finite):
//!
//! ```text
//!   the ids arena, just after A — which had 6 ids at offset 12 — was evicted
//!
//!     data  ┌─────┬────────────────────────────┬─────┐
//!           │  …  │ A's 6 ids, at offset 12    │  …  │   nothing is moved
//!           └─────┴────────────────────────────┴─────┘
//!
//!     free  len 6 → [ 12 ]   the next word with 6 ids is handed offset 12
//!           len 8 → [ ]      one list per length, 0 … MAX_LENGTH
//! ```
//!
//! So the arenas hold the live set rather than every word ever seen. Without
//! reuse they could only grow, and reclaiming them would mean copying every live
//! entry into a second buffer — twice the memory for the same number of entries.
//!
//! # Why open addressing, and not fixed buckets
//!
//! The previous version of this cache was 4-way set-associative: the table was
//! cut into buckets of four slots and a word could only live in the one bucket
//! its hash picked. That is a single load per lookup and easy to reason about,
//! but a word whose four slots are taken can never be cached at all, however
//! empty the rest of the table is:
//!
//! ```text
//!   4-way buckets — a word belongs to exactly one bucket of four
//!
//!     bucket 0        bucket 1        bucket 2
//!     │●│●│●│●│       │●│·│·│·│       │·│·│·│·│
//!      ▲
//!      └ full, so this word goes uncached — the free slots one
//!        bucket over may as well not exist
//!
//!   open addressing — the word walks on to the next free slot
//!
//!     │●│●│●│●│·│·│·│·│·│·│·│·│·│·│·│·│
//!      ▲       ▲
//!      home    └ stored here instead; only WINDOW slots taken in a
//!                row can turn a word away
//! ```
//!
//! With buckets that small, being locked out starts happening long before the
//! table is genuinely full. On our multilingual fixtures it cost between 0.1% and
//! 4% of all lookups, worst on Korean and Chinese, where a document holds far
//! more distinct words.
//!
//! Probing a window closed that gap: on the same fixtures the misses went to ~0,
//! and in the encoder (`examples/fixture_bench.rs`) the many-distinct-word
//! scripts spent 10-40% less time in the model stage. English and code came out
//! inside the noise —
//! and the noise there is wide: running one binary against *itself* swings the
//! model stage by ±30%, so measure that floor before believing anything smaller.
//!
//! # Why not a `HashMap`
//!
//! The legacy BPE model in `model.rs` caches in a thread-local
//! `AHashMap<String, Word>` (`BPE_LOCAL_CACHE`), so the comparison is concrete.
//! On a warm benchmark, where every word is already in the map, a hash map is
//! not far off this table — it has no bucket conflicts either. What it cannot do
//! is the rest of the job:
//!
//! - **Every word it accepts goes to the allocator.** The map owns its keys, so
//!   an accepted word is copied into a fresh `String`, and each value keeps a
//!   heap buffer of its own alive. That cost lands on misses, which is where a
//!   cache can least afford it: a word misses the first time it is ever seen.
//!   Here an insert writes into space that already exists.
//! - **A hit is two more random loads.** The key bytes and the value both live
//!   in their own heap blocks, elsewhere in memory from the table that pointed
//!   at them. A slot here answers the common word out of the 32 bytes the probe
//!   has already read.
//! - **It cannot make room.** A `HashMap` grows until something stops it, and
//!   the only thing it can do when stopped is refuse: the legacy cache inserts
//!   `while local.len() < capacity`, so the words from the first pages of text
//!   keep their places forever, however useless they turn out to be. Choosing
//!   *which* entry to drop needs evidence about how much each one is used, and
//!   somewhere to keep it — the frequency byte beside every slot.
//! - **Entries cost more than twice the memory.** 24 bytes for the `String` and
//!   24 for the value's vector inside the table, before the two heap blocks they
//!   point at, against 32 bytes here for the common word.
//!
//! A *shared* map costs more again: [`crate::utils::cache::Cache`], which the
//! Unigram model still uses, wraps one in an `RwLock` and then has to
//! `try_read`/`try_write` and silently drop the read or the write whenever
//! another thread holds it.

use ahash::RandomState;

/// Longest word the cache will store, in bytes.
///
/// Long words are rare and rarely repeat, so storing them buys little, and the
/// lookup for one is a hash over every byte to learn nothing. The cap is also
/// what makes the arenas' free lists possible (one list per length) and what
/// bounds a run's size at all: a tokenizer with no pre-tokenizer hands this
/// model whole documents as single "words", and uncapped the cache would
/// cheerfully memoize documents.
const MAX_LENGTH: usize = 1024;
/// Slots searched from a word's home position. Linear probing normally settles
/// in one or two, but the walk has to stop somewhere: nothing is ever removed,
/// so a saturated table would otherwise scan forever looking for an empty slot.
const WINDOW: usize = 16;
/// Ids carried in the slot itself — three `u32`s is what fits beside the key.
/// Enough for 80-96% of lookups on English or code, but only about a quarter of
/// them on Chinese or Korean, where a vocabulary holding none of those scripts
/// spends ten ids or more on one word. `examples/cache_freq_bench.rs --lengths`
/// prints the distribution per model and corpus.
const INLINE_IDS: usize = 3;
/// `len` marker for an entry whose ids, and possibly whose bytes, are in the
/// arenas.
const SPILLED: u8 = u8::MAX;
/// Set in `key` when it holds the hash of a long word instead of the word
/// itself. Packed keys carry their length (1..=15) in the top byte, so a hashed
/// key can never be mistaken for a packed one, and neither can be 0 — the
/// empty-slot marker.
const LONG_TAG: u128 = 1 << 127;
/// A `freqs` entry for a slot nothing has been stored in yet. Live entries are at
/// least [`NEWBORN`], so the one array answers both "is this slot free?" and "how
/// much is this entry being used?" — see [`WordCache::pick_slot`].
const FREE: u8 = 0;
/// Where a new entry's frequency starts: on probation, but not free.
const NEWBORN: u8 = 1;

/// A word of 1..=15 bytes packed into a `u128`: bytes in the low lanes, length
/// in the top byte. Including the length keeps `"a"` and `"a\0"` apart, and the
/// whole key comparison becomes one register-wide equality instead of a
/// `memcmp` against bytes somewhere else in memory.
fn pack_word(word: &[u8]) -> Option<u128> {
    if word.is_empty() || word.len() > 15 {
        return None;
    }
    let mut lanes = [0u8; 16];
    lanes[..word.len()].copy_from_slice(word);
    Some(u128::from_le_bytes(lanes) | ((word.len() as u128) << 120))
}

/// One entry, in the three shapes the module docs draw out. In short:
///
/// - `key` is the word itself when it fits in 15 bytes, otherwise its hash with
///   [`LONG_TAG`] set.
/// - `payload` is the token ids while `len` counts them, and becomes
///   `[key_off, ids_off, ids_len]` once `len` is [`SPILLED`] — which is what the
///   `key_off`/`ids_off`/`ids_len` readers below are for.
/// - `key_len` is 0 unless the word's bytes went to `key_bytes`, so it doubles as
///   "does a matching `key` need confirming?".
///
/// How much an entry is used is *not* here — it lives in `WordCache::freqs`, for
/// the reasons the module docs give. The byte it left behind cannot be spent on
/// anything: `key`'s 16-byte alignment fixes the slot at 32 bytes, and a fourth
/// inline id would need four. That the size is 32 is the point of the layout —
/// a hit reads the key, the ids and the bookkeeping in one go.
#[derive(Clone, Copy, Default)]
#[repr(C)]
struct Slot {
    key: u128,
    payload: [u32; INLINE_IDS],
    len: u8,
    _pad: u8,
    key_len: u16,
}

const _: () = assert!(std::mem::size_of::<Slot>() == 32);

impl Slot {
    fn is_empty(&self) -> bool {
        self.key == 0
    }

    fn spilled(&self) -> bool {
        self.len == SPILLED
    }

    fn key_off(&self) -> u32 {
        self.payload[0]
    }

    fn ids_off(&self) -> u32 {
        self.payload[1]
    }

    fn ids_len(&self) -> usize {
        self.payload[2] as usize
    }
}

/// A grow-only arena that reuses the space of evicted entries instead of
/// compacting.
///
/// Freed runs go on a free list per exact length. That is only practical because
/// `MAX_LENGTH` bounds every run: there is a list for every length a run can
/// have, so a freed run is always reusable by the next word of the same shape
/// and no length is ever rounded up to a bigger class. The price is
/// `MAX_LENGTH + 1` empty `Vec`s per arena.
struct Slab<T> {
    data: Vec<T>,
    free: Box<[Vec<u32>]>,
    /// Ceiling on `data`, not a reservation.
    budget: usize,
}

impl<T: Copy + Default> Slab<T> {
    fn new(budget: usize) -> Self {
        Self {
            data: Vec::new(),
            free: (0..=MAX_LENGTH).map(|_| Vec::new()).collect(),
            budget,
        }
    }

    /// A run of `len` items, or `None` once the budget is spent.
    fn alloc(&mut self, len: usize) -> Option<u32> {
        if let Some(off) = self.free[len].pop() {
            return Some(off);
        }
        if self.data.len() + len > self.budget {
            return None;
        }
        let off = self.data.len() as u32;
        self.data.resize(self.data.len() + len, T::default());
        Some(off)
    }

    fn release(&mut self, off: u32, len: usize) {
        self.free[len].push(off);
    }

    fn get(&self, off: u32, len: usize) -> &[T] {
        &self.data[off as usize..off as usize + len]
    }

    fn fill(&mut self, off: u32, len: usize, values: impl Iterator<Item = T>) {
        for (dst, value) in self.data[off as usize..off as usize + len]
            .iter_mut()
            .zip(values)
        {
            *dst = value;
        }
    }
}

/// Word bytes to token ids. See the module docs for the design.
pub struct WordCache {
    hasher: RandomState,
    slots: Box<[Slot]>,
    key_bytes: Slab<u8>,
    ids: Slab<u32>,
    /// How much each slot is being used, one byte per slot, [`FREE`] while the
    /// slot has never been written. Outside the slots so that a whole window's
    /// worth is 16 consecutive bytes — see [`WordCache::pick_slot`].
    freqs: Box<[u8]>,
    /// `slots.len() - 1`. The table is a power of two, so this folds a hash into
    /// a slot index.
    mask: usize,
}

impl WordCache {
    pub fn new(capacity: usize) -> Self {
        let n_slots = capacity.next_power_of_two().max(WINDOW);
        Self {
            hasher: RandomState::new(),
            slots: vec![Slot::default(); n_slots].into_boxed_slice(),
            // Ceilings, not reservations: the arenas grow only as far as the live
            // set takes them. Deliberately generous, so that the slot table is
            // what runs out first — an arena that turns inserts away while slots
            // sit empty is a capacity limit in disguise, and tighter numbers here
            // measured exactly that (tens of thousands of words refused at 35%
            // occupancy). The worst case they have to cover is a vocabulary with
            // no CJK in it, which spends ~17 ids on a single Chinese word.
            key_bytes: Slab::new(n_slots * 48),
            ids: Slab::new(n_slots * 16),
            freqs: vec![FREE; n_slots].into_boxed_slice(),
            mask: n_slots - 1,
        }
    }

    /// The ids `word` encoded to last time, if the entry is still there.
    ///
    /// Takes `&mut self` because a hit is also a vote for keeping the entry.
    pub fn get(&mut self, word: &[u8]) -> Option<&[u32]> {
        if word.len() > MAX_LENGTH {
            return None;
        }
        let (key, hash) = self.slot_key(word);
        let index = self.find(key, hash, word)?;
        self.freqs[index] = self.freqs[index].saturating_add(1);
        let slot = self.slots[index];
        if slot.spilled() {
            return Some(self.ids.get(slot.ids_off(), slot.ids_len()));
        }
        Some(&self.slots[index].payload[..slot.len as usize])
    }

    /// Remember what `word` encoded to. Silently does nothing for a word over
    /// [`MAX_LENGTH`] or when an arena is full: a cache is free to forget, and
    /// the caller has the ids either way.
    pub fn insert(&mut self, word: &[u8], ids: impl ExactSizeIterator<Item = u32>) {
        if word.len() > MAX_LENGTH {
            return;
        }
        let (key, hash) = self.slot_key(word);
        // Callers insert after a miss, so there is no existing entry to update.
        let Some(entry) = self.build_entry(key, word, ids) else {
            return;
        };
        let index = self.pick_slot(hash);
        self.reclaim(index);
        self.slots[index] = entry;
        // Both writes or neither: a slot is occupied exactly when its frequency is
        // above [`FREE`], and `pick_slot` reads emptiness out of `freqs` alone.
        self.freqs[index] = NEWBORN;
    }

    /// The value a slot stores in its `key`, and the hash that places it. A short
    /// word is its own key; a longer one keys on a tagged hash and has to be
    /// confirmed against `key_bytes`, since two long words can hash alike.
    fn slot_key(&self, word: &[u8]) -> (u128, u64) {
        let hash = self.hasher.hash_one(word);
        match pack_word(word) {
            Some(packed) => (packed, hash),
            None => ((hash as u128) | LONG_TAG, hash),
        }
    }

    /// The slot holding `word`, searched from its home position. Stopping at the
    /// first empty slot is safe because slots never go back to being empty: an
    /// empty one means the word was never stored.
    fn find(&self, key: u128, hash: u64, word: &[u8]) -> Option<usize> {
        for step in 0..WINDOW {
            let index = (hash as usize + step) & self.mask;
            let slot = &self.slots[index];
            if slot.is_empty() {
                return None;
            }
            if slot.key == key && self.holds_word(slot, word) {
                return Some(index);
            }
        }
        None
    }

    /// Whether the entry really is `word`'s. A packed key is the word itself and
    /// needs nothing further; a hashed one is only as good as its bytes.
    fn holds_word(&self, slot: &Slot, word: &[u8]) -> bool {
        slot.key_len == 0 || self.key_bytes.get(slot.key_off(), slot.key_len as usize) == word
    }

    /// The entry to store, with whatever does not fit in a slot copied into the
    /// arenas. `None` when an arena is full, so a rejected insert leaves the
    /// table and the arenas exactly as they were.
    fn build_entry(
        &mut self,
        key: u128,
        word: &[u8],
        ids: impl ExactSizeIterator<Item = u32>,
    ) -> Option<Slot> {
        let long = key & LONG_TAG != 0;
        let ids_len = ids.len();
        if !long && ids_len <= INLINE_IDS {
            let mut payload = [0u32; INLINE_IDS];
            for (dst, id) in payload.iter_mut().zip(ids) {
                *dst = id;
            }
            return Some(Slot {
                key,
                payload,
                len: ids_len as u8,
                _pad: 0,
                key_len: 0,
            });
        }
        // A short word stays in the key, which is a zero-length run here.
        let key_len = if long { word.len() } else { 0 };
        let key_off = self.key_bytes.alloc(key_len)?;
        let Some(ids_off) = self.ids.alloc(ids_len) else {
            self.key_bytes.release(key_off, key_len);
            return None;
        };
        self.key_bytes.fill(key_off, key_len, word.iter().copied());
        self.ids.fill(ids_off, ids_len, ids);
        Some(Slot {
            key,
            payload: [key_off, ids_off, ids_len as u32],
            len: SPILLED,
            _pad: 0,
            key_len: key_len as u16,
        })
    }

    /// The slot the next entry goes in: the first free one in the window, or the
    /// coldest one if they are all taken.
    ///
    /// `#[inline]` because this and [`WordCache::reclaim`] exist to give the steps
    /// of `insert` names, not to be called: left out of line they cost `insert` two
    /// calls and the pointers it had already loaded.
    ///
    /// This reads nothing but `freqs`, which is why the frequencies live outside
    /// the slots. A window's worth of them is 16 consecutive bytes, so the whole
    /// decision is a 16-byte load, a vector minimum, and — when the window is full
    /// — a vector shift to age it. Reading a `freq` out of each slot instead would
    /// mean 16 strided reads over 8 cache lines. Both jobs come out of the one
    /// array because [`FREE`] is a frequency no live entry can have.
    #[inline]
    fn pick_slot(&mut self, hash: u64) -> usize {
        let home = hash as usize & self.mask;
        let Some(&window) = self.freqs[home..].first_chunk::<WINDOW>() else {
            return self.pick_slot_wrapping(home);
        };
        let coldest = window.iter().copied().min().unwrap_or(FREE);
        let offset = window.iter().position(|&freq| freq == coldest).unwrap_or(0);
        if coldest == FREE {
            return home + offset;
        }
        // Age the window on the way past: every entry in it now has to be used
        // again before the next word arrives here, or become the next victim. The
        // floor keeps them all distinguishable from a free slot.
        for freq in &mut self.freqs[home..home + WINDOW] {
            *freq = (*freq >> 1).max(NEWBORN);
        }
        home + offset
    }

    /// [`WordCache::pick_slot`] for the one home in `mask + 1` whose window runs
    /// off the end of the table: same policy, read one slot at a time, because the
    /// frequencies are not contiguous there.
    #[inline]
    fn pick_slot_wrapping(&mut self, home: usize) -> usize {
        // Seeded at `home` rather than 0 so that a window whose entries have all
        // reached the maximum frequency still evicts one of its own.
        let mut coldest = (u8::MAX, home);
        for step in 0..WINDOW {
            let index = (home + step) & self.mask;
            let freq = self.freqs[index];
            if freq == FREE {
                return index;
            }
            if freq < coldest.0 {
                coldest = (freq, index);
            }
        }
        for step in 0..WINDOW {
            let freq = &mut self.freqs[(home + step) & self.mask];
            *freq = (*freq >> 1).max(NEWBORN);
        }
        coldest.1
    }

    /// Hand an overwritten entry's arena runs back, so the next entry can use
    /// them.
    #[inline]
    fn reclaim(&mut self, index: usize) {
        let slot = self.slots[index];
        if !slot.spilled() {
            return;
        }
        self.key_bytes
            .release(slot.key_off(), slot.key_len as usize);
        self.ids.release(slot.ids_off(), slot.ids_len());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let mut cache = WordCache::new(1 << 8);
        assert_eq!(cache.get(b"hello"), None);
        cache.insert(b"hello", [1u32, 2, 3].into_iter());
        cache.insert(b"world", [4u32].into_iter());
        assert_eq!(cache.get(b"hello"), Some(&[1u32, 2, 3][..]));
        assert_eq!(cache.get(b"world"), Some(&[4u32][..]));
        assert_eq!(cache.get(b"hell"), None);
    }

    /// The three properties the whole key encoding rests on: a packed key is
    /// unique per word, is never `0` (the empty-slot marker), and never looks
    /// like a hashed key.
    #[test]
    fn packed_keys_are_unique_and_tagged() {
        assert_ne!(pack_word(b"a"), pack_word(b"a\0"));
        assert_ne!(pack_word(&[0u8; 15]), Some(0));
        assert_eq!(pack_word(&[0u8; 15]).unwrap() & LONG_TAG, 0);
        assert_eq!(pack_word(b""), None);
        assert_eq!(pack_word(&[b'x'; 16]), None);
    }

    #[test]
    fn oversized_words_are_ignored() {
        let mut cache = WordCache::new(1 << 8);
        let big = vec![7u8; MAX_LENGTH + 1];
        cache.insert(&big, [1u32].into_iter());
        assert_eq!(cache.get(&big), None);
    }

    /// Both sides of the 15-byte packing boundary and both sides of the inline/
    /// arena boundary have to survive a round trip, including the long word whose
    /// stored `key` is only a hash and needs the byte comparison to confirm it.
    #[test]
    fn every_slot_shape_round_trips() {
        let mut cache = WordCache::new(1 << 8);
        let long = vec![b'x'; 200];
        let cases: [(&[u8], Vec<u32>); 6] = [
            (b"short", vec![1]),
            (b"short-wide", (0..40).collect()),
            (b"fifteen-bytes.", vec![2]),
            (b"sixteen-bytes.aa", vec![3]),
            (&long, vec![9]),
            (&long[..64], (0..64).collect()),
        ];
        for (word, ids) in &cases {
            cache.insert(word, ids.clone().into_iter());
        }
        for (word, ids) in &cases {
            assert_eq!(cache.get(word), Some(&ids[..]), "{word:?}");
        }
    }

    /// Two long words can hash to the same key, and then only their bytes tell
    /// them apart. Real hash collisions are too rare to write a test around, so
    /// forge one: park another word's entry on this word's home slot and stamp
    /// this word's key on it. The lookup has to walk past it.
    #[test]
    fn a_hashed_key_is_confirmed_against_the_word_bytes() {
        let mut cache = WordCache::new(1 << 8);
        let mine = vec![b'a'; 40];
        let theirs = vec![b'b'; 40];

        cache.insert(&theirs, [7u32].into_iter());
        let (their_key, their_hash) = cache.slot_key(&theirs);
        let their_slot = cache.slots[cache.find(their_key, their_hash, &theirs).unwrap()];
        let (my_key, my_hash) = cache.slot_key(&mine);
        cache.slots[my_hash as usize & cache.mask] = Slot {
            key: my_key,
            ..their_slot
        };

        cache.insert(&mine, [1u32, 2].into_iter());
        assert_eq!(cache.get(&mine), Some(&[1u32, 2][..]));
    }

    /// A word that hashes into a full window takes the coldest slot rather than
    /// being turned away, and the words that were actually used keep their place.
    #[test]
    fn a_full_window_evicts_its_coldest_entry() {
        let mut cache = WordCache::new(WINDOW);
        let words: Vec<Vec<u8>> = (0..WINDOW as u8).map(|i| vec![i; 4]).collect();
        for (i, word) in words.iter().enumerate() {
            cache.insert(word, [i as u32].into_iter());
        }
        // Every word but the first earns a hit, leaving one clear coldest slot.
        for word in &words[1..] {
            assert!(cache.get(word).is_some());
        }
        cache.insert(b"newcomer", [999u32].into_iter());
        assert_eq!(cache.get(b"newcomer"), Some(&[999u32][..]));
        assert_eq!(
            cache.get(&words[0]),
            None,
            "the unused entry should have gone"
        );
        for (i, word) in words.iter().enumerate().skip(1) {
            assert_eq!(
                cache.get(word),
                Some(&[i as u32][..]),
                "used entry {i} was dropped"
            );
        }
    }

    /// Reusing an evicted entry's arena run is where this design can go wrong:
    /// hand one run to two live entries and a hit starts returning another word's
    /// ids. Churn a table far too small for the input and demand the invariant
    /// that matters — an entry may be evicted, but a hit is never wrong.
    #[test]
    fn a_hit_never_returns_another_words_ids() {
        let mut cache = WordCache::new(64);
        let mut expected: Vec<(Vec<u8>, Vec<u32>)> = Vec::new();
        for i in 0..2000usize {
            let word = match i % 4 {
                0 => format!("w{i}"),
                1 => format!("a-long-word-past-fifteen-bytes-{i}"),
                2 => format!("k{i}xxxxxxxxxxxx"),
                _ => format!("{}-{i}", "z".repeat(i % 40)),
            }
            .into_bytes();
            let ids: Vec<u32> = (0..=(i % 9) as u32).map(|k| i as u32 * 16 + k).collect();
            cache.insert(&word, ids.clone().into_iter());
            expected.push((word, ids));
        }
        let mut live = 0;
        for (word, ids) in &expected {
            if let Some(hit) = cache.get(word) {
                assert_eq!(hit, &ids[..], "{word:?}");
                live += 1;
            }
        }
        assert!(live > 0, "everything was evicted — the test proves nothing");
    }

    /// `find` reads emptiness off the slot's key and `pick_slot` reads it off the
    /// frequency, so the two have to agree about every slot, always. They only can
    /// if `insert` writes both and the aging never lets a live entry reach
    /// [`FREE`].
    #[test]
    fn occupancy_agrees_between_slots_and_frequencies() {
        let mut cache = WordCache::new(64);
        for i in 0..2000usize {
            let word = format!("word-{i}-{}", "x".repeat(i % 20));
            cache.insert(word.as_bytes(), [i as u32, 7].into_iter());
            cache.get(word.as_bytes());
        }
        assert!(
            cache.freqs.iter().any(|&freq| freq != FREE),
            "nothing was stored — the test proves nothing"
        );
        for (index, slot) in cache.slots.iter().enumerate() {
            assert_eq!(
                slot.is_empty(),
                cache.freqs[index] == FREE,
                "slot {index} and its frequency disagree"
            );
        }
    }

    /// Freed runs have to come back. Without reuse the arenas grow with every
    /// insert until their budget is spent, and from then on the cache silently
    /// stops accepting words even though the table has room for them.
    #[test]
    fn arenas_hold_the_live_set_not_every_insert() {
        let mut cache = WordCache::new(64);
        // Same length every time, so one free list serves every word and the
        // bounds below are exact: 64 slots, plus the run reserved for the insert
        // in flight before the entry it replaces gives its own run back.
        let word = |i: usize| format!("a-long-word-past-fifteen-bytes-{i:04}");
        for i in 0..5000usize {
            cache.insert(word(i).as_bytes(), [i as u32; 8].into_iter());
        }
        assert_eq!(
            cache.get(word(4999).as_bytes()),
            Some(&[4999u32; 8][..]),
            "inserts stopped landing — the arenas ran out"
        );
        assert!(cache.key_bytes.data.len() <= 65 * 35);
        assert!(cache.ids.data.len() <= 65 * 8);
    }
}
