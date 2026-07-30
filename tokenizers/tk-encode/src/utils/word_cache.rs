//! A table that remembers which token ids a word encodes to, so a model only has
//! to work it out once.
//!
//! # Context
//!
//! This section gives more context about tokenization, why the word_cache is useful, and
//! where does it fit in the tokenization pipeline.
//!
//! A tokenizer turns text into *token ids*: small integers a language model reads.
//! It gets there in three stages:
//!   - It **normalizes** the text (for example: lowercasing),
//!   - It **pre-tokenizes** the normalized text (ie, cuts it into small pieces, usually words or fragments of words),
//!   - It runs a tokenization **model** over each piece to produce token ids
//!
//! This module calls those pieces *words*, because that is what they nearly always are.
//!
//! ```text
//!               "The cat sat on The Mat"
//!                           │
//!                           │  normalize, then pre-tokenize
//!                           ▼
//!        "the"  "cat"  "sat"  "on"   "the"  "mat"      ← words
//!          │      │      │      │      │      │
//!          ▼      ▼      ▼      ▼      ▼      ▼
//!    ┌─────────────────────────────────────────────┐
//!    │              Tokenization Model             │
//!    └─────────────────────────────────────────────┘
//!          │      │      │      │      │      │
//!          ▼      ▼      ▼      ▼      ▼      ▼
//!        [12]   [87]   [43]    [9]   [12]   [64]       ← token ids, made up here
//! ```
//!
//! The model stage is expensive, due to the underlying algorithms:
//!
//! - [**BPE**](crate::models::bpe::BPE) (byte-pair encoding) starts from the word's individual
//!   bytes and repeatedly glues the best-ranked neighboring pair together, until no pair
//!   left in the word can be merged. Every merge changes the word and the search
//!   for the next best pair starts again ([`Word::merge_all`](crate::models::bpe::Word::merge_all)).
//! - [**Unigram**](crate::models::unigram::Unigram) considers the many ways of cutting the word
//!   into pieces the vocabulary knows, scores them, and keeps the best. That is a search over a
//!   lattice of candidate cuts ([`Lattice::viterbi`](crate::models::unigram::Lattice::viterbi)).
//! - [**WordPiece**](crate::models::wordpiece::WordPiece) walks the word from the front, taking
//!   the longest piece the vocabulary holds, then carries on from where that piece ended. That is
//!   one search per piece of the word.
//!
//! It is possible to work around the expensive tokenization algorithm because:
//! - Words in text often repeats
//! - The tokenization of a word is always the same (the model gives the same ids)
//!
//! If we remember ("cache" or "memoize") the output of the tokenization model for a word,
//! we can bypass the model algorithm altogether the next time the word shows up.
//!
//! This module implements a data structure that makes caching / memoization of words efficient
//! and memory-bound:
//!
//! ```text
//!               "The cat sat on The Mat"
//!                           │
//!                           │  normalize, then pre-tokenize
//!                           ▼
//!        "the"  "cat"  "sat"  "on"   "the"  "mat"      ← words, one at a time
//!          │      │      │      │      │      │
//!          ▼      ▼      ▼      ▼      ▼      ▼
//!    ┌─────────────────────────────────────────────┐
//!    │  WordCache — have I encoded this word yet?  │
//!    └─────────────────────────────────────────────┘
//!                  │                │
//!             miss │                │ hit
//!                  ▼                │
//!       ┌─────────────────────┐     │
//!       │  Model (expensive)  │     │
//!       └─────────────────────┘     │
//!                  │                │
//!                  │ store the ids  │ the ids, straight from the table
//!                  └─────┬──────────┘
//!          ┌──────┬──────┼──────┬──────┬──────┐
//!          ▼      ▼      ▼      ▼      ▼      ▼
//!        [12]   [87]   [43]    [9]   [12]   [64]       ← the same ids, most of them for free
//!                                      ▲
//!                                      └ the second "the": a cache hit
//! ```
//!
//! This simple memoization trick gives huge performance gains: up to 20x more throughput.
//!
//! # What the cache is allowed to get wrong
//!
//! The cache is free to *forget*: if a word's ids are gone, the model works them
//! out again. What it must never do is hand back the *wrong* ids. Every
//! trade-off below spends the first freedom and none of the second.
//!
//! # Where the ideas come from
//!
//! - [Swiss Tables] — Abseil's design notes: one control byte per slot, holding a
//!   flag bit and seven bits of hash, sixteen tested at a time.
//! - [hashbrown] — the hash map behind Rust's `std::collections::HashMap`. Its
//!   portable fallback tests a group of control bytes with the same
//!   `(x - ONES) & !x & HIGHS` arithmetic used in [`tag_matches`].
//! - [Bit Twiddling Hacks] — that expression on its own, credited to Alan Mycroft,
//!   1987.
//! - [gigatoken] — a BPE tokenizer with a pre-token cache built from the same
//!   parts: `u128` packed keys with the length in the top byte, self-contained
//!   32-byte entries, ids inline. It never evicts (it doubles at 3/4 load) and
//!   leans on huge pages and prefetching, because its table is sized for DRAM
//!   rather than for a CPU cache.
//! - [Redis key eviction] — an eight-bit counter per key plus a decay period,
//!   which is the same shape as [`EvictionMetadata`].
//! - [TinyLFU] (Einziger, Friedman & Manes) and its reference implementation,
//!   [Caffeine], which ages counters by halving all of them at once
//!   (`table[i] = (table[i] >>> 1) & RESET_MASK`) rather than by dating them.
//! - [huggingface/tokenizers#2234] — an open-addressed cache for this same encode
//!   pipeline, arrived at in parallel, fused into the pre-tokenizer's split loop.
//!
//! [Swiss Tables]: https://abseil.io/about/design/swisstables
//! [hashbrown]: https://github.com/rust-lang/hashbrown/blob/master/src/control/group/generic.rs
//! [Bit Twiddling Hacks]: https://graphics.stanford.edu/~seander/bithacks.html#ZeroInWord
//! [gigatoken]: https://github.com/marcelroed/gigatoken
//! [Redis key eviction]: https://redis.io/docs/latest/develop/reference/eviction/
//! [TinyLFU]: https://arxiv.org/abs/1512.00727
//! [Caffeine]: https://github.com/ben-manes/caffeine/blob/master/caffeine/src/main/java/com/github/benmanes/caffeine/cache/FrequencySketch.java
//! [huggingface/tokenizers#2234]: https://github.com/huggingface/tokenizers/pull/2234

use ahash::RandomState;

// ---------------------------------------------------------------- the cache

/// Longest word the cache will store, in bytes.
const MAX_WORD_BYTES: usize = 1024;

/// Word bytes to token ids. See the module docs for the design.
pub struct WordCache {
    /// A table of up to N [`CachedWord`]. The main data.
    ///
    /// For a given word, we derive its index in [`Self::cached_word`] by keeping the lower N bytes of its hash:
    /// ```text
    /// hash(word) & (N - 1)
    /// ```
    ///
    /// Where N is the cache capacity (= number of slots). N has to be a power of 2.
    cached_words: Box<[CachedWord]>,

    /// Produces hashes from a word's bytes
    hasher: RandomState,

    /// Masks the lower N bytes of a value
    index_mask: usize,

    /// Sibling table of [`Self::cached_words`].
    ///
    /// One byte per slot, at the same index as the corresponding [`CachedWord`] in [`Self::cached_words`].
    /// For a given index, hold the 7 high bits of the hash of the corresponding `CachedWord`, or [`EMPTY`] if the slot is empty.
    /// Being a contiguous buffer of bytes, it allows to do a very efficient lookup of candidates in [`Self::cached_words`]
    /// using SIMD with a register techniques, probing [`PROBE_WINDOW`] slots at the same time.
    probe_lookup_table: Box<[u8]>,

    /// This computes which slot gets evicted when the lookup could not find an empty slot to insert a new word in.
    evictor: Evictor,

    /// If the bytes of the word don't fit in a CachedWord, they are stored here
    word_bytes_arena: Arena<u8>,

    /// If the IDs of the word don't fit in a CachedWord, they are stored here
    token_ids_arena: Arena<u32>,
}

impl WordCache {
    pub fn new(capacity: usize) -> Self {
        // todo: warn when the capacity is rounded up
        let n_slots = capacity.next_power_of_two().max(PROBE_WINDOW);
        Self {
            hasher: RandomState::new(),
            cached_words: vec![CachedWord::default(); n_slots].into_boxed_slice(),
            probe_lookup_table: vec![EMPTY; n_slots + PROBE_WINDOW].into_boxed_slice(),
            evictor: Evictor::new(n_slots),
            word_bytes_arena: Arena::new(n_slots * 48),
            token_ids_arena: Arena::new(n_slots * 16),
            index_mask: n_slots - 1,
        }
    }

    /// Lookup a word in the cache.
    ///
    /// If the word is found in the cache, returns the ids the word encodes to.
    /// If the word is not found in the cache, returns the index at which the [`CachedWord`] should be inserted.
    ///
    /// Takes `&mut self` because a cache hit needs to update the freshness of the slot for eviction bookkeeping.
    pub fn lookup<'c, 'w>(&'c mut self, word: &'w [u8]) -> Lookup<'c, 'w> {
        // Never cache a word longer than `MAX_WORD_BYTES`
        if word.len() > MAX_WORD_BYTES {
            return Lookup::Miss(None);
        }

        let (key, hash) = self.make_word_key(word);
        match self.find_word_in_cache(key, hash, word) {
            // Found the word in the cache => return its ids
            Probe::Found(index) => {
                self.evictor.record_hit(index);
                let slot = self.cached_words[index];
                Lookup::Hit(if slot.ids_stored_in_arena() {
                    self.token_ids_arena.get(slot.ids_off(), slot.ids_len())
                } else {
                    &self.cached_words[index].word_ids[..slot.inline_id_count as usize]
                })
            }
            // The word is not in the cache => return a miss with an indication of where the
            // encoded ids should be stored.
            Probe::Absent(placement) => Lookup::Miss(Some(placement)),
        }
    }

    /// Creates and inserts a [`CachedWord`]
    pub fn insert(&mut self, at: Placement<'_>, ids: impl ExactSizeIterator<Item = u32>) {
        let Some(cached_word) = self.build_entry(at.key, at.word, ids) else {
            // No-op if the word would spill and one of the arena is over budget
            return;
        };
        // Slot is not empty => evict it
        if self.probe_lookup_table[at.index] != EMPTY {
            self.reclaim(at.index);
            self.evictor.on_eviction();
        }
        // Update the cached slot
        self.update_lookup_table(at.index, at.lookup_tag);
        self.evictor.record_insert(at.index);
        self.cached_words[at.index] = cached_word;
    }

    /// The value a slot stores in its `key`, and the hash that places it. A short
    /// word is its own key; a longer one keys on a tagged hash and has to be
    /// confirmed against `key_arena`, since two long words can hash alike.
    ///
    /// A packed key is hashed as one `u128`, not as the word's bytes it was built
    /// from. Hashing a slice makes aHash mix the length in and then branch on it
    /// to choose a read width; a `u128` is one fixed-width fold with nothing to
    /// decide. The key already carries the length, so nothing is lost.
    fn make_word_key(&self, word: &[u8]) -> (u128, u64) {
        match pack_word(word) {
            Some(packed) => (packed, self.hasher.hash_one(packed)),
            None => {
                let hash = self.hasher.hash_one(word);
                ((hash as u128) | KEY_IS_HASH, hash)
            }
        }
    }

    /// The tags of the window starting at `home`. The mirrored tail of
    /// `probe_lookup_table` is what makes the slice contiguous for every `home`.
    fn make_probe_window(&self, start_index: usize) -> u128 {
        read_window(
            self.probe_lookup_table[start_index..start_index + PROBE_WINDOW]
                .try_into()
                .unwrap(),
        )
    }

    /// One probe over `word`'s window, from its home position, answering both
    /// questions a lookup has: which slot holds the word, and — if none does —
    /// which slot it should take.
    ///
    /// The second answer is nearly free: both come out of the window's tags,
    /// which are one load, and only a full window needs anything more. Working it
    /// out afterwards would mean reading them again and hashing the word again.
    ///
    /// Stopping at the first empty slot is safe because slots never go back to
    /// being empty: an empty one means the word was never stored.
    fn find_word_in_cache<'w>(&self, key: u128, hash: u64, word: &'w [u8]) -> Probe<'w> {
        let start_index = hash as usize & self.index_mask;
        let tag = make_lookup_tag(hash);
        let window = self.make_probe_window(start_index);
        let empty = empty_slots(window);
        // Nothing past the first empty slot can hold the word: the probe that
        // stored it would have stopped there and taken that slot.
        let mut candidates = tag_matches(window, tag) & steps_before(empty);
        // Only a hashed key can turn out to belong to a different word, and
        // whether this one is hashed is settled before the probe starts — the
        // caller built the key out of the word it is looking for.
        let hashed = key & KEY_IS_HASH != 0;
        while candidates != 0 {
            let step = first_step(candidates);
            let index = (start_index + step) & self.index_mask;
            let slot = &self.cached_words[index];
            if slot.word_bytes_or_hash == key
                && (!hashed || self.word_bytes_arena.get(slot.key_off(), slot.key_len()) == word)
            {
                return Probe::Found(index);
            }
            candidates = clear_step(candidates, step);
        }
        let step = if empty == 0 {
            self.evictor.coldest(start_index)
        } else {
            first_step(empty)
        };
        Probe::Absent(Placement {
            index: (start_index + step) & self.index_mask,
            key,
            lookup_tag: tag,
            word,
        })
    }

    /// The entry to store, with whatever does not fit in a slot copied into the
    /// arenas. `None` when an arena is full, so a rejected insert leaves the table
    /// and the arenas exactly as they were.
    fn build_entry(
        &mut self,
        key: u128,
        word: &[u8],
        ids: impl ExactSizeIterator<Item = u32>,
    ) -> Option<CachedWord> {
        let long = key & KEY_IS_HASH != 0;
        let ids_len = ids.len();
        if !long && ids_len <= MAX_INLINE_IDS {
            let mut payload = [0u32; MAX_INLINE_IDS];
            for (dst, id) in payload.iter_mut().zip(ids) {
                *dst = id;
            }
            return Some(CachedWord {
                word_bytes_or_hash: key,
                word_ids: payload,
                inline_id_count: ids_len as u8,
            });
        }
        // A short word stays in the key, which is a zero-length run here.
        let key_len = if long { word.len() } else { 0 };
        let key_off = self.word_bytes_arena.alloc(key_len)?;
        let Some(ids_off) = self.token_ids_arena.alloc(ids_len) else {
            self.word_bytes_arena.release(key_off, key_len);
            return None;
        };
        self.word_bytes_arena
            .fill(key_off, key_len, word.iter().copied());
        self.token_ids_arena.fill(ids_off, ids_len, ids);
        let lengths = (key_len as u32) << PACKED_LEN_BITS | ids_len as u32;
        Some(CachedWord {
            word_bytes_or_hash: key,
            word_ids: [key_off, ids_off, lengths],
            inline_id_count: SPILLED,
        })
    }

    /// Write a slot's tag, and the copy of it in the mirrored tail when the slot
    /// is one of the first [`PROBE_WINDOW`].
    fn update_lookup_table(&mut self, index: usize, hash_high_bits: u8) {
        self.probe_lookup_table[index] = hash_high_bits;
        if index < PROBE_WINDOW {
            self.probe_lookup_table[self.cached_words.len() + index] = hash_high_bits;
        }
    }

    /// Hand an overwritten entry's arena runs back, so the next entry can use
    /// them.
    ///
    /// `#[inline]` because this exists to give a step of `insert` a name, not to
    /// be called: left out of line it costs `insert` a call and the pointers it
    /// had already loaded.
    #[inline]
    fn reclaim(&mut self, index: usize) {
        let slot = self.cached_words[index];
        if !slot.ids_stored_in_arena() {
            return;
        }
        self.word_bytes_arena
            .release(slot.key_off(), slot.key_len());
        self.token_ids_arena.release(slot.ids_off(), slot.ids_len());
    }
}

// ---------------------------------------------------------------- what a lookup returns

/// What [`WordCache::lookup`] found.
pub enum Lookup<'c, 'w> {
    /// The ids the word encoded to last time.
    Hit(&'c [u32]),
    /// The word is not in the table. The [`Placement`] is the slot it should go
    /// in — hand it to [`WordCache::insert`] once the model has done the work, or
    /// drop it and nothing is stored. `None` when the word is too long to be
    /// worth a slot at all.
    Miss(Option<Placement<'w>>),
}

impl<'c> Lookup<'c, '_> {
    /// The ids, throwing the [`Placement`] away. Every caller in the encoder
    /// wants the placement — this is for tests asserting on what the table holds.
    #[cfg(test)]
    pub fn hit(self) -> Option<&'c [u32]> {
        match self {
            Lookup::Hit(ids) => Some(ids),
            Lookup::Miss(_) => None,
        }
    }
}

/// Where a word that missed will go, and what [`WordCache::insert`] needs to put
/// it there. Built by [`WordCache::find_slot`] out of what the probe had already
/// worked out, so storing the word costs no second hash, no second read of the
/// window and no second tag.
///
/// It carries the word rather than letting `insert` take it again, because the
/// slot, the key and the tag inside were chosen for *this* word: handing back a
/// different one would file its ids under the first word's name.
pub struct Placement<'w> {
    index: usize,
    key: u128,
    lookup_tag: u8,
    word: &'w [u8],
}

/// How a probe over a word's window ended.
enum Probe<'w> {
    /// The word is in this slot.
    Found(usize),
    /// The word is not in the table; here is the slot it should take.
    Absent(Placement<'w>),
}

// ---------------------------------------------------------------- what a slot holds

/// How many ids can be stored inline in  [`CachedWord`] without the need to spill into the arena.
///
/// Typically enough for 80-96% of lookups on English or code, but only about a quarter of
/// them on Chinese or Korean.
const MAX_INLINE_IDS: usize = 3;

/// A sentinel value stored in [`CachedWord::word_bytes_or_hash`] when it holds the hash of a long word instead of the word itself.
const KEY_IS_HASH: u128 = 1 << 127;

/// A sentinel value stored in [`CachedWord::inline_id_count`] when the ids did not fit in
/// the slot and went to the arena instead.
const SPILLED: u8 = u8::MAX;

/// Bits per length in a spilled entry's `payload[2]`, which packs the key's byte
/// count above the id count. Eleven, because both are bounded by
/// [`MAX_WORD_BYTES`] and two of them have to share one `u32`.
const PACKED_LEN_BITS: u32 = 11;

const PACKED_LEN_MASK: u32 = (1 << PACKED_LEN_BITS) - 1;

const _: () = assert!(MAX_WORD_BYTES <= PACKED_LEN_MASK as usize);

/// One entry, in the three shapes the module docs draw out. In short:
///
/// - `key` is the word itself when it fits in 15 bytes, otherwise its hash with
///   [`KEY_IS_HASH`] set.
/// - `payload` is the token ids while `inline_id_count` counts them, and becomes
///   `[key_off, ids_off, packed lengths]` once that byte is [`SPILLED`] — which
///   is what the `key_off`/`ids_off`/`key_len`/`ids_len` readers below are for.
///
/// Nothing in here is read until the slot's tag has said the word might be in it.
#[derive(Clone, Copy, Default)]
#[repr(C)]
struct CachedWord {
    word_bytes_or_hash: u128,
    word_ids: [u32; MAX_INLINE_IDS],
    inline_id_count: u8,
}

const _: () = assert!(std::mem::size_of::<CachedWord>() == 32);

impl CachedWord {
    fn ids_stored_in_arena(&self) -> bool {
        self.inline_id_count == SPILLED
    }

    fn key_off(&self) -> u32 {
        self.word_ids[0]
    }

    fn ids_off(&self) -> u32 {
        self.word_ids[1]
    }

    fn ids_len(&self) -> usize {
        (self.word_ids[2] & PACKED_LEN_MASK) as usize
    }

    fn key_len(&self) -> usize {
        (self.word_ids[2] >> PACKED_LEN_BITS) as usize
    }
}

/// A word of `1..=15` bytes packed into a `u128`: bytes in the low lanes, length
/// in the top byte. Including the length keeps `"a"` and `"a\0"` apart, and the
/// whole key comparison becomes one register-wide equality instead of a `memcmp`
/// against bytes somewhere else in memory.
///
/// `None` for anything longer, which is the caller's signal to key on a hash
/// instead.
///
/// TODO: the copy is a call into `memcpy` — about a third of what building a key
/// costs — because `len` is only known at run time, and LLVM folds a copy into a
/// load only when the length is a constant. A caller that knows what surrounds
/// the word could pass the 16 bytes starting where the word does instead,
/// turning the copy into one load plus a mask for the surplus bytes.
fn pack_word(word: &[u8]) -> Option<u128> {
    let len = word.len();
    if len == 0 || len > 15 {
        return None;
    }
    let mut lanes = [0u8; 16];
    lanes[..len].copy_from_slice(word);
    Some(u128::from_le_bytes(lanes) | ((len as u128) << 120))
}

// ---------------------------------------------------------------- which entry gives way

/// What a newly stored entry's counter starts at. One rather than zero, so that a
/// fresh entry outranks anything that has already faded to zero — otherwise a word
/// could lose its slot before it is ever looked up again. One hit then lifts it
/// above its untouched neighbours, which start here too.
const INITIAL_COUNTER: u8 = 1;

/// Highest epoch an [`EvictionMetadata`] can hold, which makes a lap 256 epochs
/// long. On the last one, [`Evictor::restart_epochs`] starts the count again.
const MAX_EPOCH: u8 = u8::MAX;

/// Missing this many epochs takes any counter to zero, because a counter is a
/// byte and each missed epoch halves it.
const MAX_HALVINGS: u8 = 8;

/// Decides which entry gives way when a word's window is full.
///
/// Every entry carries a counter of how much its word is being used, bumped on
/// each hit, and the lowest counter in a full window loses its slot. A counter
/// that only ever went up would measure the wrong thing: the first page of a
/// document can make `"the"` the most-used word in the table, and fifty pages
/// later, when the text has moved on to another subject entirely, `"the"` still
/// holds the highest count and nothing can ever catch up with it. What should
/// decide an eviction is how much a word is being used *lately*, so counters have
/// to fade.
///
/// Fading them by hand is the expensive way: every eviction would rewrite all
/// [`PROBE_WINDOW`] counters of a window. So nothing is rewritten. `epoch` counts the
/// fades the table has been through, each [`EvictionMetadata`] records the epoch
/// its counter was written in, and [`EvictionMetadata::score`] halves that
/// counter once for every epoch it has missed. No stored number changes; what
/// changes is what a stored number is worth.
///
/// ```text
///   the table is in epoch 9
///
///     A   counter 32, written in epoch 9  →  0 epochs missed  →  score 32
///     B   counter 32, written in epoch 5  →  4 epochs missed  →  score 2
///                                                                   ▲ B was hot
///                                                                     once, and
///                                                                     has coasted
/// ```
///
/// A counter is one byte, so [`MAX_HALVINGS`] missed epochs take any entry to
/// zero, however hot it once was.
struct Evictor {
    /// One [`EvictionMetadata`] per slot, and then a copy of the first [`PROBE_WINDOW`]
    /// of them, so that the window of any home is `WINDOW` entries in a row and
    /// [`Self::coldest`] can read it as one flat slice. This is the same mirrored
    /// tail [`WordCache::probe_lookup_table`] has, and [`Self::write`] keeps the
    /// copy in step.
    metadata: Box<[EvictionMetadata]>,
    /// Which epoch the table is in, from 0 to [`MAX_EPOCH`] and no higher.
    epoch: u8,
    /// Evictions to go before the next epoch.
    evictions_left: u32,
    /// Evictions per epoch: `N / WINDOW`.
    ///
    /// One epoch fades every counter in the table at once, but eviction pressure
    /// is per window, and this is the rate that lines the two up. A slot can be
    /// reached from [`PROBE_WINDOW`] different homes, so it sits in [`PROBE_WINDOW`] of the
    /// table's `N` windows, and an eviction anywhere in the table lands in any
    /// given window with probability `WINDOW / N`. Fading at that rate gives each
    /// window, on average, one fade per eviction of its own — which is what fading
    /// a window by hand would have done.
    ///
    /// Fading on every eviction instead would fade the whole table at the speed of
    /// whichever window happens to be churning hardest: counters would sit at zero
    /// and every eviction would be an arbitrary pick.
    evictions_per_epoch: u32,
}

impl Evictor {
    fn new(n_slots: usize) -> Self {
        let evictions_per_epoch = (n_slots / PROBE_WINDOW) as u32;
        Self {
            metadata: vec![EvictionMetadata::default(); n_slots + PROBE_WINDOW].into_boxed_slice(),
            epoch: 0,
            evictions_left: evictions_per_epoch,
            evictions_per_epoch,
        }
    }

    /// Count a hit on the entry in `index`, which is a vote for keeping it.
    fn record_hit(&mut self, index: usize) {
        let epoch = self.epoch;
        // Saturating, so hits past what a byte holds are not counted. An entry
        // that busy is in no danger of eviction anyway.
        let counter = self.metadata[index].score(epoch).saturating_add(1);
        self.write(index, EvictionMetadata { counter, epoch });
    }

    /// Start a freshly stored entry off, as of the current epoch.
    fn record_insert(&mut self, index: usize) {
        self.write(
            index,
            EvictionMetadata {
                counter: INITIAL_COUNTER,
                epoch: self.epoch,
            },
        );
    }

    /// The step of the window's lowest-scoring slot, ties going to the slot
    /// nearest home.
    ///
    /// Reads the window as one flat slice, which is what the mirrored tail of
    /// `metadata` is for. Folding each step back into the table instead costs an
    /// `and` per slot and leaves nothing for the compiler to widen.
    fn coldest(&self, home: usize) -> usize {
        let window = &self.metadata[home..home + PROBE_WINDOW];
        // `min_by_key` keeps the first of equal keys, which is what sends a tie
        // to the slot nearest home.
        (0..PROBE_WINDOW)
            .min_by_key(|&step| window[step].score(self.epoch))
            .unwrap()
    }

    /// Charge the table one eviction's worth of fading: step the countdown, and
    /// move the epoch on when it runs out.
    fn on_eviction(&mut self) {
        self.evictions_left -= 1;
        if self.evictions_left > 0 {
            return;
        }
        self.evictions_left = self.evictions_per_epoch;
        if self.epoch == MAX_EPOCH {
            self.restart_epochs();
        } else {
            self.epoch += 1;
        }
    }

    /// Write a slot's [`EvictionMetadata`], and the copy of it in the mirrored
    /// tail when the slot is one of the first [`PROBE_WINDOW`].
    fn write(&mut self, index: usize, metadata: EvictionMetadata) {
        self.metadata[index] = metadata;
        if index < PROBE_WINDOW {
            let n_slots = self.metadata.len() - PROBE_WINDOW;
            self.metadata[n_slots + index] = metadata;
        }
    }

    /// Spend every entry's outstanding fade and start the epoch count again: each
    /// live entry's score becomes its counter, and the epoch and every
    /// [`EvictionMetadata`] go back to zero.
    ///
    /// Run on the last epoch an [`EvictionMetadata`] can hold, and only then. The
    /// epoch has to start over at some point, and the moment it does, the
    /// difference between it and a stored epoch stops meaning anything for entries
    /// written before the restart: an entry idle for the whole lap would read as
    /// freshly written, at full strength, and could never be evicted again. So the
    /// differences are cashed in first, while they can still be read. No entry
    /// moves up or down against another.
    ///
    /// That is one pass over `metadata` every 256 epochs, which at the default
    /// capacity is once every million evictions.
    fn restart_epochs(&mut self) {
        let epoch = self.epoch;
        // The mirrored tail gets the same pass as the front, so both stay in step
        // without a second walk.
        for metadata in self.metadata.iter_mut() {
            metadata.counter = metadata.score(epoch);
            metadata.epoch = 0;
        }
        self.epoch = 0;
    }
}

/// How much an entry is being used, and when that was last written down. One per
/// slot, in [`Evictor::metadata`].
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
#[repr(C)]
struct EvictionMetadata {
    /// How much the word is being used, as of `epoch`.
    counter: u8,
    /// The epoch `counter` was written in.
    epoch: u8,
}

impl EvictionMetadata {
    /// What this entry's counter is worth at `epoch`: what was written down,
    /// halved once for every epoch that has gone by since.
    ///
    /// `self.epoch` is never ahead of `epoch` — both are reset together by
    /// [`Evictor::restart_epochs`] — so the subtraction cannot go below zero.
    fn score(&self, epoch: u8) -> u8 {
        // Widened because shifting a `u8` by MAX_HALVINGS is shifting it by its
        // own width, which Rust leaves undefined.
        (self.counter as u32 >> (epoch - self.epoch).min(MAX_HALVINGS)) as u8
    }
}

// ---------------------------------------------------------------- overflow storage

/// A grow-only buffer that reuses the space of evicted entries instead of
/// compacting.
///
/// Freed runs go on a free list per exact length. That is only practical because
/// [`MAX_WORD_BYTES`] bounds every run: there is a list for every length a run
/// can have, so a freed run is always reusable by the next word of the same shape
/// and no length is ever rounded up to a bigger class. The price is
/// `MAX_WORD_BYTES + 1` empty `Vec`s per arena.
struct Arena<T> {
    data: Vec<T>,
    free: Box<[Vec<u32>]>,
    /// Ceiling on `data`, not a reservation.
    budget: usize,
}

impl<T: Copy + Default> Arena<T> {
    fn new(budget: usize) -> Self {
        Self {
            data: Vec::new(),
            free: (0..=MAX_WORD_BYTES).map(|_| Vec::new()).collect(),
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

// ---------------------------------------------------------------- reading a window of tags

/// How many cache slots a probe can check at the same time.
///
/// Chose 16 because 16 x u8 is u128 which allows SIMD within a register techniques.
///
/// TODO: link to the probe code
const PROBE_WINDOW: usize = 16;

/// A sentinel value used to mark an empty [`CachedWord`].
const EMPTY: u8 = 0;

/// Set in a tag beside the hash bits, so that a live entry's tag is never
/// [`EMPTY`] — which is also what makes [`empty_slots`] a single `and`.
const OCCUPIED: u8 = 0x80;

/// How many bits of the hash a tag carries. Seven, because the eighth is
/// [`OCCUPIED`]; a probe therefore reads a slot it did not want once in 128.
const LOOKUP_TAG_BITS: u32 = 7;

/// The tag a hash gives its slot.
///
/// The top of the hash, because the bottom of it already chose the home slot: a
/// tag built from those bits would be the same for every slot the word can reach
/// and would tell a probe nothing.
fn make_lookup_tag(hash: u64) -> u8 {
    OCCUPIED | (hash >> (u64::BITS - LOOKUP_TAG_BITS)) as u8
}

/// Every byte of one of these is `0x01`, or `0x80`. Broadcasting a value over all
/// [`PROBE_WINDOW`] lanes and testing every lane's top bit is what turns a window into
/// arithmetic — see [`StepMask`].
const ONES: u128 = u128::from_le_bytes([0x01; PROBE_WINDOW]);

const HIGHS: u128 = u128::from_le_bytes([0x80; PROBE_WINDOW]);

/// One window's worth of yes-or-no, as the top bit of each of a `u128`'s bytes:
/// byte `step` answers for the slot `step` places along from home.
///
/// A window is [`PROBE_WINDOW`] tags, which is sixteen bytes, which is one `u128`. So
/// the questions a probe asks are answered for the whole window by ordinary
/// integer arithmetic, in about as many instructions as it takes to ask one of
/// them of one slot. No vector types, nothing target-specific: the same handful
/// of `and`s and `xor`s on every machine this builds for.
type StepMask = u128;

/// The first step a mask answers `true` for, or [`PROBE_WINDOW`] when it answers `true`
/// for none.
///
/// A flagged step `s` has its bit at `8 * s + 7`, so the count of trailing zeros
/// divided by eight is the step — and an empty mask counts 128 zeros, which
/// divides to [`PROBE_WINDOW`] and reads as "nothing".
fn first_step(mask: StepMask) -> usize {
    (mask.trailing_zeros() / 8) as usize
}

/// The same mask with `step` taken out of it.
fn clear_step(mask: StepMask, step: usize) -> StepMask {
    mask & !(0x80 << (step * 8))
}

/// Everything a mask says about the steps *before* the first one `bound` answers
/// for. All of it when `bound` answers for none.
///
/// `bound - 1` clears the lowest set bit and sets every bit below it, and
/// `!bound` drops the higher ones it left alone — so this is branchless, and the
/// all-of-it case falls out of `0 - 1` being all ones.
fn steps_before(bound: StepMask) -> StepMask {
    bound.wrapping_sub(1) & !bound
}

/// The tags of one window, read as a single value.
///
/// `from_le_bytes` puts the tag of the slot `step` places from home in byte
/// `step`, on a big-endian machine as much as on a little-endian one, so
/// [`first_step`] means the same thing everywhere.
fn read_window(tags: &[u8; PROBE_WINDOW]) -> u128 {
    u128::from_le_bytes(*tags)
}

/// Which steps of the window are empty slots.
///
/// Exact, and one instruction: a live tag always carries [`OCCUPIED`] and
/// [`EMPTY`] is zero, so "empty" is precisely "top bit clear".
fn empty_slots(window: u128) -> StepMask {
    !window & HIGHS
}

/// Which steps of the window could hold the word `tag` belongs to.
///
/// The `xor` is zero exactly in the lanes that match, and the rest is the
/// standard hunt for a zero byte: subtracting one from every lane borrows out of
/// a zero lane into its top bit, and `!x` keeps only the lanes that were small
/// enough for that top bit to be news.
///
/// A borrow can run on into the next lane, so a lane holding exactly `tag ^ 1`
/// directly above a match is answered `true` as well. It cannot go the other
/// way — a matching lane is never missed — and the probe confirms every answer
/// against the key anyway, so the cost of the rare extra one is a slot read.
fn tag_matches(window: u128, tag: u8) -> StepMask {
    let x = window ^ u128::from_le_bytes([tag; PROBE_WINDOW]);
    x.wrapping_sub(ONES) & !x & HIGHS
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Look a word up and store what it encoded to, for tests that care about
    /// what the table ends up holding rather than about the two steps.
    fn store(cache: &mut WordCache, word: &[u8], ids: impl ExactSizeIterator<Item = u32>) {
        let at = match cache.lookup(word) {
            Lookup::Miss(at) => at,
            Lookup::Hit(_) => None,
        };
        if let Some(at) = at {
            cache.insert(at, ids);
        }
    }

    /// What the entry in `index` is worth as of the table's current epoch.
    fn score(cache: &WordCache, index: usize) -> u8 {
        cache.evictor.metadata[index].score(cache.evictor.epoch)
    }

    /// The slot a word is in, or `None` if the table does not hold it.
    fn slot_of(cache: &WordCache, word: &[u8]) -> Option<usize> {
        let (key, hash) = cache.make_word_key(word);
        match cache.find_word_in_cache(key, hash, word) {
            Probe::Found(index) => Some(index),
            Probe::Absent(_) => None,
        }
    }

    #[test]
    fn roundtrip() {
        let mut cache = WordCache::new(1 << 8);
        assert_eq!(cache.lookup(b"hello").hit(), None);
        store(&mut cache, b"hello", [1u32, 2, 3].into_iter());
        store(&mut cache, b"world", [4u32].into_iter());
        assert_eq!(cache.lookup(b"hello").hit(), Some(&[1u32, 2, 3][..]));
        assert_eq!(cache.lookup(b"world").hit(), Some(&[4u32][..]));
        assert_eq!(cache.lookup(b"hell").hit(), None);
    }

    /// The two properties the key encoding rests on: a packed key is unique per
    /// word, and never looks like a hashed one.
    #[test]
    fn packed_keys_are_unique_and_never_look_hashed() {
        assert_ne!(pack_word(b"a"), pack_word(b"a\0"));
        assert_eq!(pack_word(&[0u8; 15]).unwrap() & KEY_IS_HASH, 0);
        assert_eq!(pack_word(b""), None);
        assert_eq!(pack_word(&[b'x'; 16]), None);
    }

    /// A live entry's tag has to be something [`EMPTY`] is not, or a probe reads
    /// an occupied slot as the end of the chain and stores on top of it.
    #[test]
    fn a_live_tag_is_never_the_empty_marker() {
        for hash in [0u64, 1, u64::MAX, 1 << 57, u64::MAX >> LOOKUP_TAG_BITS] {
            assert_ne!(make_lookup_tag(hash), EMPTY, "hash {hash:#x}");
        }
    }

    /// A short word's placement comes out of its packed key rather than its
    /// bytes, which leaves the hash doing all of the mixing. Words that differ in
    /// one byte pack to keys that differ in one byte, so an index taken from those
    /// bits as they are — `packed as u64` — drops most of these on one slot.
    #[test]
    fn short_words_spread_across_the_table() {
        let cache = WordCache::new(1 << 12);
        let homes: std::collections::HashSet<usize> = (0..1000)
            .map(|i| {
                let (_, hash) = cache.make_word_key(format!("tok{i}").as_bytes());
                hash as usize & cache.index_mask
            })
            .collect();
        // 1000 words over 4096 slots share homes by chance alone; ~887 distinct is
        // as good as a perfect hash gets, so the floor is well under it.
        assert!(homes.len() > 820, "only {} distinct homes", homes.len());
    }

    #[test]
    fn oversized_words_are_ignored() {
        let mut cache = WordCache::new(1 << 8);
        let big = vec![7u8; MAX_WORD_BYTES + 1];
        store(&mut cache, &big, [1u32].into_iter());
        assert_eq!(cache.lookup(&big).hit(), None);
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
            store(&mut cache, word, ids.clone().into_iter());
        }
        for (word, ids) in &cases {
            assert_eq!(cache.lookup(word).hit(), Some(&ids[..]), "{word:?}");
        }
    }

    /// A spilled entry keeps the word's byte count and its id count in eleven bits
    /// each, side by side in one `u32`. The longest word the cache accepts, when it
    /// encodes to one id per byte, is the largest either of them can get — and if
    /// they ever overlapped, the entry would come back as a different word's.
    #[test]
    fn a_word_at_the_length_limit_round_trips() {
        let mut cache = WordCache::new(1 << 8);
        let word = vec![b'q'; MAX_WORD_BYTES];
        let ids: Vec<u32> = (0..MAX_WORD_BYTES as u32).collect();
        store(&mut cache, &word, ids.clone().into_iter());
        assert_eq!(cache.lookup(&word).hit(), Some(&ids[..]));
    }

    /// A tag is seven bits, so one slot in 128 answers for a word that is not in
    /// it. That is too rare to wait for, so forge one: put a word's tag on a slot
    /// holding a different key, and demand the probe look past it rather than
    /// treat the tag as the answer.
    #[test]
    fn a_tag_collision_is_confirmed_against_the_key() {
        let mut cache = WordCache::new(PROBE_WINDOW);
        let (_, hash) = cache.make_word_key(b"beta");
        let decoy = hash as usize & cache.index_mask;
        cache.update_lookup_table(decoy, make_lookup_tag(hash));
        cache.evictor.write(
            decoy,
            EvictionMetadata {
                counter: INITIAL_COUNTER,
                epoch: 0,
            },
        );
        // Any key that is not beta's. A packed key always carries a length in its
        // top byte, so 1 cannot be one.
        cache.cached_words[decoy] = CachedWord {
            word_bytes_or_hash: 1,
            word_ids: [7, 0, 0],
            inline_id_count: 1,
        };

        store(&mut cache, b"beta", [2u32].into_iter());
        assert_eq!(cache.lookup(b"beta").hit(), Some(&[2u32][..]));
    }

    /// Two long words can hash to the same key, and then only their bytes tell
    /// them apart. Real hash collisions are too rare to write a test around, so
    /// forge one: park another word's entry on this word's home slot, stamp this
    /// word's key and tag on it, and demand the probe look past it.
    #[test]
    fn a_hashed_key_is_confirmed_against_the_word_bytes() {
        let mut cache = WordCache::new(1 << 8);
        let mine = vec![b'a'; 40];
        let theirs = vec![b'b'; 40];

        store(&mut cache, &theirs, [7u32].into_iter());
        let their_index = slot_of(&cache, &theirs).unwrap();
        let their_slot = cache.cached_words[their_index];
        let their_usage = cache.evictor.metadata[their_index];
        let (my_key, my_hash) = cache.make_word_key(&mine);
        let my_home = my_hash as usize & cache.index_mask;
        cache.cached_words[my_home] = CachedWord {
            word_bytes_or_hash: my_key,
            ..their_slot
        };
        cache.update_lookup_table(my_home, make_lookup_tag(my_hash));
        cache.evictor.write(my_home, their_usage);

        store(&mut cache, &mine, [1u32, 2].into_iter());
        assert_eq!(cache.lookup(&mine).hit(), Some(&[1u32, 2][..]));
    }

    /// A word that hashes into a full window takes the lowest-scoring slot rather
    /// than being turned away, and the words that were actually used keep their
    /// place.
    #[test]
    fn a_full_window_evicts_its_coldest_entry() {
        let mut cache = WordCache::new(PROBE_WINDOW);
        let words: Vec<Vec<u8>> = (0..PROBE_WINDOW as u8).map(|i| vec![i; 4]).collect();
        for (i, word) in words.iter().enumerate() {
            store(&mut cache, word, [i as u32].into_iter());
        }
        // Every word but the first earns a hit, leaving one clear coldest slot.
        for word in &words[1..] {
            assert!(cache.lookup(word).hit().is_some());
        }
        store(&mut cache, b"newcomer", [999u32].into_iter());
        assert_eq!(cache.lookup(b"newcomer").hit(), Some(&[999u32][..]));
        assert_eq!(
            cache.lookup(&words[0]).hit(),
            None,
            "the unused entry should have gone"
        );
        for (i, word) in words.iter().enumerate().skip(1) {
            assert_eq!(
                cache.lookup(word).hit(),
                Some(&[i as u32][..]),
                "used entry {i} was dropped"
            );
        }
    }

    /// An entry that stops being used has to fade even though nothing ever writes
    /// to it. Here one word is hit until its counter saturates and then abandoned,
    /// while other words keep the table busy.
    #[test]
    fn an_abandoned_entry_fades_without_being_touched() {
        let mut cache = WordCache::new(PROBE_WINDOW);
        store(&mut cache, b"hot", [1u32].into_iter());
        for _ in 0..u8::MAX {
            cache.lookup(b"hot").hit();
        }
        let index = slot_of(&cache, b"hot").unwrap();
        assert_eq!(score(&cache, index), u8::MAX);

        for _ in 0..MAX_HALVINGS {
            cache.evictor.on_eviction();
        }
        assert_eq!(score(&cache, index), 0);
    }

    /// An epoch is a byte, so the count has to start over at zero every 256
    /// bumps. An entry written before the restart carries an epoch larger than
    /// the current one, and left to itself the coldest thing in the table would
    /// come out reading as the hottest and could never be evicted again.
    #[test]
    fn a_lap_of_the_epoch_does_not_make_a_stale_entry_look_hot() {
        let mut cache = WordCache::new(PROBE_WINDOW);
        store(&mut cache, b"stale", [1u32].into_iter());
        for _ in 0..u8::MAX {
            cache.lookup(b"stale").hit();
        }
        let index = slot_of(&cache, b"stale").unwrap();

        // One slot per window here, so one eviction is one epoch.
        assert_eq!(cache.evictor.evictions_per_epoch, 1);
        for _ in 0..=MAX_EPOCH {
            cache.evictor.on_eviction();
        }
        assert_eq!(cache.evictor.epoch, 0, "the lap did not start over");
        assert_eq!(score(&cache, index), 0);
    }

    /// The tail of `tags` is a copy of its first [`PROBE_WINDOW`] entries, so that a
    /// window starting near the end of the table is still `WINDOW` entries in a
    /// row. Every write has to keep the two in step, or a probe whose window
    /// crosses the seam reads control bytes that no longer describe the slots.
    #[test]
    fn the_mirrored_tail_tracks_the_front() {
        let mut cache = WordCache::new(64);
        let n = cache.cached_words.len();
        for i in 0..500usize {
            store(
                &mut cache,
                format!("w{i}").as_bytes(),
                [i as u32].into_iter(),
            );
            assert_eq!(
                cache.probe_lookup_table[..PROBE_WINDOW],
                cache.probe_lookup_table[n..],
                "tags, insert {i}"
            );
            assert_eq!(
                cache.evictor.metadata[..PROBE_WINDOW],
                cache.evictor.metadata[n..],
                "eviction metadata, insert {i}"
            );
        }
        // Restarting the epochs rewrites every live entry's metadata and no tag
        // at all, so the two must still agree afterwards.
        cache.evictor.restart_epochs();
        assert_eq!(
            cache.probe_lookup_table[..PROBE_WINDOW],
            cache.probe_lookup_table[n..],
            "tags, after restart"
        );
        assert_eq!(
            cache.evictor.metadata[..PROBE_WINDOW],
            cache.evictor.metadata[n..],
            "eviction metadata, after restart"
        );
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
            store(&mut cache, &word, ids.clone().into_iter());
            expected.push((word, ids));
        }
        let mut live = 0;
        for (word, ids) in &expected {
            if let Some(hit) = cache.lookup(word).hit() {
                assert_eq!(hit, &ids[..], "{word:?}");
                live += 1;
            }
        }
        assert!(live > 0, "everything was evicted — the test proves nothing");
    }

    /// Every step a mask answers for, so it can be read against a picture of the
    /// window it came from.
    fn steps(mask: StepMask) -> Vec<usize> {
        let mut left = mask;
        let mut found = Vec::new();
        while left != 0 {
            let step = first_step(left);
            found.push(step);
            left = clear_step(left, step);
        }
        found
    }

    /// The arithmetic in [`tag_matches`], [`empty_slots`] and [`steps_before`]
    /// read back as the slots they are talking about.
    #[test]
    fn a_window_answers_for_the_slots_it_should() {
        let mut tags = [EMPTY; PROBE_WINDOW];
        tags[0] = 0x81;
        tags[3] = 0xC4;
        tags[5] = 0x81;
        tags[9] = 0x81;
        let window = read_window(&tags);

        assert_eq!(steps(tag_matches(window, 0x81)), [0, 5, 9]);
        assert_eq!(steps(tag_matches(window, 0xC4)), [3]);
        assert_eq!(
            steps(empty_slots(window)),
            [1, 2, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15]
        );
        // Slot 1 is empty, so slots 5 and 9 are out of reach: whatever put a word
        // there would have stopped at slot 1 and taken it.
        assert_eq!(
            steps(tag_matches(window, 0x81) & steps_before(empty_slots(window))),
            [0]
        );
    }

    /// A full window has no empty slot to stop at, so every match stays in reach.
    #[test]
    fn a_full_window_bounds_nothing() {
        let tags = [OCCUPIED | 0x11; PROBE_WINDOW];
        let window = read_window(&tags);
        assert_eq!(empty_slots(window), 0);
        assert_eq!(
            steps(tag_matches(window, OCCUPIED | 0x11) & steps_before(0)).len(),
            PROBE_WINDOW
        );
    }

    /// [`tag_matches`] hunts for a zero byte by subtracting one from every lane,
    /// and a borrow out of a matching lane can run on into the one above it — so a
    /// slot holding `tag ^ 1` just above a match is answered for as well. That
    /// costs one slot read, which the key comparison then rejects.
    ///
    /// What it must never do is the other thing. A tag that is really there and
    /// goes unanswered is a word that reads as absent, and the table would grow a
    /// second entry for it.
    #[test]
    fn tag_matches_never_misses_a_tag() {
        for tag in OCCUPIED..=u8::MAX {
            for step in 0..PROBE_WINDOW {
                let mut tags = [OCCUPIED; PROBE_WINDOW];
                tags[step] = tag;
                assert!(
                    steps(tag_matches(read_window(&tags), tag)).contains(&step),
                    "tag {tag:#x} at step {step} went unanswered"
                );
            }
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
            store(&mut cache, word(i).as_bytes(), [i as u32; 8].into_iter());
        }
        assert_eq!(
            cache.lookup(word(4999).as_bytes()).hit(),
            Some(&[4999u32; 8][..]),
            "inserts stopped landing — the arenas ran out"
        );
        assert!(cache.word_bytes_arena.data.len() <= 65 * 35);
        assert!(cache.token_ids_arena.data.len() <= 65 * 8);
    }
}
