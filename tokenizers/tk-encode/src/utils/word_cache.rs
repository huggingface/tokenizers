//! A table that remembers which token ids a word encodes to, so a model only has
//! to work it out once.
//!
//! # Context
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
//! Two things make that expense avoidable:
//! - Words repeat, in every kind of text.
//! - A word always encodes to the same ids.
//!
//! So a word only ever has to go through the model once. Cache the ids that came out, and
//! every later occurrence of that word skips the model altogether.
//!
//! This module is the table that does the remembering, in bounded memory:
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
//!    │  WordCache: have I encoded this word yet?   │
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
//! # What the cache is allowed to get wrong
//!
//! The cache is free to *forget*: if a word's ids are gone, the model works them
//! out again. What it must never do is hand back the *wrong* ids. Every
//! trade-off below spends the first freedom and none of the second.
//!
//! # The table
//!
//! The cache is one long row of numbered **slots**. Each slot holds one word and
//! the ids that word encodes to. The number of slots is fixed when the cache is
//! built and never changes, which is what keeps the memory bounded.
//!
//! To decide where a word belongs, the cache turns the word into a number, its
//! **hash**. Two different pieces of that number are used, and they never overlap:
//!
//! ```text
//!                          "the"
//!                            │
//!                            │  hash
//!                            ▼
//!         ┌─────────────────────────────────────────┐
//!         │ 1010 0111 ................... 0000 0101 │
//!         └────┬───────────────────────────────┬────┘
//!              │ the top 8 bits                │ the bottom bits
//!              ▼                               ▼
//!         the tag: A7                    the home slot: 5
//! ```
//!
//! The **home slot** is where the word would like to live. The **tag** is a
//! one-byte summary of the word. Tags are kept in their own row, one byte per
//! slot, beside the row of entries:
//!
//! ```text
//!    slot index:    0     1     2     3     4     5     6     7   ...
//!                ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
//!    tags        │  ·  │ 9B  │  ·  │  ·  │ C4  │ A7  │ 31  │ A7  │   1 byte each
//!                ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
//!    entries     │     │"of" │     │     │"cat"│"the"│"sat"│"hat"│  32 bytes each
//!                └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
//!                                                 ▲
//!                                                 home slot of "the",
//!                                                 and of "hat" as well
//!
//!                  · = empty slot
//! ```
//!
//! A word does not always get its home slot: another word may have taken it
//! first, which is what happened to "hat" above. So the word may be in its home
//! slot or in any of the fifteen slots after it. Those sixteen slots are the
//! word's **window**, and the word is either in there or nowhere
//! ([`WALK_WINDOW`]).
//!
//! # Looking a word up
//!
//! [`WordCache::lookup`] walks the window one tag at a time, starting at the home
//! slot, and answers both of a lookup's questions on the way: which slot holds
//! this word, and where would this word go if none of them does.
//!
//! ```text
//!    looking up "hat" (tag A7, home slot 5)
//!
//!    slot:          5     6     7     8     9    10    11    12  ...
//!                ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
//!    tags        │ A7  │ 31  │ A7  │  ·  │ 5F  │ A7  │  ·  │  ·  │
//!                ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
//!    entries     │"the"│"sat"│"hat"│     │"on" │"mat"│     │     │
//!                └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
//!                   ▲           ▲     ▲           ▲
//!                   │           │     │           └ tagged A7 too, but past the
//!                   │           │     │             empty slot: out of reach
//!                   │           │     └ empty: the search stops here
//!                   │           └ tag matches, and the key says "hat": found it
//!                   └ tag matches, but the key says "the", so keep looking
//! ```
//!
//! The walk reads tags, not entries: ruling a slot out costs one byte of memory
//! traffic rather than the 32 an entry takes. It rarely gets far. On running text a
//! word is usually in its home slot, or the slot after it, or not in the table at
//! all.
//!
//! A matching tag is a hint, not an answer. A tag is only one byte of the hash, so
//! about one slot in 255 matches a word that is not there, which is exactly what
//! slot 5 does above. Each matching slot is therefore read and its **key**
//! compared. The key is the copy of the word kept inside the slot, and it is what
//! settles the question.
//!
//! The search stops at the first empty slot. That is safe because storing stops
//! there too: a word is always put in the first empty slot of its window, so no
//! word can ever sit past one. This is why slot 10 is out of reach: whatever is
//! in it, it is not "hat", because "hat" would have taken slot 8.
//!
//! A lookup ends one of two ways ([`Lookup`]):
//!
//! - **Hit**: the ids, ready to use. Nothing is written: a hit leaves the table
//!   exactly as it was.
//! - **Miss**: a [`Placement`], the slot this word should take. The caller runs
//!   the model, then hands the placement back to [`WordCache::insert`] along with
//!   the ids. The placement carries the hash work the lookup already did, so
//!   storing the word costs no second hash and no second read of the tags.
//!
//! # Storing a word
//!
//! [`WordCache::insert`] takes the placement from the miss and the ids the model
//! produced, and does four things:
//!
//! 1. **Builds the entry.** The word and its ids go in the slot if they fit; what
//!    does not fit is copied into an overflow buffer (below). If those buffers are
//!    full, the insert is dropped and nothing changes, since the cache is allowed
//!    to forget.
//! 2. **Clears the old entry, if there was one.** When the window had no empty
//!    slot the placement points at the home slot, and that word's ids are lost.
//!    Any overflow space it was using is handed back.
//! 3. **Writes the tag**, which is what makes the slot findable.
//! 4. **Writes the entry.**
//!
//! # What a slot holds
//!
//! A slot is 32 bytes: sixteen for the key, twelve for the ids, one that says how
//! many ids there are, and three of padding. It comes in three shapes
//! ([`CachedWord`]). The first is the common case, and the reason the sizes are what
//! they are: the whole answer (word and ids) is right there, so a hit reads one
//! slot and stops.
//!
//! ```text
//!                                    ├───── key ─────┤├─── ids ───┤├─────┤
//!                                                                    how many
//!
//!    "the" → [464]                   ┌────────────────┬────────────┬─────┐
//!    short word, up to 3 ids         │ t h e        3 │ 464        │  1  │
//!                                    └────────────────┴────────────┴─────┘
//!                                      the word, then    the ids
//!                                      its length        themselves
//!
//!    "hello" → [40,71,12,9,33]       ┌────────────────┬────────────┬─────┐
//!    short word, too many ids        │ h e l l o    5 │ ids are at │  ▒  │
//!                                    └────────────────┴────────────┴─────┘
//!
//!    "counterrevolutionary" → …      ┌────────────────┬────────────┬─────┐
//!    word too long for the key       │ hash of word H │ both are at│  ▒  │
//!                                    └────────────────┴────────────┴─────┘
//!                                      H says "a hash,   where to look in
//!                                      not the word"     the buffers
//!
//!                                    ▒ = "not in the slot, follow the offsets"
//! ```
//!
//! A word of up to 15 bytes is kept whole, with its length in the sixteenth byte.
//! A longer word cannot fit, so the slot keeps the word's hash instead and the
//! bytes go to a buffer. Two different long words can hash to the same number, so
//! for those, and only those, a match is confirmed by comparing the bytes.
//!
//! The two overflow buffers ([`Arena`]) grow up to a budget and then stop. When an
//! entry is evicted, the space it used goes on a free list for its exact
//! length, so the next word of that shape reuses it. Nothing is ever compacted or
//! moved. Words longer than [`MAX_WORD_BYTES`] are not cached at all.
//!
//! # Which entry gets evicted
//!
//! When a word's window is full, one of its entries gets evicted, and it is whatever
//! sits in the home slot. Nothing is measured and nothing is ranked: the newcomer
//! takes the slot it wanted in the first place.
//!
//! A cleverer rule is possible, and this module used to have one: a use counter per
//! entry, the least-used slot of the window loses it, and the counters fade so that
//! a word that was busy early in a document cannot keep its slot forever. Measured
//! against the plain rule, it did keep slightly more of the words worth keeping, and
//! it lost more than it gained to the counter that every hit had to write, so it
//! went. Two things let the plain rule get away with it: nothing is evicted until
//! all sixteen slots of a window are taken, and words repeat closely enough that a
//! word's next use usually comes round before anything has had the chance to evict
//! it.
//!
//! # Why it is built this way
//!
//! - **Tags live in their own row** so that a walk over sixteen slots reads sixteen
//!   bytes, which is one or two cache lines. The entries are 32 bytes each, so the
//!   same sixteen slots would be 512 bytes and eight cache lines.
//! - **A tag is one byte of the hash**, so a slot that cannot hold the word is ruled
//!   out without reading it. All eight bits carry hash: emptiness needs a value no
//!   live tag can take rather than a bit of its own, and one value is cheaper than
//!   one bit ([`make_tag`]).
//! - **The tag uses the top bits of the hash** because the bottom bits already
//!   chose the home slot, and every slot in a window would share those.
//! - **A window is sixteen slots**: how far a word may end up from its home slot
//!   before the cache stops looking and evicts instead. Long enough that a full
//!   window is uncommon, short enough that walking a full one stays cheap.
//! - **The key is 128 bits** because 15 bytes of word plus one byte of length
//!   covers nearly every word whole, and comparing it is then one comparison of one
//!   value instead of following a pointer to bytes elsewhere in memory.
//! - **The length is part of the key**, or `"a"` and `"a\0"` would look identical.
//! - **Up to three ids fit in the slot** because most words encode to one, two or
//!   three, and that makes a hit a single read.
//! - **The number of slots is a power of two**, so picking the home slot is one
//!   bit operation rather than a division. So is folding a step of the walk back
//!   into the table when a window runs off the end.
//! - **A miss hands back a placement** because the lookup already knows the hash,
//!   the tag and the slot to use, and making the insert work them out again would
//!   double that cost on every new word.
//! - **Freed overflow space is filed by exact length**: every run is at most
//!   [`MAX_WORD_BYTES`] long, so there can be one free list per length and a freed
//!   run always fits the next word of that shape exactly.
//!
//! # Where the ideas come from
//!
//! - [Swiss Tables] is where the control byte comes from: one byte per slot, holding
//!   a flag bit and seven bits of hash. Here the whole byte is hash, because the flag
//!   bit was there to be tested sixteen lanes at a time and this walk tests one.
//! - [gigatoken] is a BPE tokenizer with a pre-token cache built from the same
//!   parts: `u128` packed keys with the length in the top byte, self-contained
//!   32-byte entries, ids inline. It never evicts (it doubles at 3/4 load) and
//!   leans on huge pages and prefetching, because its table is sized for DRAM
//!   rather than for a CPU cache.
//! - [TinyLFU] (Einziger, Friedman & Manes) is the use-counter-and-fade rule this
//!   module tried for eviction and dropped, as its reference implementation
//!   [Caffeine] does it. Kept here for whoever wants to try it again.
//! - [huggingface/tokenizers#2234] is an open-addressed cache for this same encode
//!   pipeline, arrived at in parallel, fused into the pre-tokenizer's split loop.
//!
//! [Swiss Tables]: https://abseil.io/about/design/swisstables
//! [gigatoken]: https://github.com/marcelroed/gigatoken
//! [TinyLFU]: https://arxiv.org/abs/1512.00727
//! [Caffeine]: https://github.com/ben-manes/caffeine/blob/master/caffeine/src/main/java/com/github/benmanes/caffeine/cache/FrequencySketch.java
//! [huggingface/tokenizers#2234]: https://github.com/huggingface/tokenizers/pull/2234

use ahash::RandomState;

// ---------------------------------------------------------------- the cache

/// Longest word the cache will store, in bytes.
const MAX_WORD_BYTES: usize = 1024;

/// Word bytes to token ids. See the module docs for the design.
pub struct WordCache {
    /// The slots, one word each. Their number is a power of two, so a word's home
    /// slot is just the bottom bits of its hash:
    /// ```text
    /// hash(word) & index_mask
    /// ```
    cached_words: Box<[CachedWord]>,

    /// Hashes a slot's key: the packed word when it is short enough, the word's
    /// bytes when it is not. See [`WordCache::make_word_key`].
    hasher: RandomState,

    /// `cached_words.len() - 1`. Masks a hash down to its bottom bits, which give the
    /// word's home slot.
    index_mask: usize,

    /// One tag per slot, at the slot's own index: the top byte of the word's hash
    /// ([`make_tag`]), or [`EMPTY`] when the slot holds nothing.
    ///
    /// A walk reads this row instead of the entries, so ruling a slot out costs one
    /// byte rather than the 32 an entry takes.
    tags: Box<[u8]>,

    /// Holds the word's bytes when they don't fit in a [`CachedWord`] key.
    word_bytes_arena: Arena<u8>,

    /// Holds the word's ids when they don't fit in a [`CachedWord`].
    token_ids_arena: Arena<u32>,
}

impl WordCache {
    /// `capacity` is rounded up to a power of two, and to at least one full window.
    pub fn new(capacity: usize) -> Self {
        let n_slots = capacity.next_power_of_two().max(WALK_WINDOW);
        Self {
            hasher: RandomState::new(),
            cached_words: vec![CachedWord::default(); n_slots].into_boxed_slice(),
            tags: vec![EMPTY; n_slots].into_boxed_slice(),
            word_bytes_arena: Arena::new(n_slots * 48),
            token_ids_arena: Arena::new(n_slots * 16),
            index_mask: n_slots - 1,
        }
    }

    /// The ids `word` encoded to last time, or the [`Placement`] it should be stored
    /// in once the model has worked them out. See [`Lookup`].
    pub fn lookup<'c, 'w>(&'c self, word: &'w [u8]) -> Lookup<'c, 'w> {
        if word.len() > MAX_WORD_BYTES {
            return Lookup::Miss(None);
        }

        let (key, hash) = self.make_word_key(word);
        match self.find_word_in_cache(key, hash, word) {
            Walk::Found(index) => {
                let slot = self.cached_words[index];
                Lookup::Hit(if slot.ids_stored_in_arena() {
                    self.token_ids_arena.get(slot.ids_off(), slot.ids_len())
                } else {
                    &self.cached_words[index].word_ids[..slot.inline_id_count as usize]
                })
            }
            Walk::Absent(placement) => Lookup::Miss(Some(placement)),
        }
    }

    /// Store `ids` as the encoding of the word that `at` was built for.
    ///
    /// Overwrites whatever the slot held, which only happens when the word's window
    /// was full. The module docs say why the choice does not have to be cleverer than
    /// that. If the arenas have no room for the entry, nothing is stored.
    pub fn insert(&mut self, at: Placement<'_>, ids: impl ExactSizeIterator<Item = u32>) {
        let Some(cached_word) = self.build_entry(at.key, at.word, ids) else {
            return;
        };
        if self.tags[at.index] != EMPTY {
            self.reclaim(at.index);
        }
        self.tags[at.index] = at.tag;
        self.cached_words[at.index] = cached_word;
    }

    /// The value a slot stores in its key, and the hash that places it. A short
    /// word is its own key; a longer one keys on its hash and has to be confirmed
    /// against [`WordCache::word_bytes_arena`], since two long words can hash alike.
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

    /// One walk over `word`'s window, from its home slot, answering both
    /// questions a lookup has: which slot holds the word, and which slot it should
    /// take if none of them does.
    ///
    /// The second answer comes out of the same walk, so a word that is not in the
    /// table is placed without hashing it again or reading the tags again.
    ///
    /// The walk stops at the first empty slot, which is safe because storing stops
    /// there too: an empty slot means the word was never stored.
    fn find_word_in_cache<'w>(&self, key: u128, hash: u64, word: &'w [u8]) -> Walk<'w> {
        let home = hash as usize & self.index_mask;
        let tag = make_tag(hash);
        // Only a hashed key can turn out to belong to a different word, and
        // whether this one is hashed is settled before the walk starts, since the
        // caller built the key out of the word it is looking for.
        let hashed = key & KEY_IS_HASH != 0;
        let mut free = None;
        for step in 0..WALK_WINDOW {
            let index = (home + step) & self.index_mask;
            let slot_tag = self.tags[index];
            if slot_tag == EMPTY {
                free = Some(index);
                break;
            }
            if slot_tag == tag {
                let slot = &self.cached_words[index];
                if slot.word_bytes_or_hash == key
                    && (!hashed
                        || self.word_bytes_arena.get(slot.word_off(), slot.word_len()) == word)
                {
                    return Walk::Found(index);
                }
            }
        }
        Walk::Absent(Placement {
            // A full window means one of its words gets evicted, and the one that
            // does is the word in the home slot. The module docs say why the
            // choice does not have to be any cleverer than that.
            index: free.unwrap_or(home),
            key,
            tag,
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
        let word_len = if long { word.len() } else { 0 };
        let word_off = self.word_bytes_arena.alloc(word_len)?;
        let Some(ids_off) = self.token_ids_arena.alloc(ids_len) else {
            self.word_bytes_arena.release(word_off, word_len);
            return None;
        };
        self.word_bytes_arena
            .fill(word_off, word_len, word.iter().copied());
        self.token_ids_arena.fill(ids_off, ids_len, ids);
        let lengths = (word_len as u32) << PACKED_LEN_BITS | ids_len as u32;
        Some(CachedWord {
            word_bytes_or_hash: key,
            word_ids: [word_off, ids_off, lengths],
            inline_id_count: SPILLED,
        })
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
            .release(slot.word_off(), slot.word_len());
        self.token_ids_arena.release(slot.ids_off(), slot.ids_len());
    }
}

// ---------------------------------------------------------------- what a lookup returns

/// What [`WordCache::lookup`] found.
pub enum Lookup<'c, 'w> {
    /// The ids the word encoded to last time.
    Hit(&'c [u32]),
    /// The word is not in the table. The [`Placement`] is the slot it should go
    /// in. Hand it to [`WordCache::insert`] once the model has done the work, or
    /// drop it and nothing is stored. `None` when the word is too long to be
    /// worth a slot at all.
    Miss(Option<Placement<'w>>),
}

impl<'c> Lookup<'c, '_> {
    /// The ids, throwing the [`Placement`] away. Every caller in the encoder
    /// wants the placement, so this is for tests asserting on what the table holds.
    #[cfg(test)]
    pub fn hit(self) -> Option<&'c [u32]> {
        match self {
            Lookup::Hit(ids) => Some(ids),
            Lookup::Miss(_) => None,
        }
    }
}

/// Where a word that missed will go, and what [`WordCache::insert`] needs to put
/// it there. Built by [`WordCache::find_word_in_cache`] out of what the walk had
/// already worked out, so storing the word costs no second hash, no second walk
/// over the tags and no second tag.
///
/// It carries the word rather than letting `insert` take it again, because the
/// slot, the key and the tag inside were chosen for *this* word: handing back a
/// different one would file its ids under the first word's name.
pub struct Placement<'w> {
    index: usize,
    key: u128,
    tag: u8,
    word: &'w [u8],
}

/// How a walk over a word's window ended.
enum Walk<'w> {
    /// The word is in this slot.
    Found(usize),
    /// The word is not in the table; here is the slot it should take.
    Absent(Placement<'w>),
}

// ---------------------------------------------------------------- what a slot holds

/// How many ids a [`CachedWord`] holds before it has to spill into
/// [`WordCache::token_ids_arena`]. Three is enough for most words in an alphabetic
/// script, and for far fewer of them in Chinese or Korean, where a word turns into
/// more ids.
const MAX_INLINE_IDS: usize = 3;

/// A sentinel value stored in [`CachedWord::word_bytes_or_hash`] when it holds the hash of a long word instead of the word itself.
const KEY_IS_HASH: u128 = 1 << 127;

/// A sentinel value stored in [`CachedWord::inline_id_count`] when the ids did not fit in
/// the slot and went to the arena instead.
const SPILLED: u8 = u8::MAX;

/// Bits per length in a spilled entry's third `word_ids` lane, which packs the
/// word's byte count above the id count. Eleven, because the two have to share one
/// `u32`, and [`MAX_WORD_BYTES`] is the most either of them can be: the cache turns
/// longer words away, and no model emits more than one id per byte.
const PACKED_LEN_BITS: u32 = 11;

const PACKED_LEN_MASK: u32 = (1 << PACKED_LEN_BITS) - 1;

const _: () = assert!(MAX_WORD_BYTES <= PACKED_LEN_MASK as usize);

/// One entry, in the three shapes the module docs draw out. In short:
///
/// - `word_bytes_or_hash` is the key: the word itself when it fits in 15 bytes,
///   otherwise its hash with [`KEY_IS_HASH`] set.
/// - `word_ids` is the token ids while `inline_id_count` counts them, and becomes
///   `[word_off, ids_off, packed lengths]` once that byte is [`SPILLED`], which is
///   what [`CachedWord::word_off`] and the readers beside it are for.
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

    fn word_off(&self) -> u32 {
        self.word_ids[0]
    }

    fn ids_off(&self) -> u32 {
        self.word_ids[1]
    }

    fn ids_len(&self) -> usize {
        (self.word_ids[2] & PACKED_LEN_MASK) as usize
    }

    fn word_len(&self) -> usize {
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
/// The 16 bytes are taken in one load and the surplus masked off, rather than
/// copied `len` bytes at a time: `len` is only known at run time, so the copy is a
/// call into `memcpy` and the branch on its length mispredicts on every short word.
/// The load reads past the word, so it is only taken when all 16 bytes fall in the
/// page the word already sits in.
fn pack_word(word: &[u8]) -> Option<u128> {
    let len = word.len();
    if len == 0 || len > 15 {
        return None;
    }
    let ptr = word.as_ptr();
    let raw = if ptr as usize & 0xFFF <= 0x1000 - 16 {
        // SAFETY: the read stays inside one page, and that page is mapped because
        // `word` is not empty. The bytes past `len` are masked off below.
        u128::from_le(unsafe { ptr.cast::<u128>().read_unaligned() })
    } else {
        let mut lanes = [0u8; 16];
        lanes[..len].copy_from_slice(word);
        u128::from_le_bytes(lanes)
    };
    Some((raw & (u128::MAX >> (8 * (16 - len)))) | ((len as u128) << 120))
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

// ---------------------------------------------------------------- the row of tags

/// How many slots [`WordCache::find_word_in_cache`] walks before it stops looking
/// and evicts: a word is in the sixteen slots from its home slot on, or nowhere.
///
/// Sixteen one-byte tags cross at most two cache lines, and a walk rarely reads
/// that many, since it stops at the first empty slot, usually a step or two along.
const WALK_WINDOW: usize = 16;

/// The tag of a slot that holds nothing. No live entry ever carries it
/// ([`make_tag`]), which is what lets a walk stop at the first one it reads,
/// and lets [`WordCache::insert`] tell a slot it has to clear from one it can write
/// straight into.
const EMPTY: u8 = 0;

/// How many bits of the hash a tag carries. All eight of them: a walk therefore
/// reads a slot it did not want about once in 255.
const TAG_BITS: u32 = 8;

/// The tag a hash gives its slot, which is never [`EMPTY`].
///
/// The top of the hash, because the bottom of it already chose the home slot: a
/// tag built from those bits would be the same for every slot the word can reach
/// and would tell a walk nothing.
///
/// Top bytes of 0 and 1 both come out as 1, since [`EMPTY`] is 0 and no live tag may
/// be that. A word with either of those top bytes therefore shares its tag with one
/// other, and pays a wasted entry read twice as often as the rest. A live tag of
/// [`EMPTY`] would cost far more: a walk would stop at that slot, losing every entry
/// stored after it in the window, and the insert that overwrote it would hand its
/// arena runs back to nobody.
fn make_tag(hash: u64) -> u8 {
    ((hash >> (u64::BITS - TAG_BITS)) as u8).max(EMPTY + 1)
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

    /// The slot a word is in, or `None` if the table does not hold it.
    fn slot_of(cache: &WordCache, word: &[u8]) -> Option<usize> {
        let (key, hash) = cache.make_word_key(word);
        match cache.find_word_in_cache(key, hash, word) {
            Walk::Found(index) => Some(index),
            Walk::Absent(_) => None,
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

    /// The wide load and the copy it falls back to near a page boundary have to
    /// agree, or the same word keys differently depending on where it landed.
    #[test]
    fn a_packed_key_is_the_words_bytes_whichever_read_took_it() {
        let naive = |word: &[u8]| {
            let mut lanes = [0u8; 16];
            lanes[..word.len()].copy_from_slice(word);
            Some(u128::from_le_bytes(lanes) | ((word.len() as u128) << 120))
        };
        let buffer = vec![b'z'; 3 * 0x1000];
        for len in 1..=15usize {
            let word: Vec<u8> = (0..len as u8).map(|i| i.wrapping_add(1)).collect();
            assert_eq!(pack_word(&word), naive(&word));

            // the same word placed so that reading 16 bytes would cross into the
            // next page, which is the case the wide load has to decline
            let start = 0x1000 - (buffer.as_ptr() as usize & 0xFFF) + 0x1000 - len;
            let at = &mut buffer.clone()[start..start + len];
            at.copy_from_slice(&word);
            assert_eq!(pack_word(at), naive(&word));
        }
    }

    /// A live entry's tag has to be something [`EMPTY`] is not, or a walk reads an
    /// occupied slot as the end of the chain, stores on top of it, and drops the
    /// arena runs it was using on the floor. Every tag is a byte of hash now, so
    /// the hashes whose top byte is zero are the ones that have to be caught.
    #[test]
    fn a_live_tag_is_never_the_empty_marker() {
        for hash in [0u64, 1, u64::MAX, 1 << 57, u64::MAX >> TAG_BITS] {
            assert_ne!(make_tag(hash), EMPTY, "hash {hash:#x}");
        }
        // Any other top byte is a tag in its own right, kept as it is.
        assert_eq!(make_tag(0xA7 << (u64::BITS - TAG_BITS)), 0xA7);
    }

    /// A short word's placement comes out of its packed key rather than its
    /// bytes, which leaves the hash doing all of the mixing. Words that differ in
    /// one byte pack to keys that differ in one byte, so an index taken from those
    /// bits as they are (`packed as u64`) drops most of these on one slot.
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
    /// encodes to one id per byte, is the largest either of them can get. If they
    /// ever overlapped, a hit would read the wrong lengths and hand back the wrong
    /// ids.
    #[test]
    fn a_word_at_the_length_limit_round_trips() {
        let mut cache = WordCache::new(1 << 8);
        let word = vec![b'q'; MAX_WORD_BYTES];
        let ids: Vec<u32> = (0..MAX_WORD_BYTES as u32).collect();
        store(&mut cache, &word, ids.clone().into_iter());
        assert_eq!(cache.lookup(&word).hit(), Some(&ids[..]));
    }

    /// A tag is one byte, so one slot in 255 answers for a word that is not in
    /// it. That is too rare to wait for, so forge one: put a word's tag on a slot
    /// holding a different key, and demand the walk look past it rather than
    /// treat the tag as the answer.
    #[test]
    fn a_tag_collision_is_confirmed_against_the_key() {
        let mut cache = WordCache::new(WALK_WINDOW);
        let (_, hash) = cache.make_word_key(b"beta");
        let decoy = hash as usize & cache.index_mask;
        cache.tags[decoy] = make_tag(hash);
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
    /// word's key and tag on it, and demand the walk look past it.
    #[test]
    fn a_hashed_key_is_confirmed_against_the_word_bytes() {
        let mut cache = WordCache::new(1 << 8);
        let mine = vec![b'a'; 40];
        let theirs = vec![b'b'; 40];

        store(&mut cache, &theirs, [7u32].into_iter());
        let their_index = slot_of(&cache, &theirs).unwrap();
        let their_slot = cache.cached_words[their_index];
        let (my_key, my_hash) = cache.make_word_key(&mine);
        let my_home = my_hash as usize & cache.index_mask;
        cache.cached_words[my_home] = CachedWord {
            word_bytes_or_hash: my_key,
            ..their_slot
        };
        cache.tags[my_home] = make_tag(my_hash);

        store(&mut cache, &mine, [1u32, 2].into_iter());
        assert_eq!(cache.lookup(&mine).hit(), Some(&[1u32, 2][..]));
    }

    /// A word that hashes into a full window is stored rather than turned away,
    /// and the entry evicted is the one in its home slot, even when that entry has
    /// just been used, which is the whole of what this policy costs.
    #[test]
    fn a_full_window_evicts_its_home_slot() {
        let mut cache = WordCache::new(WALK_WINDOW);
        let words: Vec<Vec<u8>> = (0..WALK_WINDOW as u8).map(|i| vec![i; 4]).collect();
        for (i, word) in words.iter().enumerate() {
            store(&mut cache, word, [i as u32].into_iter());
        }
        let (_, hash) = cache.make_word_key(b"newcomer");
        let home = hash as usize & cache.index_mask;
        let evicted = words
            .iter()
            .position(|word| slot_of(&cache, word) == Some(home))
            .expect("the table is full, so some word holds that slot");
        for _ in 0..8 {
            assert!(cache.lookup(&words[evicted]).hit().is_some());
        }

        store(&mut cache, b"newcomer", [999u32].into_iter());
        assert_eq!(cache.lookup(b"newcomer").hit(), Some(&[999u32][..]));
        assert_eq!(cache.lookup(&words[evicted]).hit(), None);
        for (i, word) in words.iter().enumerate() {
            if i != evicted {
                assert_eq!(
                    cache.lookup(word).hit(),
                    Some(&[i as u32][..]),
                    "entry {i} was dropped as well"
                );
            }
        }
    }

    /// A window that runs off the end of the table carries on at the start, and a
    /// word stored past that seam has to be found there. Nothing mirrors the first
    /// tags at the end of the row. Every step of the walk is folded back into the
    /// table instead.
    #[test]
    fn a_word_stored_past_the_end_of_the_table_is_found() {
        let mut cache = WordCache::new(WALK_WINDOW);
        let last_slot = cache.cached_words.len() - 1;
        let homed_on_the_last_slot: Vec<Vec<u8>> = (0..4000u32)
            .map(|i| format!("w{i}").into_bytes())
            .filter(|word| {
                let (_, hash) = cache.make_word_key(word);
                hash as usize & cache.index_mask == last_slot
            })
            .take(3)
            .collect();
        assert_eq!(homed_on_the_last_slot.len(), 3);

        for (i, word) in homed_on_the_last_slot.iter().enumerate() {
            store(&mut cache, word, [i as u32].into_iter());
        }
        for (i, word) in homed_on_the_last_slot.iter().enumerate() {
            assert_eq!(cache.lookup(word).hit(), Some(&[i as u32][..]), "{word:?}");
        }
        // The first one took the last slot, so the other two could only go past it.
        assert!(slot_of(&cache, &homed_on_the_last_slot[1]).unwrap() < last_slot);
    }

    /// Reusing an evicted entry's arena run is where this design can go wrong:
    /// hand one run to two live entries and a hit starts returning another word's
    /// ids. Churn a table far too small for the input and demand the invariant
    /// that matters: an entry may be evicted, but a hit is never wrong.
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
        assert!(
            live > 0,
            "everything was evicted, so the test proves nothing"
        );
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
            "inserts stopped landing: the arenas ran out"
        );
        assert!(cache.word_bytes_arena.data.len() <= 65 * 35);
        assert!(cache.token_ids_arena.data.len() <= 65 * 8);
    }
}
