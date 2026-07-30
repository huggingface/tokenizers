//! A table that remembers which token ids a word encodes to, so a model only has
//! to work it out once.
//!
//! # What this is for
//!
//! A tokenizer turns text into *token ids*: the small integers a language model
//! reads. It gets there in three stages. It **normalizes** the text (lowercasing
//! it, say), it **pre-tokenizes** it — cuts it into small pieces, which are
//! usually words or fragments of words — and then it runs a **model** over each
//! piece to produce ids. This module calls one of those pieces a *word*, because
//! that is what it nearly always is.
//!
//! The model stage is the expensive one, and it pays that cost for every word:
//!
//! - **BPE** (byte-pair encoding) starts from the word's individual bytes and
//!   repeatedly glues the best-ranked neighbouring pair together, until no pair
//!   left in the word can be merged. Every merge changes the word and the search
//!   for the next best pair starts again.
//! - **Unigram** considers the many ways of cutting the word into pieces the
//!   vocabulary knows, scores them, and keeps the best. That is a search over a
//!   lattice of candidate cuts.
//! - **WordPiece** walks the word from the front, taking the longest piece the
//!   vocabulary holds, then carries on from where that piece ended. That is one
//!   search per piece of the word.
//!
//! Both are pure: the same word and the same vocabulary always give the same ids.
//! And words repeat, relentlessly — a few thousand distinct words cover nearly
//! every word of a document, and a few dozen cover a large share of it on their
//! own. So the second time a word shows up, its ids should come out of a table
//! rather than out of the model. Storing the answer to a pure function and
//! reusing it is called *memoization*, and this module is where the answers go:
//!
//! ```text
//!            "the cat sat on the mat"
//!                       │
//!                       │  normalize, then pre-tokenize
//!                       ▼
//!         "the"   "cat"   "sat"   "on"   "the"   "mat"
//!            │       │       │      │       │      │
//!            ▼       ▼       ▼      ▼       ▼      ▼
//!         ┌────────────────────────────────────────────┐
//!         │  WordCache — have I encoded this word yet? │
//!         └────────────────────────────────────────────┘
//!            │ hit                         │ miss
//!            │ (the ids, from the table)   ▼
//!            │                     the model works it out,
//!            │                     and the ids are stored here
//!            ▼                             ▼
//!          [12]    [87]    [43]    [9]    [12]   [64]   ← ids, made up here
//!                                          ▲
//!                                          └ second "the": a hit
//! ```
//!
//! The payoff is large. On `big.txt` in 4 KB chunks, one thread encoding with
//! GPT-2's tokenizer goes from 27 MB/s to 230 MB/s with this table in place.
//! Llama-3 gains less — 78 to 116 MB/s — because its config sets `ignore_merges`,
//! which looks the whole word up in the vocabulary before any merging happens, so
//! the work a hit saves there is a vocabulary lookup rather than a merge loop.
//!
//! ## What "cache" buys, and what it costs
//!
//! A cache is not a complete record. It has a fixed memory budget, it is allowed
//! to forget a word (the model can always encode it again), and it is *not*
//! allowed to be wrong: a hit must return the ids that word really encodes to,
//! never another word's. Those three sentences decide most of what follows.
//!
//! Both sides of the table sit on the hot path, so both have to be cheap:
//!
//! - a **hit** has to be far cheaper than encoding the word, or the cache is
//!   just overhead;
//! - a **miss** has to cost close to nothing, because every word pays for one the
//!   first time it is seen, and the set of distinct words in a document has no
//!   upper bound — Korean and Chinese text keeps producing words nobody has seen
//!   before.
//!
//! One `WordCache` belongs to one encoding scratch buffer, which belongs to one
//! thread at a time. Nothing here is shared between threads, so there are no
//! locks, no atomics, and no memory ordering to reason about.
//!
//! ## Words to know
//!
//! The design borrows its words from hash tables and from CPU caches. Everything
//! below uses these ones and no synonyms:
//!
//! | word | meaning |
//! |---|---|
//! | **slot** | one fixed-size place in the table, holding one word's entry |
//! | **home** | the slot a word's hash picks out; where a probe starts |
//! | **probe** | walking forward from home, slot by slot, looking for the word |
//! | **window** | the [`WINDOW`] slots from home onward — as far as a probe goes |
//! | **step** | how far along the window something is, `0` being home itself |
//! | **tag** | one byte per slot, a few bits of its word's hash, kept apart from the slot |
//! | **spill** | move part of an entry out of its slot, into an arena |
//! | **arena** | one flat buffer holding the parts that did not fit in slots |
//! | **counter** | how much an entry is being used; the lowest one is evicted first |
//! | **epoch** | the table's clock; used to fade counters without rewriting them |
//!
//! # The five pieces
//!
//! The table is `N` slots, `N` being the requested capacity rounded up to a power
//! of two. Beside the slots are two small arrays — a probe reads *those* rather
//! than the slots — and two arenas for what a slot is too small to hold. Sizes
//! below are for the default capacity of 65,536 slots
//! ([`crate::utils::cache::DEFAULT_CACHE_CAPACITY`]):
//!
//! ```text
//!   slots       [Slot;  N]     32 B each    one word's entry per slot     2 MiB
//!   tags        [u8;    N+W]    1 B each    what a probe reads           64 KiB
//!   usage       [Usage; N+W]    2 B each    what decides who gives way  128 KiB
//!   key_arena   Arena<u8>                   bytes of words over 15 B    grows to
//!   id_arena    Arena<u32>                  id runs longer than 3       the live set
//!
//!   W = WINDOW = 16. The two arrays are N+W long, not N: see "the seam" below.
//! ```
//!
//! Every byte of every entry lives in one of those five. There are no per-entry
//! heap allocations, which is what makes a miss cheap — storing a word writes
//! into space that already exists.
//!
//! # Looking a word up
//!
//! Hash the word, and the low bits of the hash pick its home slot. The table is a
//! power of two, so "fold the hash into a slot number" is one bitwise `and` with
//! `N - 1` ([`WordCache::index_mask`]) rather than a division.
//!
//! Two words can easily want the same home. When that happens the loser moves
//! along: a probe checks home, then home + 1, then home + 2, up to [`WINDOW`]
//! slots. This is *open addressing with linear probing* — "open" because a word
//! is not confined to one place, "linear" because it walks one slot at a time.
//!
//! ```text
//!   lookup(b"tion")    home = hash(b"tion") & (N - 1) = 4
//!
//!     slot 3  │ "port"  │
//!     slot 4  │ "ation" │ ← home, but another word got here first: step on
//!     slot 5  │ "tion"  │ ← the key matches: hit, hand back its ids
//!     slot 6  │  empty  │
//!     slot 7  │ "the"   │
//!
//!   lookup(b"xyz")     home = 4 → not "ation", not "tion", slot 6 is empty → miss
//! ```
//!
//! A probe may stop at an empty slot rather than walking the full window. Nothing
//! here ever empties a slot again once it has been used — eviction overwrites, it
//! does not delete — so an empty slot proves that no word was ever stored past it.
//! That one property saves this table more than the early exit; the section on
//! eviction below spells out the rest.
//!
//! ## A miss hands back the slot to use
//!
//! A miss returns a [`Placement`]: the slot the word *would* go in — slot 6
//! above — which the caller passes to [`WordCache::insert`] once the model has
//! encoded the word. Choosing that slot needs exactly the reads that failing to
//! find the word has already done, so it comes for free here, whereas doing it
//! inside `insert` would mean hashing the word and reading the window a second
//! time.
//!
//! The caller may also drop the placement and store nothing. Nothing has changed
//! yet: an entry only loses its slot when something is written over it.
//!
//! ## A probe does not read the slots
//!
//! Done slot by slot, a probe asks one question of each slot it passes: *is this my
//! word?* Reading a slot to ask it drags 32 bytes through the CPU's caches for an
//! answer that is nearly always "no". A whole window of them is 512 bytes, spread
//! over eight cache lines.
//!
//! So the question is asked somewhere else. Every slot has a one-byte **tag** in
//! an array of its own, carrying [`TAG_BITS`] bits of its word's hash. A window's
//! worth of tags is sixteen bytes — which is exactly one `u128`:
//!
//! ```text
//!   tags   ┌──┬──┬──┬──┬──┬──┬──┬──┬─ … ─┬──┐   16 tags = 16 bytes = one u128
//!          │A3│00│F1│A3│5C│00│00│B8│     │2E│
//!          └──┴──┴──┴──┴──┴──┴──┴──┴─────┴──┘
//!     slot   4  5  6  7  8  9 10 11       19
//!     step   0  1  2  3  4  5  6  7       15
//!
//!          00 marks a slot nothing was ever stored in.
//! ```
//!
//! Read as one integer, that window answers a question about all sixteen slots at
//! once, using nothing but ordinary arithmetic — see [`tag_matches`] and
//! [`empty_slots`]. A slot is read only after its tag has said the word might be in
//! it, and a tag holds seven bits of hash: a word the table does not hold reads no
//! slot at all, about 127 times in 128.
//!
//! This is the [Swiss Tables] design, from Google's Abseil, which Rust's own
//! `HashMap` also uses through [hashbrown]: one control byte per slot, holding one
//! flag bit and seven bits of hash, tested sixteen at a time.
//!
//! ### Sixteen answers out of one subtraction
//!
//! Abseil and hashbrown test their sixteen bytes with SIMD instructions — *Single
//! Instruction, Multiple Data*, the CPU's vector unit, which applies one operation
//! to sixteen lanes of a register at once. There is no SIMD in this module. It
//! uses SWAR instead — *SIMD Within A Register* — where a plain integer is treated
//! as a row of lanes and ordinary arithmetic does the work.
//!
//! The trick is the classic hunt for a zero byte. Take the window, `xor` it with the
//! tag being looked for, and a lane goes to zero exactly where the tag matches.
//! Then subtract one from every lane — that is [`ONES`], `01` in all sixteen
//! lanes — because only a zero lane has to borrow, and borrowing flips its top bit
//! on. Finally keep nothing but those top bits, which is [`HIGHS`]. Worked through
//! four lanes, looking for tag `A3`:
//!
//! ```text
//!     step              0     1     2     3
//!     tags             A3    81    00    A3
//!     x = tags ^ A3    00    22    A3    00    ← zero exactly where the tag matches
//!     x - ONES         FF    20    A2    FF    ← the zero lanes borrowed: top bit on
//!     & !x             FF    00    00    FF    ← drops lanes that were already ≥ 80
//!     & HIGHS          80    00    00    80    ← "maybe" at steps 0 and 3
//! ```
//!
//! The `& !x` step earns its place: a lane whose `xor` came out `80` or more already
//! has its top bit set without borrowing anything, and `!x` throws exactly those
//! away. The whole test is `(x - ONES) & !x & HIGHS`, three instructions on one
//! register. It is the expression Alan Mycroft posted to a newsgroup in 1987 (see
//! [Bit Twiddling Hacks]), and the one hashbrown itself falls back to on targets
//! without SIMD.
//!
//! Two consequences of doing it this way are worth knowing:
//!
//! - **A borrow can cross into the lane above.** The subtraction is one operation on
//!   one register, so a lane holding exactly `tag ^ 1` directly above a match is
//!   answered "maybe" as well, and so is a run of them. That costs one slot read
//!   each, which the key comparison then rejects. It can never go the other way — a
//!   tag that is really there is never missed — and that direction is the one that
//!   would be a bug, so the `tag_matches_never_misses_a_tag` test pins it.
//! - **No vector types, nothing target-specific.** Sixteen bytes is one `u128` on
//!   every machine this builds for. Hand-written SIMD was tried: it came out at
//!   roughly a third of the instructions on the two targets it could be written
//!   for, against doubling the size of this module and keeping two
//!   implementations honest with each other. Not worth it here, where the probe
//!   is waiting on memory rather than on arithmetic.
//!
//! ### Why the *high* bits of the hash
//!
//! The low bits of the hash already chose the home slot. Two words that share a
//! home share those bits, so tags built from them would be equal across the whole
//! window and would tell a probe nothing. The high bits are the ones still free to
//! differ — see [`slot_tag`].
//!
//! ### Why usage is a third array
//!
//! What a tag deliberately does *not* carry is how much its entry is being used.
//! Only two paths need that: a hit, which has already found its slot, and a full
//! window, which has to give one up. Keeping it one array further out means a
//! probe that finds nothing never touches it. Widening the tags to hold it would
//! instead double the bytes every probe reads.
//!
//! ### The seam
//!
//! A window starting at slot `N - 3` runs off the end of the table and wraps
//! around to slot 0. Reading sixteen tags "in a row" would then need two reads and
//! a join. Instead both arrays are `N + WINDOW` long, and the last [`WINDOW`]
//! entries mirror the first [`WINDOW`]. Every window is contiguous, and every
//! write keeps the mirror in step ([`WordCache::write_tag`],
//! [`WordCache::write_usage`]). The slot *index* still wraps with `index_mask`;
//! only the reads of tags and usage use the mirror.
//!
//! # What a slot holds
//!
//! A slot is 32 bytes: a 16-byte `key`, three `u32`s of `payload`, and one byte
//! saying how many of those payload words are token ids. It takes one of three
//! shapes, depending on how long the word is and how many ids it encoded to. The
//! first shape is the common case for English or code — a short word encoding to
//! one or two tokens — and it touches nothing outside the slot:
//!
//! ```text
//!  (1) at most 15 bytes, at most 3 ids — the slot holds everything
//!
//!     b"tion" → [5378, 262]
//!    key             │ t  i  o  n  00 … 00 │ 4 │   the word, length in the top byte
//!    payload         │ 5378 │ 262 │  ·  │          the ids, right here
//!    inline_id_count │ 2 │                         2 of the 3 payload words are ids
//!
//!  (2) at most 15 bytes, more than 3 ids — the ids move to an arena
//!
//!     b"电脑" (6 bytes) → 8 ids
//!    key             │ e7 94 b5 e8 84 91 … │ 6 │   still the word itself
//!    payload         │  ·  │  40  │ 0:8 │          8 ids, in the id arena at 40
//!    inline_id_count │ SPILLED │                   payload is coordinates, not ids
//!
//!  (3) over 15 bytes — the bytes move too, and the key becomes a hash
//!
//!     b"unbelievabilities" (17 bytes) → 6 ids
//!    key             │ hash(word) │ KEY_IS_HASH │  no room for 17 bytes
//!    payload         │  12  │  40  │ 17:6 │        17 bytes at 12, 6 ids at 40
//!    inline_id_count │ SPILLED │
//! ```
//!
//! Shapes (1) and (2) carry the word itself in the key, so an equal key *is* an
//! equal word: one register-wide comparison, no memory to chase. Packing the
//! length into the top byte is what keeps `"a"` and `"a\0"` apart, and it is also
//! what lets shape (3) be told apart from the others — a hashed key sets
//! [`KEY_IS_HASH`], the top bit, which no length in `1..=15` can reach. Shape (3)
//! is therefore the one case where a matching key is not proof: two long words can
//! hash alike, so a hit there also compares the bytes in the key arena.
//!
//! Both spilled shapes have two lengths to record — how many bytes are in the key
//! arena, and how many ids are in the id arena — and both are bounded by
//! [`MAX_WORD_BYTES`], so [`PACKED_LEN_BITS`] bits each is enough and the two
//! share `payload[2]`.
//!
//! [gigatoken]'s pre-token cache is built from the same two parts — a word of up to
//! 15 bytes packed into a `u128` with its length in the top byte, and a
//! self-contained 32-byte entry with the ids inline. We replicated it to measure
//! against this table while designing it, and it is also where the reasoning
//! diverges: its table holds over a million pre-tokens, far more than any CPU cache
//! can, so it spends its effort on touching exactly one cache line per probe
//! (entries in line-aligned pairs, software prefetching, huge pages) and it never
//! evicts — it doubles in size instead.
//!
//! # How an entry earns its place
//!
//! Every entry carries a **counter** of how much its word is being used, bumped on
//! each hit. When a word arrives and all [`WINDOW`] slots of its window are taken,
//! the lowest counter loses its slot.
//!
//! A counter that only ever went up would measure the wrong thing. The first page
//! of a document can make `"the"` the most-used word in the table; fifty pages
//! later the text has moved on to another subject entirely, but `"the"` still holds
//! the highest count and nothing can ever catch up with it. What should decide an
//! eviction is how much a word is being used *lately*. So counters have to fade.
//!
//! ## Fading without writing anything
//!
//! Fading them by hand is the expensive way: every eviction would rewrite all
//! sixteen counters of a window. So nothing is rewritten. The table keeps a single
//! number — the **epoch** — and bumps it every so often. Each [`Usage`] records
//! the epoch its counter was written in, and a counter is *read* as
//!
//! ```text
//!   score = counter >> (epoch - written_in)     halve it once per epoch missed
//! ```
//!
//! No stored number changes. What changes is what a stored number is worth:
//!
//! ```text
//!   the table is at epoch 9
//!
//!     A   counter 32, written in 9  →  0 epochs missed  →  score 32
//!     B   counter 32, written in 5  →  4 epochs missed  →  score 32 >> 4 = 2
//!                                                              ▲ B was hot once
//!                                                                and has coasted
//! ```
//!
//! A counter is one byte, so [`MAX_HALVINGS`] missed epochs take any entry to
//! zero, however hot it once was. Whenever an entry *is* written — a hit, or a
//! fresh insert — its outstanding fade is applied first: the score becomes the new
//! counter, stamped with the current epoch.
//!
//! Redis solves the same problem the same way: an eight-bit counter per key plus a
//! decay period, so a key nobody reads any more loses its count without anything
//! having to sweep the keyspace. Caffeine, the reference implementation of TinyLFU,
//! takes the other route and physically halves every counter in its sketch once a
//! sample of accesses has gone by. Both are linked at the end.
//!
//! ## How often the epoch moves
//!
//! One bump fades every counter in the table at once, and the table is far bigger
//! than a window. Bumping on every eviction would therefore fade all of it at the
//! speed of whichever window happens to be churning hardest: counters would sit at
//! zero and every eviction would be an arbitrary pick.
//!
//! The rate that matches is one bump per `N / WINDOW` evictions
//! ([`WordCache::evictions_per_epoch`]). A slot can be reached from [`WINDOW`]
//! different homes, so it sits in [`WINDOW`] of the table's `N` windows, and an
//! eviction somewhere in the table lands in one of them with probability
//! `WINDOW / N`. Bumping at that rate gives each window, on average, one fade per
//! eviction of its own — which is what fading a window by hand would have done.
//!
//! ## Starting the count again
//!
//! An epoch is a byte, so after [`LAST_EPOCH`] bumps the count runs out of numbers
//! and has to start over at zero. The moment it does, every epoch written before
//! the restart is a *larger* number than the current one, and the difference
//! between them stops meaning anything: an entry idle for the whole lap would read
//! as freshly written, at full strength, and could never be evicted again. That is
//! dead weight forever — the exact failure fading exists to prevent.
//!
//! So the last bump of a lap does not restart the count on its own. It first walks
//! `usage`, writing each entry's current score into its counter, and only then puts
//! the epoch and every [`Usage`] back to zero together — see
//! [`WordCache::restart_epochs`]. No entry moves up or down against another; the
//! differences are simply cashed in while they can still be read. That is one pass
//! over `usage` every 256 bumps, which at the default capacity is once every
//! million evictions.
//!
//! # Where an evicted entry's space goes
//!
//! Whatever an evicted entry held in the arenas is handed back when it is
//! overwritten, and the next word of the same shape takes it. Runs are never moved
//! and never rounded up to a bigger size, because there is a free list for every
//! length a run can have — which is only possible because [`MAX_WORD_BYTES`] makes
//! that a finite list of lengths (see [`Arena`]):
//!
//! ```text
//!   the id arena, just after A — which had 6 ids at offset 12 — was evicted
//!
//!     data  ┌─────┬────────────────────────────┬─────┐
//!           │  …  │ A's 6 ids, at offset 12    │  …  │   nothing is moved
//!           └─────┴────────────────────────────┴─────┘
//!
//!     free  len 6 → [ 12 ]   the next word with 6 ids is handed offset 12
//!           len 8 → [    ]   one list per length, 0 … MAX_WORD_BYTES
//! ```
//!
//! So the arenas hold the live set rather than every word ever seen. Without reuse
//! they could only grow, and reclaiming them would mean copying every live entry
//! into a second buffer — twice the memory for the same number of entries.
//!
//! Overwriting the victim where it lies is also what keeps the table itself small.
//! Removing an entry from an open-addressed table normally needs a *tombstone* — a
//! marker saying "something was here, keep walking" — or a backward shift of the
//! entries after it, because a probe stops at the first empty slot and a hole in
//! the middle of a chain would cut it short. Here nothing is ever removed: a slot
//! goes from empty to occupied once, and after that it only ever changes owner.
//! That is what lets a probe stop at the first empty slot, and it is also why the
//! window has to be bounded — in a saturated table, that bound is the only thing
//! stopping a probe from walking the whole table.
//!
//! # The dials, and what each one decides
//!
//! | constant | value | what it decides |
//! |---|---|---|
//! | [`WINDOW`] | 16 | slots a probe covers; also how many tags fit in one `u128` |
//! | [`TAG_BITS`] | 7 | hash bits per tag; a probe reads a slot it did not want once in 128 |
//! | [`INLINE_IDS`] | 3 | ids that fit beside the key, so a hit reads only the slot |
//! | [`MAX_WORD_BYTES`] | 1024 | longest word stored; bounds every arena run |
//! | [`MAX_HALVINGS`] | 8 | halvings that take a one-byte counter to zero |
//! | [`LAST_EPOCH`] | 255 | bumps in one lap, after which the count restarts |
//! | [`INITIAL_COUNTER`] | 1 | what a freshly stored entry's counter starts at |
//! | [`PACKED_LEN_BITS`] | 11 | bits per length in a spilled entry; 2047 ≥ `MAX_WORD_BYTES` |
//! | capacity | 65,536 | slots, passed in by the caller; the arenas are sized from it |
//!
//! Every other constant in this file is a marker rather than a dial — [`EMPTY`],
//! [`OCCUPIED`], [`SPILLED`], [`KEY_IS_HASH`], [`ONES`], [`HIGHS`] — and each one's
//! value is fixed by what it has to stay distinguishable from. Their own docs say
//! from what.
//!
//! Capacity is the one dial with a real cost, and it is paid in memory: 2.2 MiB of
//! arrays per cache at 65,536 slots, plus arenas, once per thread that encodes.
//! Every doubling up to that point still bought throughput on the last sweep
//! (gpt2 over `big.txt`, 256-byte chunks, 16 threads: 389 MB/s at 1,024 slots, 594
//! at 8,192, 903 at 65,536), so shrinking the table is not a cheap way to save
//! memory.
//!
//! # Designs measured and not kept
//!
//! ## Fixed buckets instead of open addressing
//!
//! The obvious alternative is what CPU caches do — *4-way set associativity*: cut
//! the table into buckets of four slots, and let a word live only in the one bucket
//! its hash picks. That is a single load per lookup and easy to reason about. But a
//! word whose four slots are taken can never be cached at all, however empty the
//! rest of the table is:
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
//!   open addressing — the word moves along to the next free slot
//!
//!     │●│●│●│●│·│·│·│·│·│·│·│·│·│·│·│·│
//!      ▲       ▲
//!      home    └ stored here instead; only WINDOW slots taken in a
//!                row can turn a word away
//! ```
//!
//! With buckets that small, being locked out starts happening long before the
//! table is genuinely full. On our multilingual fixtures it cost between 0.1% and
//! 4% of all lookups, worst on Korean and Chinese, where a document holds far more
//! distinct words. Probing a window closed that gap: those misses went to ~0, and
//! in the encoder (`examples/fixture_bench.rs`) the many-distinct-word scripts
//! spent 10-40% less time in the model stage. English and code came out inside the
//! noise — and the noise there is wide. Running one binary against *itself* swings
//! the model stage by ±30%, so measure that floor before believing anything
//! smaller.
//!
//! ## A `HashMap`
//!
//! The comparison is concrete, because the crate still contains two of them: the
//! legacy BPE model in `models/bpe/model.rs` caches in a thread-local
//! `AHashMap<String, Word>` (`BPE_LOCAL_CACHE`), and [`crate::utils::cache::Cache`]
//! wraps another one in an `RwLock` for the legacy Unigram model.
//!
//! On a warm benchmark, where every word is already in the map, a hash map is not
//! far off this table — it has no bucket conflicts either. What it cannot do is
//! the rest of the job:
//!
//! - **Every word it accepts goes to the allocator.** The map owns its keys, so an
//!   accepted word is copied into a fresh `String`, and each value keeps a heap
//!   buffer of its own alive. That cost lands on misses, which is where a cache can
//!   least afford it: a word misses the first time it is ever seen. An insert here
//!   writes into space that already exists.
//! - **A hit is two more random loads.** The key bytes and the value each live in
//!   their own heap block, elsewhere in memory from the table that pointed at them.
//!   A slot here answers the common word out of the 32 bytes the probe has already
//!   decided to read.
//! - **It cannot make room.** A `HashMap` grows until something stops it, and the
//!   only thing it can do when stopped is refuse: the legacy cache inserts
//!   `while local.len() < capacity`, so the words from the first pages of text keep
//!   their places forever, however useless they turn out to be. Choosing *which*
//!   entry to drop needs evidence about how much each one is used, and somewhere to
//!   keep it — the counter in every [`Usage`].
//! - **Entries cost more than twice the memory.** 24 bytes for the `String` and 24
//!   for the value's vector, inside the table, before the two heap blocks they point
//!   at — against 35 bytes here for the common word (32 of slot, 1 of tag, 2 of
//!   usage).
//!
//! Sharing one map between threads costs more again: `Cache` has to `try_read` /
//! `try_write` and silently drop the read or the write whenever another thread
//! holds the lock.
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
//!   which is the same shape as [`Usage`].
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

/// Longest word the cache will store, in bytes.
///
/// Long words are rare and rarely repeat, so storing them buys little, and the
/// lookup for one is a hash over every byte to learn nothing. Sweeps over 64,
/// 256 and 1024 came out flat, so the exact value does not matter for speed —
/// what the cap is really for is the pathological input. A tokenizer with no
/// pre-tokenizer hands the model whole documents as single "words", and uncapped
/// the cache would cheerfully memoize documents. 1024 sits far above every
/// pre-token we have measured (median 3-4 bytes on code, 5 on English, 14 on
/// Korean) and far below a document.
///
/// It is also what makes the arenas' free lists possible — one list per length —
/// and what bounds an id run: a token covers at least one byte of its word, so a
/// word this long cannot encode to more than this many ids.
const MAX_WORD_BYTES: usize = 1024;
/// Slots a probe covers, counting from the word's home.
///
/// The probe has to stop somewhere: nothing is ever removed from this table, so
/// in a saturated one a probe looking for an empty slot would otherwise scan
/// forever. Linear probing normally settles in one or two slots — mean probe
/// length measured at 1.27 — and sixteen is a bound, not a typical cost.
///
/// Sixteen is also how many bytes a `u128` holds, so a window's worth of tags is
/// one integer, so a window is asked a question in one go rather than sixteen —
/// see [`tag_matches`].
const WINDOW: usize = 16;
/// Ids carried in the slot itself — three `u32`s is what fits beside the key in
/// 32 bytes.
///
/// Enough for 80-96% of lookups on English or code, but only about a quarter of
/// them on Chinese or Korean, where a vocabulary holding none of those scripts
/// spends ten ids or more on a single word.
const INLINE_IDS: usize = 3;
/// Set in `key` when it holds the hash of a long word instead of the word
/// itself. It is the top bit, and a packed key carries its length (`1..=15`) in
/// the top byte, so a hashed key can never be mistaken for a packed one.
const KEY_IS_HASH: u128 = 1 << 127;

/// The id count in [`Slot::inline_id_count`] when the ids did not fit in the slot
/// and went to the arena instead.
const SPILLED: u8 = u8::MAX;
/// What a newly stored entry's counter starts at. One rather than zero, so that a
/// fresh entry outranks anything that has already faded to zero — otherwise a word
/// could lose its slot before it is ever looked up again. One hit then lifts it
/// above its untouched neighbours, which start here too.
const INITIAL_COUNTER: u8 = 1;
/// Highest epoch a [`Usage`] can hold, which makes a lap 256 bumps long. On the
/// last one, [`WordCache::restart_epochs`] starts the count again.
const LAST_EPOCH: u8 = u8::MAX;
/// Missing this many epochs takes any counter to zero, because a counter is a
/// byte and each missed epoch halves it.
const MAX_HALVINGS: u8 = 8;

/// The tag of a slot nothing has ever been stored in.
const EMPTY: u8 = 0;
/// Set in a tag beside the hash bits, so that a live entry's tag is never
/// [`EMPTY`] — which is also what makes [`empty_slots`] a single `and`.
const OCCUPIED: u8 = 0x80;
/// How many bits of the hash a tag carries. Seven, because the eighth is
/// [`OCCUPIED`]; a probe therefore reads a slot it did not want once in 128.
const TAG_BITS: u32 = 7;

/// Bits per length in a spilled entry's `payload[2]`, which packs the key's byte
/// count above the id count. Eleven, because both are bounded by
/// [`MAX_WORD_BYTES`] and two of them have to share one `u32`.
const PACKED_LEN_BITS: u32 = 11;
const PACKED_LEN_MASK: u32 = (1 << PACKED_LEN_BITS) - 1;
const _: () = assert!(MAX_WORD_BYTES <= PACKED_LEN_MASK as usize);

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

/// The tag a hash gives its slot.
///
/// The top of the hash, because the bottom of it already chose the home slot: a
/// tag built from those bits would be the same for every slot the word can reach
/// and would tell a probe nothing.
fn slot_tag(hash: u64) -> u8 {
    OCCUPIED | (hash >> (u64::BITS - TAG_BITS)) as u8
}

/// How much an entry is being used, and when that was last written down. One per
/// slot, in [`WordCache::usage`].
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
#[repr(C)]
struct Usage {
    /// How much the word is being used, as of `epoch`.
    counter: u8,
    /// The epoch `counter` was written in.
    epoch: u8,
}

impl Usage {
    /// What this entry's counter is worth at `epoch`: what was written down,
    /// halved once for every epoch that has gone by since.
    ///
    /// `self.epoch` is never ahead of `epoch` — both are reset together by
    /// [`WordCache::restart_epochs`] — so the subtraction cannot go below zero.
    fn score(&self, epoch: u8) -> u8 {
        // Widened because shifting a `u8` by MAX_HALVINGS is shifting it by its
        // own width, which Rust leaves undefined.
        (self.counter as u32 >> (epoch - self.epoch).min(MAX_HALVINGS)) as u8
    }
}

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
struct Slot {
    key: u128,
    payload: [u32; INLINE_IDS],
    /// How many of `payload`'s words are token ids, or [`SPILLED`].
    inline_id_count: u8,
}

const _: () = assert!(std::mem::size_of::<Slot>() == 32);

impl Slot {
    fn spilled(&self) -> bool {
        self.inline_id_count == SPILLED
    }

    fn key_off(&self) -> u32 {
        self.payload[0]
    }

    fn ids_off(&self) -> u32 {
        self.payload[1]
    }

    fn ids_len(&self) -> usize {
        (self.payload[2] & PACKED_LEN_MASK) as usize
    }

    fn key_len(&self) -> usize {
        (self.payload[2] >> PACKED_LEN_BITS) as usize
    }
}

/// Every byte of one of these is `0x01`, or `0x80`. Broadcasting a value over all
/// [`WINDOW`] lanes and testing every lane's top bit is what turns a window into
/// arithmetic — see [`StepMask`].
const ONES: u128 = u128::from_le_bytes([0x01; WINDOW]);
const HIGHS: u128 = u128::from_le_bytes([0x80; WINDOW]);

/// One window's worth of yes-or-no, as the top bit of each of a `u128`'s bytes:
/// byte `step` answers for the slot `step` places along from home.
///
/// A window is [`WINDOW`] tags, which is sixteen bytes, which is one `u128`. So
/// the questions a probe asks are answered for the whole window by ordinary
/// integer arithmetic, in about as many instructions as it takes to ask one of
/// them of one slot. No vector types, nothing target-specific: the same handful
/// of `and`s and `xor`s on every machine this builds for.
type StepMask = u128;

/// The first step a mask answers `true` for, or [`WINDOW`] when it answers `true`
/// for none.
///
/// A flagged step `s` has its bit at `8 * s + 7`, so the count of trailing zeros
/// divided by eight is the step — and an empty mask counts 128 zeros, which
/// divides to [`WINDOW`] and reads as "nothing".
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
fn read_window(tags: &[u8; WINDOW]) -> u128 {
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
    let x = window ^ u128::from_le_bytes([tag; WINDOW]);
    x.wrapping_sub(ONES) & !x & HIGHS
}

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
/// it there. Found by the probe that missed, so storing the word costs no second
/// hash and no second read of the window.
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

/// How a probe over a word's window ended: the slot the word is in, or the slot
/// it should take if the caller decides to store it.
enum Probe {
    Hit(usize),
    Miss(usize),
}

/// Word bytes to token ids. See the module docs for the design.
pub struct WordCache {
    hasher: RandomState,
    slots: Box<[Slot]>,
    /// One tag per slot, and then a copy of the first [`WINDOW`] of them, so that
    /// the window of any home is `WINDOW` bytes in a row and can be read as one
    /// `u128`. [`WordCache::write_tag`] keeps the copy in step.
    tags: Box<[u8]>,
    /// One [`Usage`] per slot, mirrored like `tags`. Read only once a tag has
    /// answered for its slot, or when a full window has to give one up, so it is
    /// kept out of `tags`: a probe that finds nothing never touches it.
    usage: Box<[Usage]>,
    /// The bytes of words too long to fit in a key.
    key_arena: Arena<u8>,
    /// The id runs too long to fit in a slot.
    id_arena: Arena<u32>,
    /// `slots.len() - 1`. The table is a power of two, so this folds a hash into
    /// a slot index.
    index_mask: usize,
    /// Where the fade count stands, from 0 to [`LAST_EPOCH`] and no higher. A
    /// [`Usage::epoch`] is what this held when that entry was last written, so
    /// the difference between them is how many fades it has sat through.
    /// [`WordCache::restart_epochs`] puts this and every `Usage` back to zero
    /// together when the count runs out of room.
    epoch: u8,
    /// Evictions still to come before the next `epoch` bump.
    evictions_left: u32,
    /// Evictions per `epoch` bump — see the module docs on how often the epoch
    /// moves.
    evictions_per_epoch: u32,
}

impl WordCache {
    pub fn new(capacity: usize) -> Self {
        let n_slots = capacity.next_power_of_two().max(WINDOW);
        let evictions_per_epoch = (n_slots / WINDOW) as u32;
        Self {
            hasher: RandomState::new(),
            slots: vec![Slot::default(); n_slots].into_boxed_slice(),
            tags: vec![EMPTY; n_slots + WINDOW].into_boxed_slice(),
            usage: vec![Usage::default(); n_slots + WINDOW].into_boxed_slice(),
            // Ceilings, not reservations: the arenas grow only as far as the live
            // set takes them. Deliberately generous, so that the slot table is
            // what runs out first — an arena that turns inserts away while slots
            // sit empty is a capacity limit in disguise, and tighter numbers here
            // measured exactly that (tens of thousands of words refused at 35%
            // occupancy). The worst case they have to cover is a vocabulary with
            // no CJK in it, which spends ~17 ids on a single Chinese word.
            key_arena: Arena::new(n_slots * 48),
            id_arena: Arena::new(n_slots * 16),
            index_mask: n_slots - 1,
            epoch: 0,
            evictions_left: evictions_per_epoch,
            evictions_per_epoch,
        }
    }

    /// The ids `word` encoded to last time, or the slot to put them in when the
    /// model has worked them out.
    ///
    /// Takes `&mut self` because a hit is also a vote for keeping the entry.
    pub fn lookup<'c, 'w>(&'c mut self, word: &'w [u8]) -> Lookup<'c, 'w> {
        if word.len() > MAX_WORD_BYTES {
            return Lookup::Miss(None);
        }
        let (key, hash) = self.slot_key(word);
        let index = match self.probe(key, hash, word) {
            Probe::Hit(index) => {
                let epoch = self.epoch;
                // Saturating, so hits past what a byte holds are not counted. An
                // entry that busy is in no danger of eviction anyway.
                let counter = self.usage[index].score(epoch).saturating_add(1);
                self.write_usage(index, Usage { counter, epoch });
                let slot = self.slots[index];
                return Lookup::Hit(if slot.spilled() {
                    self.id_arena.get(slot.ids_off(), slot.ids_len())
                } else {
                    &self.slots[index].payload[..slot.inline_id_count as usize]
                });
            }
            Probe::Miss(index) => index,
        };
        Lookup::Miss(Some(Placement {
            index,
            key,
            tag: slot_tag(hash),
            word,
        }))
    }

    /// Remember what the word encoded to, in the slot its [`WordCache::lookup`]
    /// picked out. Silently does nothing when an arena is full: a cache is free to
    /// forget, and the caller has the ids either way.
    pub fn insert(&mut self, at: Placement<'_>, ids: impl ExactSizeIterator<Item = u32>) {
        let Some(entry) = self.build_entry(at.key, at.word, ids) else {
            return;
        };
        // A slot that is not empty belongs to somebody else, so this is an
        // eviction: give the arena runs back and charge the table one round of
        // fading. Both wait until here, so a lookup whose caller decides not to
        // store anything after all costs the entry sitting there nothing.
        if self.tags[at.index] != EMPTY {
            self.reclaim(at.index);
            self.fade();
        }
        // Fading moves the epoch, so stamp the new entry after it.
        self.write_tag(at.index, at.tag);
        self.write_usage(
            at.index,
            Usage {
                counter: INITIAL_COUNTER,
                epoch: self.epoch,
            },
        );
        self.slots[at.index] = entry;
    }

    /// The value a slot stores in its `key`, and the hash that places it. A short
    /// word is its own key; a longer one keys on a tagged hash and has to be
    /// confirmed against `key_arena`, since two long words can hash alike.
    ///
    /// A packed key is hashed as one `u128`, not as the word's bytes it was built
    /// from. Hashing a slice makes aHash mix the length in and then branch on it
    /// to choose a read width; a `u128` is one fixed-width fold with nothing to
    /// decide. The key already carries the length, so nothing is lost.
    fn slot_key(&self, word: &[u8]) -> (u128, u64) {
        match pack_word(word) {
            Some(packed) => (packed, self.hasher.hash_one(packed)),
            None => {
                let hash = self.hasher.hash_one(word);
                ((hash as u128) | KEY_IS_HASH, hash)
            }
        }
    }

    /// The tags of the window starting at `home`. The mirrored tail of `tags` is
    /// what makes the slice contiguous for every `home`.
    fn tag_window(&self, home: usize) -> u128 {
        read_window(self.tags[home..home + WINDOW].try_into().unwrap())
    }

    /// The step of the window's lowest-scoring slot, ties going to the slot
    /// nearest home.
    ///
    /// Reads the window as one flat slice, which is what the mirrored tail of
    /// `usage` is for. Folding each step back into the table instead costs an
    /// `and` per slot and leaves nothing for the compiler to widen.
    fn coldest(&self, home: usize, epoch: u8) -> usize {
        let window = &self.usage[home..home + WINDOW];
        // `min_by_key` keeps the first of equal keys, which is what sends a tie
        // to the slot nearest home.
        (0..WINDOW)
            .min_by_key(|&step| window[step].score(epoch))
            .unwrap()
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
    fn probe(&self, key: u128, hash: u64, word: &[u8]) -> Probe {
        let home = hash as usize & self.index_mask;
        let window = self.tag_window(home);
        let empty = empty_slots(window);
        // Nothing past the first empty slot can hold the word: the probe that
        // stored it would have stopped there and taken that slot.
        let mut candidates = tag_matches(window, slot_tag(hash)) & steps_before(empty);
        // Only a hashed key can turn out to belong to a different word, and
        // whether this one is hashed is settled before the probe starts — the
        // caller built the key out of the word it is looking for.
        let hashed = key & KEY_IS_HASH != 0;
        while candidates != 0 {
            let step = first_step(candidates);
            let index = (home + step) & self.index_mask;
            let slot = &self.slots[index];
            if slot.key == key
                && (!hashed || self.key_arena.get(slot.key_off(), slot.key_len()) == word)
            {
                return Probe::Hit(index);
            }
            candidates = clear_step(candidates, step);
        }
        let step = if empty == 0 {
            self.coldest(home, self.epoch)
        } else {
            first_step(empty)
        };
        Probe::Miss((home + step) & self.index_mask)
    }

    /// The entry to store, with whatever does not fit in a slot copied into the
    /// arenas. `None` when an arena is full, so a rejected insert leaves the table
    /// and the arenas exactly as they were.
    fn build_entry(
        &mut self,
        key: u128,
        word: &[u8],
        ids: impl ExactSizeIterator<Item = u32>,
    ) -> Option<Slot> {
        let long = key & KEY_IS_HASH != 0;
        let ids_len = ids.len();
        if !long && ids_len <= INLINE_IDS {
            let mut payload = [0u32; INLINE_IDS];
            for (dst, id) in payload.iter_mut().zip(ids) {
                *dst = id;
            }
            return Some(Slot {
                key,
                payload,
                inline_id_count: ids_len as u8,
            });
        }
        // A short word stays in the key, which is a zero-length run here.
        let key_len = if long { word.len() } else { 0 };
        let key_off = self.key_arena.alloc(key_len)?;
        let Some(ids_off) = self.id_arena.alloc(ids_len) else {
            self.key_arena.release(key_off, key_len);
            return None;
        };
        self.key_arena.fill(key_off, key_len, word.iter().copied());
        self.id_arena.fill(ids_off, ids_len, ids);
        let lengths = (key_len as u32) << PACKED_LEN_BITS | ids_len as u32;
        Some(Slot {
            key,
            payload: [key_off, ids_off, lengths],
            inline_id_count: SPILLED,
        })
    }

    /// Write a slot's tag, and the copy of it in the mirrored tail when the slot
    /// is one of the first [`WINDOW`].
    fn write_tag(&mut self, index: usize, tag: u8) {
        self.tags[index] = tag;
        if index < WINDOW {
            self.tags[self.slots.len() + index] = tag;
        }
    }

    /// The same, for a slot's [`Usage`].
    fn write_usage(&mut self, index: usize, usage: Usage) {
        self.usage[index] = usage;
        if index < WINDOW {
            self.usage[self.slots.len() + index] = usage;
        }
    }

    /// One eviction's worth of fading: step the countdown, and move the epoch on
    /// when it runs out.
    fn fade(&mut self) {
        self.evictions_left -= 1;
        if self.evictions_left > 0 {
            return;
        }
        self.evictions_left = self.evictions_per_epoch;
        if self.epoch == LAST_EPOCH {
            self.restart_epochs();
        } else {
            self.epoch += 1;
        }
    }

    /// Spend every entry's outstanding fade and start the epoch count again: each
    /// live entry's score becomes its counter, and the epoch and every [`Usage`]
    /// go back to zero.
    ///
    /// Run on the last epoch a `Usage` can hold, and only then. The epoch has to
    /// start over at some point, and the moment it does, the difference between it
    /// and a stored epoch stops meaning anything for entries written before the
    /// restart. So the differences are cashed in first, while they can still be
    /// read.
    fn restart_epochs(&mut self) {
        let epoch = self.epoch;
        // Tags are untouched, and the mirrored tail of `usage` gets the same pass
        // as the front, so both stay in step without a second walk.
        for usage in self.usage.iter_mut() {
            usage.counter = usage.score(epoch);
            usage.epoch = 0;
        }
        self.epoch = 0;
    }

    /// Hand an overwritten entry's arena runs back, so the next entry can use
    /// them.
    ///
    /// `#[inline]` because this exists to give a step of `insert` a name, not to
    /// be called: left out of line it costs `insert` a call and the pointers it
    /// had already loaded.
    #[inline]
    fn reclaim(&mut self, index: usize) {
        let slot = self.slots[index];
        if !slot.spilled() {
            return;
        }
        self.key_arena.release(slot.key_off(), slot.key_len());
        self.id_arena.release(slot.ids_off(), slot.ids_len());
    }
}

/// Handles for `examples/word_cache_bench.rs`, which times the parts of the cache
/// the encoder never calls directly. Compiled out of every build that does not
/// ask for them.
#[cfg(feature = "bench-internals")]
impl WordCache {
    /// Just the key building at the front of a lookup: pack the word, hash it.
    pub fn bench_slot_key(&self, word: &[u8]) -> u64 {
        self.slot_key(word).1
    }

    /// One eviction's worth of fading, without an eviction.
    pub fn bench_fade(&mut self) {
        self.fade()
    }

    /// The whole-table pass that starts the epoch count again.
    pub fn bench_restart_epochs(&mut self) {
        self.restart_epochs()
    }

    /// How many slots hold an entry. Equal to the capacity once the table is
    /// saturated, which is the state the eviction measurements need.
    pub fn bench_occupancy(&self) -> usize {
        self.tags[..self.slots.len()]
            .iter()
            .filter(|&&tag| tag != EMPTY)
            .count()
    }
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
        let (key, hash) = cache.slot_key(word);
        match cache.probe(key, hash, word) {
            Probe::Hit(index) => Some(index),
            Probe::Miss(_) => None,
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
        for hash in [0u64, 1, u64::MAX, 1 << 57, u64::MAX >> TAG_BITS] {
            assert_ne!(slot_tag(hash), EMPTY, "hash {hash:#x}");
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
                let (_, hash) = cache.slot_key(format!("tok{i}").as_bytes());
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
        let mut cache = WordCache::new(WINDOW);
        let (_, hash) = cache.slot_key(b"beta");
        let decoy = hash as usize & cache.index_mask;
        cache.write_tag(decoy, slot_tag(hash));
        cache.write_usage(
            decoy,
            Usage {
                counter: INITIAL_COUNTER,
                epoch: 0,
            },
        );
        // Any key that is not beta's. A packed key always carries a length in its
        // top byte, so 1 cannot be one.
        cache.slots[decoy] = Slot {
            key: 1,
            payload: [7, 0, 0],
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
        let their_slot = cache.slots[their_index];
        let their_usage = cache.usage[their_index];
        let (my_key, my_hash) = cache.slot_key(&mine);
        let my_home = my_hash as usize & cache.index_mask;
        cache.slots[my_home] = Slot {
            key: my_key,
            ..their_slot
        };
        cache.write_tag(my_home, slot_tag(my_hash));
        cache.write_usage(my_home, their_usage);

        store(&mut cache, &mine, [1u32, 2].into_iter());
        assert_eq!(cache.lookup(&mine).hit(), Some(&[1u32, 2][..]));
    }

    /// A word that hashes into a full window takes the lowest-scoring slot rather
    /// than being turned away, and the words that were actually used keep their
    /// place.
    #[test]
    fn a_full_window_evicts_its_coldest_entry() {
        let mut cache = WordCache::new(WINDOW);
        let words: Vec<Vec<u8>> = (0..WINDOW as u8).map(|i| vec![i; 4]).collect();
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
        let mut cache = WordCache::new(WINDOW);
        store(&mut cache, b"hot", [1u32].into_iter());
        for _ in 0..u8::MAX {
            cache.lookup(b"hot").hit();
        }
        let index = slot_of(&cache, b"hot").unwrap();
        assert_eq!(cache.usage[index].score(cache.epoch), u8::MAX);

        for _ in 0..MAX_HALVINGS {
            cache.fade();
        }
        assert_eq!(cache.usage[index].score(cache.epoch), 0);
    }

    /// An epoch is a byte, so the count has to start over at zero every 256
    /// bumps. An entry written before the restart carries an epoch larger than
    /// the current one, and left to itself the coldest thing in the table would
    /// come out reading as the hottest and could never be evicted again.
    #[test]
    fn a_lap_of_the_epoch_does_not_make_a_stale_entry_look_hot() {
        let mut cache = WordCache::new(WINDOW);
        store(&mut cache, b"stale", [1u32].into_iter());
        for _ in 0..u8::MAX {
            cache.lookup(b"stale").hit();
        }
        let index = slot_of(&cache, b"stale").unwrap();

        // One slot per window here, so one eviction is one epoch.
        assert_eq!(cache.evictions_per_epoch, 1);
        for _ in 0..=LAST_EPOCH {
            cache.fade();
        }
        assert_eq!(cache.epoch, 0, "the lap did not start over");
        assert_eq!(cache.usage[index].score(cache.epoch), 0);
    }

    /// The tail of `tags` is a copy of its first [`WINDOW`] entries, so that a
    /// window starting near the end of the table is still `WINDOW` entries in a
    /// row. Every write has to keep the two in step, or a probe whose window
    /// crosses the seam reads control bytes that no longer describe the slots.
    #[test]
    fn the_mirrored_tail_tracks_the_front() {
        let mut cache = WordCache::new(64);
        let n = cache.slots.len();
        for i in 0..500usize {
            store(
                &mut cache,
                format!("w{i}").as_bytes(),
                [i as u32].into_iter(),
            );
            assert_eq!(cache.tags[..WINDOW], cache.tags[n..], "tags, insert {i}");
            assert_eq!(cache.usage[..WINDOW], cache.usage[n..], "usage, insert {i}");
        }
        // Restarting the epochs rewrites every live entry's usage and no tag at
        // all, so the two must still agree afterwards.
        cache.restart_epochs();
        assert_eq!(cache.tags[..WINDOW], cache.tags[n..], "tags, after restart");
        assert_eq!(
            cache.usage[..WINDOW],
            cache.usage[n..],
            "usage, after restart"
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
        let mut tags = [EMPTY; WINDOW];
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
        let tags = [OCCUPIED | 0x11; WINDOW];
        let window = read_window(&tags);
        assert_eq!(empty_slots(window), 0);
        assert_eq!(
            steps(tag_matches(window, OCCUPIED | 0x11) & steps_before(0)).len(),
            WINDOW
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
            for step in 0..WINDOW {
                let mut tags = [OCCUPIED; WINDOW];
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
        assert!(cache.key_arena.data.len() <= 65 * 35);
        assert!(cache.id_arena.data.len() <= 65 * 8);
    }
}
