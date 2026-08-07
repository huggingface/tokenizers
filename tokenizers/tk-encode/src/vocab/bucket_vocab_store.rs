use std::collections::HashSet;

use ahash::RandomState;
use ptr_hash::{FastPtrHash, PtrHashParams, hash::NoHash};
use std::fmt;

/// Hashes a word key. Fixed seeds so a vocabulary always hashes identically.
///
/// One pass. xxh3-128 was tried, to give a long key 63 bits of discrimination independent of the 64
/// that place it; it cost 5.8 -> 7.0 ns per probe and ~10-25% on chinese and russian, whose pretokens
/// are mostly long, so it was dropped. A long key reuses its placement hash as its discriminant and
/// adds the length instead: a false hit needs a 64-bit collision at equal length (~2^-64 per query)
/// rather than being impossible as the old `memcmp` made it.
static KEY_HASHER: RandomState = RandomState::with_seeds(
    0x243F_6A88_85A3_08D3,
    0x1319_8A2E_0370_7344,
    0xA409_3822_299F_31D0,
    0x082E_FA98_EC4E_6C89,
);

type Mphf = FastPtrHash<NoHash, u64>;

// No hasher on the struct: both hashes below are fixed, so build and query agree without one
// having to be carried along to keep them consistent.

/// Bit 31 of a stored id: the token provably encodes to itself, so a pretoken equal to it can be
/// emitted without running the merge loop. See `PipelineBPE::prove_fold`.
///
/// Packed into the id rather than kept beside it, following the same convention as a packed merge
/// value, where the low bits are the product id and the high bits are a flag field (`SAFE_MASK`).
/// The lookup already loads this entry to verify the key, so the flag costs no memory, no extra
/// load, and no second structure to keep in step with the vocabulary.
const FOLD_BIT: u32 = 1 << 31;
/// The id half. 2^31 ids is far past any vocabulary.
const VOCAB_ID_MASK: u32 = FOLD_BIT - 1;

/// Tokens up to this many bytes are their own key: the bytes fit beside the length in a `u64`.
///
/// Seven, not fifteen, because that is what the corpus is. English averages 4.83 bytes per
/// pretoken and code 4.08, so a `u128` key was paying double width, a two-part head/tail read and a
/// 32-byte entry to describe words that fit in a register.
pub(crate) const INLINE_KEY_BYTES: usize = 7;

/// Mixes a short key into the well-distributed `u64` the MPHF wants.
///
/// One multiply and one shift. An inline key *is* the token, so the compare is exact no matter how
/// the slot was chosen -- the hash only has to spread well enough for the MPHF to separate the keys,
/// and aHash's rounds, or splitmix64's second multiply, are wasted on that. Dropping the mixing
/// entirely does not work: packed short keys share their high bytes, and construction fails outright
/// with "indistinguishable hashes in bucket".
#[inline(always)]
fn mix(z: u64) -> u64 {
    let z = z.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z ^ (z >> 29)
}

/// The token's bytes as a fixed-width key, and the `u64` the MPHF is indexed by, from one pass.
///
/// * up to [`INLINE_KEY_BYTES`] bytes: the key *is* the bytes, with the length in the top byte, so
///   the compare is proof and `"ab"` cannot collide with `"ab\0"`.
/// * longer: the key is aHash of the bytes, which mixes the length in, and doubles as the placement
///   hash. A false hit then needs a full 64-bit collision (~2^-64 per query) rather than being
///   impossible as a `memcmp` against the byte arena made it.
///
/// [`crate::utils::word_cache`] keys through this very function, so a pretoken probed in both tables
/// is hashed once for the pair.
/// [`key_and_hash`] when the caller can guarantee `readable` bytes exist from the word's start.
///
/// A pretoken sits inside a chunk, so as long as it is not within 8 bytes of the chunk's end, its
/// key can be one unaligned 8-byte load masked to the length -- instead of a head load, a tail load
/// and a variable shift to stitch them. `pack` measured 1.6-2.1 ns/word, the largest single piece of
/// the fold path, and the fold answers 88-94% of pretokens.
///
/// # Safety
/// Reading is safe for any `readable >= 8`; the mask discards whatever came from past the word.
#[inline]
pub fn key_and_hash_readable(word: &[u8], readable: usize) -> (u64, u64) {
    let len = word.len();
    if len > INLINE_KEY_BYTES || readable < 8 {
        return key_and_hash(word);
    }
    // SAFETY: `readable >= 8` bytes exist from `word.as_ptr()`, and `len <= 7 < 8`.
    let raw = unsafe { word.as_ptr().cast::<u64>().read_unaligned() };
    // Two table loads, both independent of `raw`, so they overlap its load instead of queueing
    // behind a shift chain. Computing them instead -- `u64::MAX >> (64 - 8 * len)` and `len << 56` --
    // measured 1.020x against this 1.034x: the arithmetic is more instructions on a path that is
    // bound by how many it runs, and the loads were never waiting on anything. `len <= 7` is guaranteed above, so neither index is checked at runtime.
    // SAFETY: `len <= INLINE_KEY_BYTES == 7`, and both tables have 8 entries.
    let (mask, tag) = unsafe { (*KEY_MASK.get_unchecked(len), *LEN_TAG.get_unchecked(len)) };
    let key = (raw & mask) | tag;
    debug_assert_eq!(key, key_and_hash(word).0, "masked load must match the stitched pack");
    (key, mix(key))
}

#[inline]
pub fn key_and_hash(word: &[u8]) -> (u64, u64) {
    let len = word.len();
    if len > INLINE_KEY_BYTES {
        let hash = KEY_HASHER.hash_one(word);
        return (hash, hash);
    }
    // One unaligned load of the whole key range, masked to the length. Reading past the word is not
    // allowed, so read the tail and shift: still register-only, no `memcpy`.
    let raw = if len >= 4 {
        let head = u32::from_le_bytes(word[..4].try_into().unwrap()) as u64;
        let tail = u32::from_le_bytes(word[len - 4..].try_into().unwrap()) as u64;
        head | tail << (8 * (len - 4))
    } else if len >= 1 {
        let first = word[0] as u64;
        let middle = (word[len / 2] as u64) << (8 * (len / 2));
        let last = (word[len - 1] as u64) << (8 * (len - 1));
        first | middle | last
    } else {
        0
    };
    let key = raw | (len as u64) << 56;
    (key, mix(key))
}

/// One probe entry: a digest to confirm the slot, and the id to return. **8 bytes.**
///
/// The probe is the encode path's most expensive step -- 5.08 ns per span, measured, which is an L2
/// miss: the MPHF scatters a corpus's few thousand hot words across the whole entry table, so each
/// one lands on its own line. Halving the entry halves the lines the hot set occupies. 32 bytes ->
/// 16 -> 8 across this session, and 50257 entries is now 400 KB where it started at 1.6 MB.
///
/// A 32-bit digest, not the full key. Perfect hashing already guarantees that an *in-vocabulary*
/// word reaches its own slot, so the stored value only has to reject an out-of-vocabulary query --
/// about 6% of latin pretokens. A wrong id needs one of those to collide in 32 bits: ~2^-32 per
/// missing word, against the ~2^-64 the long-key path already accepts.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
struct Entry {
    digest: u32,
    /// The token id in the low 31 bits, [`FOLD_BIT`] in the top.
    id: u32,
}

const _: () = assert!(size_of::<Entry>() == 8);

/// `KEY_MASK[len]` keeps the low `len` bytes; `LEN_TAG[len]` is the length in the top byte.
///
/// Two tiny always-resident loads instead of a four-deep dependent ALU chain
/// (`len -> 8*len -> 64-x -> shift -> and`). Packing the key measured 1.94 ns per span in situ, more
/// than the hash, the MPHF lookup and the entry load put together, and that chain is why: the loads
/// below issue in parallel with the word's own load, where the shifts could not.
static KEY_MASK: [u64; 8] = [
    0x0000_0000_0000_0000,
    0x0000_0000_0000_00FF,
    0x0000_0000_0000_FFFF,
    0x0000_0000_00FF_FFFF,
    0x0000_0000_FFFF_FFFF,
    0x0000_00FF_FFFF_FFFF,
    0x0000_FFFF_FFFF_FFFF,
    0x00FF_FFFF_FFFF_FFFF,
];
static LEN_TAG: [u64; 8] = [
    0 << 56,
    1 << 56,
    2 << 56,
    3 << 56,
    4 << 56,
    5 << 56,
    6 << 56,
    7 << 56,
];

/// The 32 bits an entry stores to reject an out-of-vocabulary query. Derived from the key by a
/// different multiply than the placement hash, so it is not a restatement of the slot.
#[inline(always)]
/// The 32 bits that confirm a slot really holds the queried token.
///
/// Taken from the hash rather than recomputed from the key. `mix` already multiplies the key by this
/// crate's odd constant, and the old digest multiplied by the *same* constant a second time, so every
/// pretoken paid two 64-bit multiplies where one does. `hash` is `m ^ (m >> 29)` for that product, so
/// its top half is still a deterministic, well-spread function of the key -- which is all a digest
/// has to be. Build and query both go through here, so they cannot disagree.
fn digest_of(hash: u64) -> u32 {
    (hash >> 32) as u32
}

/// `slot -> (offset into `bytes`, length)`. Off the probe path on purpose: only the reverse lookup
/// wants it, and keeping it in `Entry` made every probe drag 8 dead bytes through cache.
#[derive(Clone, Copy, Debug, Default)]
struct Span {
    start: u32,
    len: u16,
}

/// The BucketVocabStore optimizes for space and speed. We don't use a HashMap to prevent duplicating the
/// keys. Instead, we just use an `id_to_slot` and `entries` table. When you query bytes, you hash
/// on the fly and get an `index` into the `entries` table. When you query an `id`, you fetch in
/// the `id_to_slot` the same index.
///
/// Entries store start, len and the actual `id` of the token.
/// Example:
///
/// ```
/// use tk_encode::vocab::bucket_vocab_store::BucketVocabStore;
/// let vocab = BucketVocabStore::build(vec![
///     (b"a".to_vec(), 0),
///     (b"bb".to_vec(), 5),
///     (b"ccc".to_vec(), 100),
/// ]);
/// vocab.token_to_id("a");
/// vocab.id_to_token(100);
/// ```
#[derive(Clone)]
pub struct BucketVocabStore {
    mphf: Mphf,
    /// All token bytes, concatenated. Ordered by MPHF slot.
    bytes: Box<[u8]>,
    /// `entries[slot]` -> (key, id). Ordered by MPHF slot. The probe touches only this.
    entries: Box<[Entry]>,
    /// `spans[slot]` -> where the token's bytes are. Reverse lookup only.
    spans: Box<[Span]>,
    /// `id_to_slot[token_id] -> entry_idx` -> index into entries as the entries are not really sorted.
    id_to_slot: Box<[u32]>,
    /// Number of real tokens. Cached at build so `len()` is O(1): `entries` is sized to the
    /// MPHF's non-minimal slot range (with phantom padding slots), so its length is not the
    /// token count.
    n: usize,
}

impl fmt::Debug for BucketVocabStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BucketVocabStore")
            .field("bytes", &self.bytes)
            .field("id_to_slot", &self.id_to_slot)
            .field("entries", &self.entries)
            .finish()
    }
}
impl PartialEq for BucketVocabStore {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        // early exit as soon as there is a missmatch
        for id in 0..self.len() {
            if self.id_to_token(id as u32) != other.id_to_token(id as u32) {
                return false;
            }
        }
        true
    }
}

impl Default for BucketVocabStore {
    fn default() -> Self {
        Self::new()
    }
}

impl BucketVocabStore {
    pub fn build(tokens: Vec<(Vec<u8>, u32)>) -> Self {
        let n = tokens.len();

        // 1. Pre-hash token bytes -> u64 keys using near perfect hash func.
        //    Via the packed key, so build and query fold the same fixed-width value.
        let keys: Vec<u64> = tokens
            .iter()
            .map(|(s, _)| key_and_hash(s.as_slice()).1)
            .collect();

        // 2. A perfect hash needs distinct keys. Collisions are astronomically unlikely
        //    (~n^2/2^65); if one ever fires, switch the key type to u128. The byte check below makes
        //    a collision a correct miss at query time, but it would drop a token at build, so guard.

        let mut seen = HashSet::with_capacity(n);
        for k in &keys {
            let overlap = seen.insert(*k);
            if !overlap {
                println!(
                    "Either 2 keys are the same or 64-bit hash collision in vocab; rebuild with u128 keys: {:?}",
                    tokens
                        .iter()
                        .map(|(s, _)| String::from_utf8_lossy(s))
                        .collect::<Vec<_>>()
                );
            }
        }

        // 3. Build the (non-minimal) `FastPtrHash` via `PtrHashParams::default_fast()`; query with `.index()`.
        let params = PtrHashParams::default_fast();
        let mphf = Mphf::new(&seen.into_iter().collect::<Vec<u64>>(), params);

        // FastPtrHash is non-minimal: `index()` may return a slot up to `max_index()` (>= n),
        // so `entries` must be sized to cover the whole slot range. Slots never written by the
        // build loop stay as the default `Entry { len: 0, .. }` (phantom/padding slots), which
        // enumeration/count paths filter out via `len > 0`.
        let n_slots = mphf.max_index();

        // 4. Place each token at its MPHF slot; build the slab and the id->slot reverse table.
        let total: usize = tokens.iter().map(|(s, _)| s.len()).sum();
        let max_id = tokens.iter().map(|(_, id)| *id).max().unwrap();
        let mut bytes = Vec::with_capacity(total);
        let mut entries = vec![Entry::default(); n_slots];
        let mut spans = vec![Span::default(); n_slots];
        let mut id_to_slot = vec![u32::MAX; max_id as usize + 1];
        for (s, id) in &tokens {
            assert!(
                s.len() <= u16::MAX as usize,
                "token longer than 65535 bytes"
            );
            assert!(*id <= VOCAB_ID_MASK, "token id {id} needs bit 31, which holds FOLD_BIT");
            let (key, hash) = key_and_hash(s.as_slice());
            let slot = mphf.index(&hash);
            entries[slot] = Entry {
                digest: digest_of(hash),
                id: *id,
            };
            spans[slot] = Span {
                start: bytes.len() as u32,
                len: s.len() as u16,
            };
            id_to_slot[*id as usize] = slot as u32;
            bytes.extend_from_slice(s);
        }

        Self {
            mphf,
            bytes: bytes.into_boxed_slice(),
            entries: entries.into_boxed_slice(),
            spans: spans.into_boxed_slice(),
            id_to_slot: id_to_slot.into_boxed_slice(),
            n,
        }
    }

    pub fn new() -> Self {
        // convenient to build empty edit later.
        let empty: [u64; 0] = [];
        Self {
            mphf: FastPtrHash::<NoHash, u64>::new(&empty, PtrHashParams::default_fast()),
            bytes: Box::new([]),
            entries: Box::new([]),
            spans: Box::new([]),
            id_to_slot: Box::new([]),
            n: 0,
        }
    }

    /// This function is the equivalent of `get` on a HashaMap, it return the id
    /// corresponding to the key `q`. Since `mphf` always return a slot, we check
    /// whether the token indexed by that slot actually match the query. We don't
    /// care about collisions on query because of this!
    #[inline]
    pub fn get_bytes(&self, q: &[u8]) -> Option<u32> {
        if self.entries.is_empty() {
            return None;
        }
        let (key, hash) = key_and_hash(q);
        let slot = self.mphf.index(&hash);
        let e = self.entries[slot];
        // Digest equality confirms `q` really is the token at this slot: perfect hashing only
        // guarantees a valid slot for in-vocab keys, so this is what rejects an out-of-vocabulary
        // query. An unwritten padding slot holds key 0, which no token can pack to.
        (e.digest == digest_of(hash)).then_some(e.id & VOCAB_ID_MASK)
    }

    /// The id for `q`, together with whether that entry may be folded. One probe and one entry
    /// load: the flag is a bit of the id the probe already read.
    #[inline]
    pub fn get_bytes_foldable(&self, q: &[u8]) -> Option<(u32, bool)> {
        let (key, hash) = key_and_hash(q);
        self.get_keyed_foldable(key, hash)
    }

    /// The slot a hash lands in. Split out of the probe so a caller with many words can issue all
    /// the pilot loads before it needs any of the answers -- see [`Self::entry_at`].
    #[inline(always)]
    pub fn probe_slot(&self, hash: u64) -> usize {
        self.mphf.index(&hash)
    }

    /// The `(key, id)` at a slot, without deciding anything.
    ///
    /// A probe is a chain of two dependent loads -- pilot, then entry -- and at one word at a time
    /// the whole chain is exposed latency. A caller holding N words can run `probe_slot` for all of
    /// them, then `entry_at` for all of them, and the CPU has N independent misses outstanding
    /// instead of one. Same loads, same table, N times the memory parallelism.
    #[inline(always)]
    pub fn entry_at(&self, slot: usize) -> (u32, u32) {
        let e = self.entries[slot];
        (e.digest, e.id)
    }

    /// Decide a probe from what [`Self::entry_at`] already loaded.
    #[inline(always)]
    pub fn resolve_foldable(hash: u64, entry: (u32, u32)) -> Option<(u32, bool)> {
        let (edigest, eid) = entry;
        (edigest == digest_of(hash)).then_some((eid & VOCAB_ID_MASK, eid & FOLD_BIT != 0))
    }

    /// [`Self::get_bytes_foldable`] for a caller that already has the word's key and hash.
    ///
    /// The word cache keys words exactly the same way, so a pretoken that misses the fold and then
    /// goes to the cache would otherwise be hashed twice. This lets one pass serve both.
    #[inline]
    pub fn get_keyed_foldable(&self, key: u64, hash: u64) -> Option<(u32, bool)> {
        let _ = key;
        if self.entries.is_empty() {
            return None;
        }
        let slot = self.mphf.index(&hash);
        let e = self.entries[slot];
        (e.digest == digest_of(hash)).then_some((e.id & VOCAB_ID_MASK, e.id & FOLD_BIT != 0))
    }

    /// Records that this token folds to itself. Called once per entry at load, after the proof.
    pub fn set_foldable(&mut self, id: u32) {
        if let Some(&slot) = self.id_to_slot.get(id as usize)
            && slot != u32::MAX
        {
            self.entries[slot as usize].id |= FOLD_BIT;
        }
    }

    #[inline]
    pub fn token_to_id(&self, s: &str) -> Option<u32> {
        self.get_bytes(s.as_bytes())
    }

    /// `id -> token bytes`, borrowing from the slab (no allocation).
    #[inline]
    pub fn id_to_token_bytes(&self, id: u32) -> Option<&[u8]> {
        let slot = *self.id_to_slot.get(id as usize)?;
        if slot == u32::MAX {
            return None; // id is within range but absent from the vocab
        }
        let sp = self.spans[slot as usize];
        let start = sp.start as usize;
        self.bytes.get(start..start + sp.len as usize)
    }

    #[inline]
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        // we are not sure its a valid utf8 so if not, adds replacement char
        self.id_to_token_bytes(id)
            .map(|b| String::from_utf8_lossy(b).into_owned())
    }

    pub fn len(&self) -> usize {
        self.n
    }

    /// One past the highest id this vocabulary can hold.
    ///
    /// Ids are not dense: a config may leave gaps, so [`Self::len`] counts entries and is *not* an
    /// id bound. Anything that walks ids has to bound itself by this and skip the holes, which
    /// [`Self::id_to_token_bytes`] reports as `None`.
    pub fn id_space(&self) -> usize {
        self.id_to_slot.len()
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn content(&self) -> Vec<(String, u32)> {
        // `spans` says which slots the build actually wrote: a padding slot keeps length 0.
        self.entries
            .iter()
            .zip(self.spans.iter())
            .filter(|(_, sp)| sp.len > 0)
            // Mask: the stored id carries FOLD_BIT, which must never escape this type.
            .map(|(e, _)| e.id & VOCAB_ID_MASK)
            .filter_map(|id| self.id_to_token(id).map(|token| (token, id)))
            .collect()
    }

    /// Alias for `content` — kept so `Model::get_vocab` call sites resolve.
    pub fn get_vocab(&self) -> Vec<(String, u32)> {
        self.content()
    }

    /// convenient when we want to re-build a vocab
    pub fn byte_content(&self) -> Vec<(Vec<u8>, u32)> {
        self.entries
            .iter()
            .zip(self.spans.iter())
            .filter(|(_, sp)| sp.len > 0)
            // Mask: the stored id carries FOLD_BIT, which must never escape this type.
            .map(|(e, _)| e.id & VOCAB_ID_MASK)
            .filter_map(|id| self.id_to_token_bytes(id).map(|token| (token.to_vec(), id)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_token() {
        let vocab = BucketVocabStore::build(vec![(b"Hel".to_vec(), 0)]);
        assert_eq!(vocab.token_to_id("Hel"), Some(0));
        assert_eq!(vocab.token_to_id("lo"), None);
        assert_eq!(vocab.id_to_token(0), Some("Hel".to_string()));
        assert_eq!(vocab.id_to_token(1000), None);
    }

    #[test]
    fn many_tokens_roundtrip() {
        let toks: Vec<(Vec<u8>, u32)> = [
            "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "Ġthe", "▁hello", "\n",
            "12345",
        ]
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_bytes().to_vec(), i as u32))
        .collect();
        let n = toks.len();
        let vocab = BucketVocabStore::build(toks.clone());

        for (s, id) in &toks {
            assert_eq!(vocab.get_bytes(s), Some(*id), "fwd {s:?}");
            assert_eq!(vocab.id_to_token_bytes(*id), Some(s.as_slice()), "rev {id}");
        }
        for q in ["", "zzz", "th", "theX", "fo", "doggo"] {
            assert_eq!(vocab.token_to_id(q), None, "oov {q:?}");
        }
        assert_eq!(vocab.id_to_token(n as u32), None);
        assert_eq!(vocab.len(), n);
    }

    #[test]
    fn sparse_ids_with_gaps() {
        let vocab = BucketVocabStore::build(vec![
            (b"a".to_vec(), 0),
            (b"bb".to_vec(), 5),
            (b"ccc".to_vec(), 100),
        ]);
        assert_eq!(vocab.token_to_id("a"), Some(0));
        assert_eq!(vocab.token_to_id("bb"), Some(5));
        assert_eq!(vocab.token_to_id("ccc"), Some(100));
        assert_eq!(vocab.id_to_token(0), Some("a".to_string()));
        assert_eq!(vocab.id_to_token(5), Some("bb".to_string()));
        assert_eq!(vocab.id_to_token(100), Some("ccc".to_string()));
        assert_eq!(vocab.id_to_token(1), None);
        assert_eq!(vocab.id_to_token(50), None);
    }

    #[test]
    fn empty_store() {
        let vocab = BucketVocabStore::new();
        assert!(vocab.is_empty());
        assert_eq!(vocab.len(), 0);
        assert_eq!(vocab.token_to_id("anything"), None);
        assert_eq!(vocab.get_bytes(b""), None);
        assert_eq!(vocab.id_to_token(0), None);
        assert!(vocab.content().is_empty());
    }

    #[test]
    fn eq_matches_on_dense_content() {
        // Models use dense ids (0..n); equality must reflect the token set on that range.
        let a = BucketVocabStore::build(vec![(b"x".to_vec(), 0), (b"y".to_vec(), 1)]);
        let b = BucketVocabStore::build(vec![(b"y".to_vec(), 1), (b"x".to_vec(), 0)]);
        let c = BucketVocabStore::build(vec![(b"x".to_vec(), 0), (b"z".to_vec(), 1)]);
        let d = BucketVocabStore::build(vec![(b"x".to_vec(), 0)]);
        assert_eq!(a, b); // insertion order does not matter
        assert_ne!(a, c); // different token at id 1
        assert_ne!(a, d); // different length
    }

    #[test]
    fn content_views_agree() {
        let vocab = BucketVocabStore::build(vec![(b"hi".to_vec(), 0), (b"yo".to_vec(), 1)]);
        let mut got = vocab.get_vocab();
        got.sort();
        assert_eq!(got, vec![("hi".to_string(), 0), ("yo".to_string(), 1)]);
        assert_eq!(vocab.content(), vocab.get_vocab());
        let mut bytes = vocab.byte_content();
        bytes.sort();
        assert_eq!(bytes, vec![(b"hi".to_vec(), 0), (b"yo".to_vec(), 1)]);
    }

    #[test]
    fn non_utf8_token_is_lossy_on_string_but_exact_on_bytes() {
        let raw = vec![0xffu8, 0xfe];
        let vocab = BucketVocabStore::build(vec![(raw.clone(), 0)]);
        assert_eq!(vocab.get_bytes(&raw), Some(0));
        assert_eq!(vocab.id_to_token_bytes(0), Some(raw.as_slice()));
        assert_eq!(vocab.id_to_token(0), Some("\u{fffd}\u{fffd}".to_string()));
    }
}
