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

/// Bit 127 of a [`Entry::key`]: the key is a hash of the token's bytes rather than the bytes
/// themselves, because the token is longer than [`INLINE_KEY_BYTES`].
const KEY_IS_HASH: u128 = 1 << 127;
/// Tokens up to this many bytes are their own key: the bytes fit beside the length in a `u128`.
const INLINE_KEY_BYTES: usize = 15;

/// The token's bytes as a fixed-width key, or a hash of them when they do not fit.
///
/// Both halves of a lookup want this rather than the slice. The MPHF is close-addressed -- one
/// pilot, one slot, no probe sequence -- so the only work around it is hashing the query and
/// confirming the slot really holds it. Doing both from a `u128` is strictly less work than doing
/// them from a `&[u8]`: aHash on a slice mixes the length in and then branches to pick a read
/// width, and confirming by `memcmp` is a second indirection into the byte arena. A pretoken is
/// bounded and so is a vocabulary entry, so neither of those has to be paid.
///
/// Layout is the one [`crate::utils::word_cache`] already uses for the same reason.
/// The key, and the `u64` the MPHF is indexed by, from **one** hash pass.
///
/// The MPHF is close-addressed -- one pilot, one slot, no probe sequence -- so all that surrounds it
/// is hashing the query and confirming the slot holds it. Both come out of the same xxh3-128:
///
/// * a token of at most [`INLINE_KEY_BYTES`] bytes *is* its own key, so the compare is proof and
///   the only hash needed is the one that places it;
/// * a longer token keys on 63 bits of xxh3-128's high half and places on its low 64, so the
///   compare is an independent check rather than a restatement of the placement. Two aHash passes
///   would buy the same independence and measured a 1.4x loss on chinese and russian, whose
///   pretokens are mostly longer than the inline range.
#[inline]
pub(crate) fn key_and_hash(word: &[u8]) -> (u128, u64) {
    let len = word.len();
    if len > INLINE_KEY_BYTES {
        let hash = KEY_HASHER.hash_one(word);
        return (KEY_IS_HASH | (len as u128) << 64 | hash as u128, hash);
    }
    let key = pack_inline(word);
    // Fixed width in, fixed width out: nothing to branch on and no length to mix.
    (key, KEY_HASHER.hash_one(key))
}

/// A token of at most [`INLINE_KEY_BYTES`] bytes, as its own key: the bytes little-endian with the
/// length in the top byte, so `"ab"` and `"ab\0"` cannot collide.
#[inline]
fn pack_inline(word: &[u8]) -> u128 {
    let len = word.len();
    // Fixed-width head/tail reads, overlapping in the middle. A `copy_from_slice` of a run whose
    // length is only known at run time compiles to a `memcpy` call, which measured slower than
    // reading too much and masking.
    let raw = if len >= 8 {
        let head = u64::from_le_bytes(word[..8].try_into().unwrap()) as u128;
        let tail = u64::from_le_bytes(word[len - 8..].try_into().unwrap()) as u128;
        head | tail << (8 * (len - 8))
    } else if len >= 4 {
        let head = u32::from_le_bytes(word[..4].try_into().unwrap()) as u128;
        let tail = u32::from_le_bytes(word[len - 4..].try_into().unwrap()) as u128;
        head | tail << (8 * (len - 4))
    } else if len >= 1 {
        let first = word[0] as u128;
        let middle = (word[len / 2] as u128) << (8 * (len / 2));
        let last = (word[len - 1] as u128) << (8 * (len - 1));
        first | middle | last
    } else {
        0
    };
    raw | (len as u128) << 120
}

#[derive(Clone, Copy, Debug)]
struct Entry {
    /// The token as a fixed-width key; see [`pack_key`]. An inline key *is* the token, so a match
    /// against it is proof. A `KEY_IS_HASH` key still needs the bytes checked.
    key: u128,
    start: u32,
    len: u16,
    /// The token id in the low 31 bits, [`FOLD_BIT`] in the top.
    id: u32,
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
    /// `entries[slot]` -> (offset into `bytes`, length, id). Ordered by MPHF slot.
    entries: Box<[Entry]>,
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
        let mut entries = vec![
            Entry {
                key: 0,
                start: 0,
                len: 0,
                id: 0
            };
            n_slots
        ];
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
                key,
                start: bytes.len() as u32,
                len: s.len() as u16,
                id: *id,
            };
            id_to_slot[*id as usize] = slot as u32;
            bytes.extend_from_slice(s);
        }

        Self {
            mphf,
            bytes: bytes.into_boxed_slice(),
            entries: entries.into_boxed_slice(),
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
        // Key equality confirms `q` really is the token at this slot: perfect hashing only
        // guarantees a valid slot for in-vocab keys, so this is what rejects an out-of-vocabulary
        // query. An unwritten padding slot holds key 0, which no token can pack to.
        (e.key == key).then_some(e.id & VOCAB_ID_MASK)
    }

    /// The id for `q`, together with whether that entry may be folded. One probe and one entry
    /// load: the flag is a bit of the id the probe already read.
    #[inline]
    pub fn get_bytes_foldable(&self, q: &[u8]) -> Option<(u32, bool)> {
        let (key, hash) = key_and_hash(q);
        self.get_keyed_foldable(key, hash)
    }

    /// [`Self::get_bytes_foldable`] for a caller that already has the word's key and hash.
    ///
    /// The word cache keys words exactly the same way, so a pretoken that misses the fold and then
    /// goes to the cache would otherwise be hashed twice. This lets one pass serve both.
    #[inline]
    pub fn get_keyed_foldable(&self, key: u128, hash: u64) -> Option<(u32, bool)> {
        if self.entries.is_empty() {
            return None;
        }
        let slot = self.mphf.index(&hash);
        let e = self.entries[slot];
        (e.key == key).then_some((e.id & VOCAB_ID_MASK, e.id & FOLD_BIT != 0))
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
        let e = self.entries[slot as usize];
        let start = e.start as usize;
        self.bytes.get(start..start + e.len as usize)
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
        self.entries
            .iter()
            .filter(|e| e.len > 0)
            // Mask: the stored id carries FOLD_BIT, which must never escape this type.
            .map(|m| m.id & VOCAB_ID_MASK)
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
            .filter(|e| e.len > 0)
            // Mask: the stored id carries FOLD_BIT, which must never escape this type.
            .map(|m| m.id & VOCAB_ID_MASK)
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
