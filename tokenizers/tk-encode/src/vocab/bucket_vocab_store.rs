use std::collections::HashSet;

use ahash::RandomState;
use ptr_hash::{FastPtrHash, PtrHashParams, hash::NoHash};
use std::fmt;

use crate::utils::packed_key::{LONG_TAG, pack};

type Mphf = FastPtrHash<NoHash, u64>;

// Fixed seeds so a given vocab always hashes identically (the hasher is also stored on the struct,
// so build and query are guaranteed consistent regardless).
const SEEDS: [u64; 4] = [
    0x243F_6A88_85A3_08D3,
    0x1319_8A2E_0370_7344,
    0xA409_3822_299F_31D0,
    0x082E_FA98_EC4E_6C89,
];

/// One slot. `key` is the token itself when it fits in 15 bytes, otherwise its hash
/// with `LONG_TAG` set — see [`crate::utils::packed_key`].
///
/// Carrying the token here rather than only its coordinates is what makes a lookup
/// one memory access instead of two: confirming a hit is a register compare, so
/// `start`/`len` and the `bytes` slab are only touched for a token too long to pack.
/// That costs 32 bytes a slot against 12 (`key`'s alignment rounds 28 up), which on
/// llama-3's 128k tokens is 4.1 MB of entries against 1.5 MB. Measured on real
/// vocabularies and real pre-tokenized words it halves `get_bytes` on English and
/// code — `examples/vocab_entry_bench.rs` runs both layouts in one binary.
///
/// `len == 0` still marks a slot the build never wrote, which is what the
/// enumeration paths filter on.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
struct Entry {
    key: u128,
    id: u32,
    start: u32,
    len: u16,
    _pad: u16,
}

const _: () = assert!(std::mem::size_of::<Entry>() == 32);

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
    hasher: RandomState,
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
    longest_token_len: usize,
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

/// What a slot stores in its `key`, and the hash that places it.
///
/// A free function rather than a method because [`BucketVocabStore::build`] needs it
/// before there is a store, and both must derive the key the same way: the MPHF is
/// built from these hashes, so two derivations means every lookup misses.
#[inline]
fn slot_key(hasher: &RandomState, token: &[u8], head: Option<&[u8; 16]>) -> (u128, u64) {
    match pack(token, head) {
        // The key is the token, so it is also the whole hash input — one fixed-width
        // fold, where hashing the slice would mix the length in and then branch on it.
        Some(packed) => (packed, hasher.hash_one(packed)),
        None => {
            let hash = hasher.hash_one(token);
            ((hash as u128) | LONG_TAG, hash)
        }
    }
}

impl BucketVocabStore {
    pub fn build(tokens: Vec<(Vec<u8>, u32)>) -> Self {
        let n = tokens.len();

        let hasher = RandomState::with_seeds(SEEDS[0], SEEDS[1], SEEDS[2], SEEDS[3]);

        // 1. Pre-hash tokens -> u64 keys using near perfect hash func. Goes through
        //    `slot_key` because a query has to land on the same key, and the MPHF is
        //    built from these: derive them two different ways and every lookup misses.
        let keys: Vec<u64> = tokens
            .iter()
            .map(|(s, _)| slot_key(&hasher, s, None).1)
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
        let mut id_to_slot = vec![u32::MAX; max_id as usize + 1];
        let mut longest_token_len = 0;
        for (s, id) in &tokens {
            assert!(
                s.len() <= u16::MAX as usize,
                "token longer than 65535 bytes"
            );
            let (key, hash) = slot_key(&hasher, s, None);
            let slot = mphf.index(&hash);
            entries[slot] = Entry {
                key,
                id: *id,
                start: bytes.len() as u32,
                len: s.len() as u16,
                _pad: 0,
            };
            id_to_slot[*id as usize] = slot as u32;
            bytes.extend_from_slice(s);
            longest_token_len = longest_token_len.max(s.len())
        }

        Self {
            mphf,
            hasher,
            bytes: bytes.into_boxed_slice(),
            entries: entries.into_boxed_slice(),
            id_to_slot: id_to_slot.into_boxed_slice(),
            n,
            longest_token_len,
        }
    }

    pub fn new() -> Self {
        // convenient to build empty edit later.
        let empty: [u64; 0] = [];
        Self {
            mphf: FastPtrHash::<NoHash, u64>::new(&empty, PtrHashParams::default_fast()),
            hasher: RandomState::new(),
            bytes: Box::new([]),
            entries: Box::new([]),
            id_to_slot: Box::new([]),
            n: 0,
            longest_token_len: 0,
        }
    }

    /// This function is the equivalent of `get` on a HashaMap, it return the id
    /// corresponding to the key `q`. Since `mphf` always return a slot, we check
    /// whether the token indexed by that slot actually match the query. We don't
    /// care about collisions on query because of this!
    ///
    /// `head` is 16 readable bytes starting where `q` does, when the caller has them
    /// — a pre-tokenized word has the rest of its chunk behind it, so `Split::head`
    /// supplies one. `None` is always correct and returns the same id; it just packs
    /// the query with a copy whose length is known only at run time, which on real
    /// vocabularies is the difference between halving this function's cost and
    /// shaving a tenth off it.
    #[inline]
    pub fn get_bytes(&self, q: &[u8], head: Option<&[u8; 16]>) -> Option<u32> {
        if self.entries.is_empty() || q.len() > self.longest_token_len {
            return None;
        }
        let (key, hash) = slot_key(&self.hasher, q, head);
        let e = self.entries[self.mphf.index(&hash)];
        if e.key != key {
            return None;
        }
        // A packed key *is* the token, so an equal key is an equal token and there is
        // nothing left to check. A tagged one is only a hash, and two long tokens can
        // hash alike, so that case still confirms against the bytes.
        if key & LONG_TAG == 0 {
            return Some(e.id);
        }
        let (start, len) = (e.start as usize, e.len as usize);
        (len == q.len() && self.bytes[start..start + len] == *q).then_some(e.id)
    }

    #[inline]
    pub fn token_to_id(&self, s: &str) -> Option<u32> {
        self.get_bytes(s.as_bytes(), None)
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

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn content(&self) -> Vec<(String, u32)> {
        self.entries
            .iter()
            .filter(|e| e.len > 0)
            .filter_map(|m| self.id_to_token(m.id).map(|token| (token, m.id)))
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
            .filter_map(|m| {
                self.id_to_token_bytes(m.id)
                    .map(|token| (token.to_vec(), m.id))
            })
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
            assert_eq!(vocab.get_bytes(s, None), Some(*id), "fwd {s:?}");
            assert_eq!(vocab.id_to_token_bytes(*id), Some(s.as_slice()), "rev {id}");
        }
        for q in ["", "zzz", "th", "theX", "fo", "doggo"] {
            assert_eq!(vocab.token_to_id(q), None, "oov {q:?}");
        }
        assert_eq!(vocab.id_to_token(n as u32), None);
        assert_eq!(vocab.len(), n);
    }

    /// A token past 15 bytes keys on a hash instead of on its own bytes, so its slot
    /// still needs the byte comparison that a packed key makes unnecessary. Nothing
    /// else in this module reaches that branch.
    #[test]
    fn long_tokens_are_confirmed_against_their_bytes() {
        let long = b"a-token-well-past-fifteen-bytes".to_vec();
        let other = b"another-token-of-the-same-size!".to_vec();
        assert_eq!(long.len(), other.len(), "the point is a same-length miss");
        let vocab = BucketVocabStore::build(vec![(long.clone(), 7), (b"short".to_vec(), 1)]);

        assert_eq!(vocab.get_bytes(&long, None), Some(7));
        assert_eq!(vocab.id_to_token_bytes(7), Some(long.as_slice()));
        // Not in the vocabulary: it lands on some slot regardless, and only the bytes
        // say so. The length check cannot help here.
        assert_eq!(vocab.get_bytes(&other, None), None);
    }

    /// The window is a cheaper way to read the query, not part of the key, so it must
    /// not change the answer — for a packed key or a hashed one.
    #[test]
    fn a_window_does_not_change_the_id() {
        let chunk = b"short a-token-well-past-fifteen-bytes tail-padding-so-a-window-exists";
        let vocab = BucketVocabStore::build(vec![
            (b"short".to_vec(), 1),
            (b"a-token-well-past-fifteen-bytes".to_vec(), 7),
        ]);
        for (start, len, want) in [(0usize, 5usize, Some(1)), (6, 31, Some(7)), (38, 4, None)] {
            let q = &chunk[start..start + len];
            let head: &[u8; 16] = chunk[start..start + 16].try_into().unwrap();
            assert_eq!(vocab.get_bytes(q, None), want, "{q:?}");
            assert_eq!(vocab.get_bytes(q, Some(head)), want, "{q:?} with a window");
        }
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
        assert_eq!(vocab.get_bytes(b"", None), None);
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
        assert_eq!(vocab.get_bytes(&raw, None), Some(0));
        assert_eq!(vocab.id_to_token_bytes(0), Some(raw.as_slice()));
        assert_eq!(vocab.id_to_token(0), Some("\u{fffd}\u{fffd}".to_string()));
    }
}
