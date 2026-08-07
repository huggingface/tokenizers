use std::collections::HashSet;

use ahash::RandomState;
use ptr_hash::{FastPtrHash, PtrHashParams, hash::NoHash};
use std::fmt;

static KEY_HASHER: RandomState = RandomState::with_seeds(
    0x243F_6A88_85A3_08D3,
    0x1319_8A2E_0370_7344,
    0xA409_3822_299F_31D0,
    0x082E_FA98_EC4E_6C89,
);

type Mphf = FastPtrHash<NoHash, u64>;


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

/// A word this long or shorter packs into the *cheap* tier: eight-byte masked load, one multiply.
pub(crate) const INLINE_KEY_BYTES: usize = 7;
/// ...and this long or shorter into the wide tier: sixteen-byte masked load, wyhash fold. Still
/// exact -- the key is the word -- so a cache hit on anything up to this length cannot be wrong.
///
/// Two tiers rather than one, because collapsing them lost: making every length take the wide path
/// (`mix128` is a 64x64->128 multiply where `mix` is one multiply, and the wide masked load needs 16
/// readable bytes so short words near a chunk's end fall back to the stitched pack) cost the corpora
/// whose pretokens are *already* under seven bytes -- hindi 0.857, tamil 0.868, bengali 0.870, whose
/// pretokens are one or two three-byte characters. The wide tier only has to serve the range that
/// was paying ahash: 8..15 bytes, which is where ASCII words live.
pub(crate) const WIDE_KEY_BYTES: usize = 15;

/// Set on the key of a word too long even for the wide tier, where the key is 64 bits of hash rather
/// than the word. Neither packed tier can set it: the cheap tier leaves bits 64.. clear and the wide
/// tier puts a length of at most 15 at bits 120..=126.
pub(crate) const KEY_IS_HASH: u128 = 1 << 127;

#[inline(always)]
fn mix(z: u64) -> u64 {
    let z = z.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z ^ (z >> 29)
}

/// Fold a wide packed key to the 64 bits the placement and the digest come from: one 64x64->128
/// multiply of the halves, then xor the product's halves -- wyhash's core.
///
/// **Do not "simplify" this to `lo ^ hi`.** That fold is linear, so the same xor delta at the same
/// byte offset in each half cancels: ` dignified` and ` signifies` differ only at bytes 1 and 9, both
/// `d` vs `s`, so they folded to one hash and a vocabulary holding one answered a query for the
/// other. Regression test: `the_fold_separates_words_whose_halves_cancel`.
#[inline(always)]
fn mix128(key: u128) -> u64 {
    const A: u64 = 0xA076_1D64_78BD_642F;
    const B: u64 = 0xE703_7ED1_A0B4_28DB;
    let lo = key as u64;
    let hi = (key >> 64) as u64;
    let product = ((lo ^ A) as u128).wrapping_mul((hi ^ B) as u128);
    (product as u64) ^ ((product >> 64) as u64)
}

/// `KEY_MASK_WIDE[len]` keeps the low `len` bytes of a sixteen-byte load; `LEN_TAG_WIDE[len]` is the
/// length that goes on top. Spelled as loops because sixteen 128-bit literals are harder to check
/// than the rule they follow.
static KEY_MASK_WIDE: [u128; WIDE_KEY_BYTES + 1] = {
    let mut mask = [0u128; WIDE_KEY_BYTES + 1];
    let mut len = 1;
    while len <= WIDE_KEY_BYTES {
        mask[len] = (1u128 << (8 * len)) - 1;
        len += 1;
    }
    mask
};
static LEN_TAG_WIDE: [u128; WIDE_KEY_BYTES + 1] = {
    let mut tag = [0u128; WIDE_KEY_BYTES + 1];
    let mut len = 0;
    while len <= WIDE_KEY_BYTES {
        tag[len] = (len as u128) << 120;
        len += 1;
    }
    tag
};

///
///
///
///
/// # Safety
/// `inline(always)`, and the long-word arm is a separate `inline(never)` function, because plain
/// `#[inline]` left this outlined: a profile of warm english had 8.3% of self time inside
/// `key_and_hash`, for what should be a masked load, an `or` and a multiply. LLVM weighs the whole
/// body when it decides, and the body used to contain the ahash call for a word too long to pack, so
/// the cheap path paid for the expensive one's size. Splitting them lets the pack inline into the
/// span loop while the hash stays a call -- which is also the shape it wants, since the long arm is
/// already dominated by hashing rather than by call overhead.
#[inline(always)]
pub fn key_and_hash_readable(word: &[u8], readable: usize) -> (u128, u64) {
    let len = word.len();
    // Cheap tier first: it is the common case for every script whose characters are multi-byte, and
    // it is strictly less work than the wide one.
    if len <= INLINE_KEY_BYTES && readable >= 8 {
        // SAFETY: `readable >= 8` bytes exist from `word.as_ptr()`, and `len <= 7 < 8`.
        let raw = unsafe { word.as_ptr().cast::<u64>().read_unaligned() };
        // SAFETY: `len <= INLINE_KEY_BYTES == 7`, and both tables have 8 entries.
        let (mask, tag) = unsafe { (*KEY_MASK.get_unchecked(len), *LEN_TAG.get_unchecked(len)) };
        let key = (raw & mask) | tag;
        debug_assert_eq!(
            key as u128,
            key_and_hash(word).0,
            "masked load must match the stitched pack"
        );
        return (key as u128, mix(key));
    }
    // Wide tier: an ASCII word of 8..15 bytes used to reach ahash from here.
    if len <= WIDE_KEY_BYTES && readable >= 16 {
        // SAFETY: `readable >= 16` bytes exist from `word.as_ptr()`, and `len <= 15 < 16`.
        let raw = unsafe { word.as_ptr().cast::<u128>().read_unaligned() };
        // SAFETY: `len <= WIDE_KEY_BYTES == 15`, and both tables have 16 entries.
        let (mask, tag) = unsafe {
            (
                *KEY_MASK_WIDE.get_unchecked(len),
                *LEN_TAG_WIDE.get_unchecked(len),
            )
        };
        let key = (raw & mask) | tag;
        debug_assert_eq!(key, key_and_hash(word).0, "masked load must match the stitched pack");
        return (key, mix128(key));
    }
    key_and_hash(word)
}

/// A word too long to pack into the key: hash it for real. Kept out of line so that the packing
/// path above and in [`key_and_hash`] can be inlined without dragging ahash in with them. Not
/// `#[cold]` on purpose -- for CJK this is the *common* arm, and telling the predictor otherwise
/// would cost more than the call.
#[inline(never)]
fn key_and_hash_long(word: &[u8]) -> (u128, u64) {
    let hash = KEY_HASHER.hash_one(word);
    ((hash as u128) | KEY_IS_HASH, hash)
}

#[inline(always)]
pub fn key_and_hash(word: &[u8]) -> (u128, u64) {
    let len = word.len();
    if len > WIDE_KEY_BYTES {
        return key_and_hash_long(word);
    }
    // Wide tier, stitched (no 16 readable bytes to load from): head and tail overlap and the bytes
    // they share are the same bytes, so the `|` cannot lose one.
    if len > INLINE_KEY_BYTES {
        let head = u64::from_le_bytes(word[..8].try_into().unwrap()) as u128;
        let tail = u64::from_le_bytes(word[len - 8..].try_into().unwrap()) as u128;
        let key = (head | tail << (8 * (len - 8))) | (len as u128) << 120;
        return (key, mix128(key));
    }
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
    (key as u128, mix(key))
}

///
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
struct Entry {
    digest: u32,
    /// The token id in the low 31 bits, [`FOLD_BIT`] in the top.
    id: u32,
}

const _: () = assert!(size_of::<Entry>() == 8);

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

#[inline(always)]
fn digest_of(hash: u64) -> u32 {
    (hash >> 32) as u32
}

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
    entries: Box<[Entry]>,
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
            let (_, hash) = key_and_hash(s.as_slice());
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
        let (_, hash) = key_and_hash(q);
        let slot = self.mphf.index(&hash);
        let e = self.entries[slot];
        (e.digest == digest_of(hash)).then_some(e.id & VOCAB_ID_MASK)
    }

    /// The id for `q`, together with whether that entry may be folded. One probe and one entry
    /// load: the flag is a bit of the id the probe already read.
    #[inline]
    pub fn get_bytes_foldable(&self, q: &[u8]) -> Option<(u32, bool)> {
        let (_, hash) = key_and_hash(q);
        self.get_keyed_foldable(hash)
    }

    #[inline(always)]
    pub fn probe_slot(&self, hash: u64) -> usize {
        self.mphf.index(&hash)
    }

    #[inline(always)]
    pub fn entry_at(&self, slot: usize) -> (u32, u32) {
        let e = self.entries[slot];
        (e.digest, e.id)
    }

    #[inline(always)]
    pub fn resolve_foldable(hash: u64, entry: (u32, u32)) -> Option<(u32, bool)> {
        let (edigest, eid) = entry;
        (edigest == digest_of(hash)).then_some((eid & VOCAB_ID_MASK, eid & FOLD_BIT != 0))
    }

    #[inline]
    pub fn get_keyed_foldable(&self, hash: u64) -> Option<(u32, bool)> {
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

    /// The pair a linear `lo ^ hi` fold could not separate: same length, and the only two bytes that
    /// differ sit one in each half at the same offset, so their xor deltas cancelled and a vocabulary
    /// holding one answered a query for the other. This is why `mix128` multiplies.
    #[test]
    fn the_fold_separates_words_whose_halves_cancel() {
        let (dignified, signifies) = (b" dignified", b" signifies");
        assert_ne!(
            key_and_hash(dignified).1,
            key_and_hash(signifies).1,
            "the halves' deltas cancelled -- mix128 must not fold linearly"
        );
        assert_ne!(key_and_hash(dignified).0, key_and_hash(signifies).0);
        let vocab = BucketVocabStore::build(vec![(signifies.to_vec(), 7)]);
        assert_eq!(vocab.get_bytes(signifies), Some(7));
        assert_eq!(vocab.get_bytes(dignified), None);
    }

    /// Every offset, not just the one that bit us: the same bit flipped in byte `i` of each half must
    /// still land on two hashes.
    #[test]
    fn no_paired_byte_delta_cancels() {
        for i in 0..7usize {
            let a = *b"0123456789abcde";
            let mut b = a;
            b[i] ^= 0x17;
            b[i + 8] ^= 0x17;
            assert_ne!(
                key_and_hash(&a).1,
                key_and_hash(&b).1,
                "paired delta at byte {i} cancelled"
            );
        }
    }

    /// A packed key *is* the word, in both tiers, so no two words up to `WIDE_KEY_BYTES` may share
    /// one -- that is what makes a cache hit on a short word exact rather than probabilistic. The two
    /// tiers must not collide with each other either: the cheap one leaves bits 64.. clear, the wide
    /// one always sets a length of at least 8 at bits 120...
    #[test]
    fn both_tiers_pack_words_uniquely() {
        use std::collections::HashMap;
        let mut seen: HashMap<u128, Vec<u8>> = HashMap::new();
        for len in 1..=WIDE_KEY_BYTES {
            for b in [b'a', b'b', 0x00, 0xFF] {
                let word = vec![b; len];
                let key = key_and_hash(&word).0;
                assert_eq!(key & KEY_IS_HASH, 0, "a packed key must not look hashed");
                if let Some(prev) = seen.insert(key, word.clone()) {
                    panic!("{prev:?} and {word:?} packed to the same key");
                }
            }
        }
    }

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
