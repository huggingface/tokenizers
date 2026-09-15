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

pub(crate) const INLINE_KEY_BYTES: usize = 7;

#[inline(always)]
fn mix(z: u64) -> u64 {
    let z = z.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z ^ (z >> 29)
}

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
pub fn key_and_hash_readable(word: &[u8], readable: usize) -> (u64, u64) {
    let len = word.len();
    if len > INLINE_KEY_BYTES || readable < 8 {
        return key_and_hash(word);
    }
    // SAFETY: `readable >= 8` bytes exist from `word.as_ptr()`, and `len <= 7 < 8`.
    // `from_le_bytes` rather than a `cast::<u64>()` reinterpret, because the key layout is
    // little-endian: `key_and_hash` builds it with `u32::from_le_bytes`. A native load on a
    // big-endian host would keep `word[8-len..8]` instead of `word[0..len]` once masked, and
    // put `LEN_TAG` over `word[0]`.
    let raw = u64::from_le_bytes(unsafe { word.as_ptr().cast::<[u8; 8]>().read_unaligned() });
    // SAFETY: `len <= INLINE_KEY_BYTES == 7`, and both tables have 8 entries.
    let (mask, tag) = unsafe { (*KEY_MASK.get_unchecked(len), *LEN_TAG.get_unchecked(len)) };
    let key = (raw & mask) | tag;
    debug_assert_eq!(
        key,
        key_and_hash(word).0,
        "masked load must match the stitched pack"
    );
    (key, mix(key))
}

/// A word too long to pack into the key: hash it for real. Kept out of line so that the packing
/// path above and in [`key_and_hash`] can be inlined without dragging ahash in with them. Not
/// `#[cold]` on purpose -- for CJK this is the *common* arm, and telling the predictor otherwise
/// would cost more than the call.
#[inline(never)]
fn key_and_hash_long(word: &[u8]) -> (u64, u64) {
    let hash = KEY_HASHER.hash_one(word);
    (hash, hash)
}

#[inline(always)]
pub fn key_and_hash(word: &[u8]) -> (u64, u64) {
    let len = word.len();
    if len > INLINE_KEY_BYTES {
        return key_and_hash_long(word);
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
    (key, mix(key))
}

/// One MPHF slot: eight bytes, so a probe is a single load.
///
/// The hash picks the slot and `digest` confirms it -- the key itself is never stored, which is why
/// a query cannot tell a collision from a hit on its own and does not need to (see [`BucketVocabStore::get_bytes`]).
/// A slot the build never wrote stays [`Default`], and occupancy is read off the parallel `spans`.
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
            assert!(
                *id <= VOCAB_ID_MASK,
                "token id {id} needs bit 31, which holds FOLD_BIT"
            );
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
        let (key, hash) = key_and_hash(q);
        self.get_keyed_foldable(key, hash)
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

    /// Whether every entry carries the fold bit.
    ///
    /// A BPE model built with `ignore_merges` gets the bit on all of them unconditionally, where
    /// otherwise only the entries that proved they reduce to themselves earn it. That makes this
    /// the question a writer asks to decide whether to spell the flag: if every entry folds, either
    /// answer rebuilds the same table, and if any does not, the flag was off.
    pub fn all_foldable(&self) -> bool {
        self.entries
            .iter()
            .zip(self.spans.iter())
            .filter(|(_, sp)| sp.len > 0)
            .all(|(e, _)| e.id & FOLD_BIT != 0)
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

    // The key is a little-endian pack of the word with its length on top -- not whatever a `u64`
    // load of those bytes happens to give. Built with shifts here so the expected value is the
    // same on a big-endian target.
    #[test]
    fn inline_key_is_little_endian() {
        let words: [&[u8]; 8] = [
            b"",
            b"a",
            b" the",
            b"hello",
            b"abcdefg",
            &[0x00],
            &[0xff, 0x00],
            &[0x80, 0x7f, 0x01, 0xfe, 0x00, 0x11, 0xc0],
        ];
        for word in words {
            let mut expect = (word.len() as u64) << 56;
            for (i, b) in word.iter().enumerate() {
                expect |= (*b as u64) << (8 * i);
            }
            assert_eq!(key_and_hash(word), (expect, mix(expect)), "{word:?}");
        }
    }

    // The fast path masks one 8-byte load instead of stitching the word together, so both have to
    // land on the same key. On big-endian a native load gives a different one: the mask keeps the
    // wrong end of the word and the length tag lands on `word[0]`.
    #[test]
    fn fast_path_matches_stitched_pack() {
        // Words are slices of a bigger buffer -- that is what makes 8 bytes readable past the end.
        let buf: &[u8] = b"\x00\xffhello \xc0\xc1world\x80\x7f\x01\xfe\x00 the quick\xff";
        for start in 0..8 {
            for len in 0..=INLINE_KEY_BYTES + 3 {
                let word = &buf[start..start + len];
                let want = key_and_hash(word);
                let readable = buf.len() - start;
                assert_eq!(
                    key_and_hash_readable(word, readable),
                    want,
                    "start {start} len {len}"
                );
                // Under 8 readable bytes the load would run off the end, so it must not happen.
                assert_eq!(
                    key_and_hash_readable(word, len),
                    want,
                    "start {start} len {len}"
                );
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
