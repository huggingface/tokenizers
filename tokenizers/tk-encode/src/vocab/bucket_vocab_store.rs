use std::collections::HashMap;
use std::sync::LazyLock;

use ahash::RandomState;
use ptr_hash::PtrHashParams;
use std::fmt;
use std::hash::{BuildHasher, Hasher};

use crate::Result;
use crate::utils::perfect_hash::{Mphf, mix};

/// Keyed per process, from the OS: with a published seed an attacker picks *both* sides of a
/// collision (a token in a vocabulary they ship, and an input they ship) and finds a pair in 2^32
/// by birthday. A fresh seed makes that 2^64 again. Nothing persists a [`BucketVocabStore`], so a
/// seed that changes per process costs nothing.
static KEY_HASHER: LazyLock<RandomState> = LazyLock::new(|| {
    // Test-only: `TK_HASH_SEED=k` pins the seed, so a test that depends on where a word lands in the
    // table reproduces instead of flaking at 1 run in 60.
    #[cfg(test)]
    if let Ok(seed) = std::env::var("TK_HASH_SEED") {
        let s: u64 = seed.parse().unwrap_or(0);
        return RandomState::with_seeds(s | 1, s ^ 0x9e37, s ^ 0x7f4a, s ^ 0x1234_5678);
    }
    RandomState::new()
});

#[inline(always)]
fn key_hasher() -> &'static RandomState {
    &KEY_HASHER
}

/// Bit 31 of a stored id: the token provably encodes to itself, so a pretoken equal to it can be
/// emitted without running the merge loop. See `BPE::prove_fold`.
///
/// Packed into the id rather than kept beside it, following the same convention as a packed merge
/// value, where the low bits are the product id and the high bits are a flag field (`SAFE_MASK`).
/// The lookup already loads this entry to verify the key, so the flag costs no memory, no extra
/// load, and no second structure to keep in step with the vocabulary.
const FOLD_BIT: u32 = 1 << 31;
/// The id half. 2^31 ids is far past any vocabulary.
const VOCAB_ID_MASK: u32 = FOLD_BIT - 1;

pub(crate) const INLINE_KEY_BYTES: usize = 7;

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
pub(crate) fn key_and_hash_readable(word: &[u8], readable: usize) -> (u64, u64) {
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
    // Arithmetic instead of the two 8-entry tables: one `bzhi`/shift pair on the low bytes and the
    // length in the top byte. `len <= 7`, so `8 * len <= 56` and the shift is defined.
    let key = (raw & ((1u64 << (8 * len)) - 1)) | (len as u64) << 56;
    debug_assert_eq!(
        key,
        key_and_hash(word).0,
        "masked load must match the stitched pack"
    );
    (key, mix(key))
}

/// A word too long to pack into the key: hash it for real.
///
/// `#[inline]`, not `#[inline(never)]`: outlining this cost 0.9 ns per word (measured, lookup bench),
/// because the caller is a per-pre-token loop with nothing to overlap a call+ret with. Not `#[cold]`
/// either -- for CJK this is the *common* arm.
#[inline]
fn key_and_hash_long(word: &[u8]) -> (u64, u64) {
    // `hash_one(&[u8])` runs `Hash for [u8]`: a length prefix, then the bytes -- two ahash rounds.
    // The length is already in the digest (top byte), so the prefix is redundant here.
    let mut hasher = key_hasher().build_hasher();
    hasher.write(word);
    let hash = hasher.finish();
    // The length goes in the top byte, so comparing digests *is* comparing lengths: no span load,
    // no extra compare. `Entry.digest` also cannot be shared across the two arms however the hash is
    // inverted (#2722), because a packed key's top byte is at most 7 and these are 8..254. `0xFF` is
    // the escape for a token too long to fit in a byte, which keeps the same 56 bits of verifier
    // without a length.
    // `min`: the escape byte is the same value the length caps at, and it lowers to one `csel`.
    let tag = word.len().min(0xFF) as u64;
    (tag << 56 | (hash & 0x00FF_FFFF_FFFF_FFFF), hash)
}

#[inline(always)]
pub(crate) fn key_and_hash(word: &[u8]) -> (u64, u64) {
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
    // The length goes in the top byte, which the packing leaves free for every length <= 7.
    (raw | (len as u64) << 56, mix(raw | (len as u64) << 56))
}

/// One MPHF slot: sixteen bytes, so a probe is a single load.
///
/// The hash picks the slot and the stored key confirms it. The stored key is the *whole* 64-bit
/// key, not a fold of it: a 32-bit digest gives a false hit every 2^32 lookups, which is reachable
/// by search in seconds (see huggingface/tokenizers#2722), and comparing the other 32 bits costs
/// nothing -- it is the same load and the same one-instruction compare.
///
/// The key, not the hash: `key_and_hash` returns `mix(key)` for a word of at most
/// [`INLINE_KEY_BYTES`] bytes, and `mix` is a published bijection over 2^64 (`unmix` is in the test
/// below), so a hash comparison can be inverted into a *different* short word - one long token in
/// 256 has a hash that lands in the 2^56-sized image of the packed arm. The packed key itself is
/// injective per arm, and the arms' key spaces are disjoint by construction, so comparing keys makes
/// equal digests mean equal words.
///
/// A slot the build never wrote stays [`Default`], and occupancy is read off the parallel `spans`.
///
/// TODO: let's detail the single load design, the key mask, the INLINE_KEY_BYTES etc
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
struct Entry {
    digest: u64,
    /// The token id in the low 31 bits, [`FOLD_BIT`] in the top.
    id: u32,
}

// 16 bytes is still one cache line fetch, so the wider digest is free at probe time.
const _: () = assert!(size_of::<Entry>() == 16);

/// The layout `key_and_hash` builds, spelled out for the tests to check against: the low `len` bytes
/// of the word (`KEY_MASK[len]` of them), with `len` itself in the top byte.
#[cfg(test)]
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
#[cfg(test)]
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
#[derive(Clone)]
pub(crate) struct BucketVocabStore {
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
    pub(crate) fn build(tokens: Vec<(Vec<u8>, u32)>) -> Result<Self> {
        let n = tokens.len();

        let keys: Vec<u64> = tokens
            .iter()
            .map(|(s, _)| key_and_hash(s.as_slice()).1)
            .collect();

        let mut unique_entries: HashMap<u64, usize> = HashMap::with_capacity(n);

        for (index, hash) in keys.iter().enumerate() {
            let Some(&previous_index) = unique_entries.get(hash) else {
                unique_entries.insert(*hash, index);
                continue;
            };
            let ((previous_bytes, previous_id), (current_bytes, current_id)) =
                (&tokens[previous_index], &tokens[index]);

            // Hash collision: two different byte strings resolve to the same hash
            if previous_bytes != current_bytes {
                return Err(format!(
                    "failed to build the vocabulary store: 64-bit hash collision for tokens \
                     {:?} (id={previous_id}) and {:?} (id={current_id}).",
                    String::from_utf8_lossy(previous_bytes),
                    String::from_utf8_lossy(current_bytes)
                )
                .into());
            }

            // Duplicate entries in the vocabulary: warn and continue
            warn!(
                "duplicate entries in the tokenizer vocabulary for token {:?}. \
                The latest occurrence (token id {current_id}) will take precedence.",
                String::from_utf8_lossy(current_bytes)
            );
        }

        // 3. Build the (non-minimal) `FastPtrHash` via `PtrHashParams::default_fast()`; query with `.index()`.
        let params = PtrHashParams::default_fast();
        let mphf = Mphf::new(&unique_entries.into_keys().collect::<Vec<u64>>(), params);

        // FastPtrHash is non-minimal: `index()` may return a slot up to `max_index()` (>= n),
        // so `entries` must be sized to cover the whole slot range. Slots never written by the
        // build loop stay as the default `Entry { len: 0, .. }` (phantom/padding slots), which
        // enumeration/count paths filter out via `len > 0`.
        let n_slots = mphf.max_index();

        // 4. Place each token at its MPHF slot; build the slab and the id->slot reverse table.
        let total: usize = tokens.iter().map(|(s, _)| s.len()).sum();
        let id_space = tokens
            .iter()
            .map(|(_, id)| *id as usize + 1)
            .max()
            .unwrap_or(0);
        let mut bytes = Vec::with_capacity(total);
        let mut entries = vec![Entry::default(); n_slots];
        let mut spans = vec![Span::default(); n_slots];
        let mut id_to_slot = vec![u32::MAX; id_space];
        for (s, id) in &tokens {
            assert!(
                s.len() <= u16::MAX as usize,
                "token longer than 65535 bytes"
            );
            assert!(
                *id <= VOCAB_ID_MASK,
                "token id {id} needs bit 31, which holds FOLD_BIT"
            );
            let (key, hash) = key_and_hash(s.as_slice());
            let slot = mphf.index(&hash);
            entries[slot] = Entry {
                digest: key,
                id: *id,
            };
            spans[slot] = Span {
                start: bytes.len() as u32,
                len: s.len() as u16,
            };
            id_to_slot[*id as usize] = slot as u32;
            bytes.extend_from_slice(s);
        }

        Ok(Self {
            mphf,
            bytes: bytes.into_boxed_slice(),
            entries: entries.into_boxed_slice(),
            spans: spans.into_boxed_slice(),
            id_to_slot: id_to_slot.into_boxed_slice(),
            n,
        })
    }

    pub(crate) fn new() -> Self {
        // convenient to build empty edit later.
        let empty: [u64; 0] = [];
        Self {
            mphf: Mphf::new(&empty, PtrHashParams::default_fast()),
            bytes: Box::new([]),
            entries: Box::new([]),
            spans: Box::new([]),
            id_to_slot: Box::new([]),
            n: 0,
        }
    }

    /// This function is the equivalent of `get` on a HashaMap, it return the id
    /// corresponding to the key `q`. Since `mphf` always return a slot, we check
    /// whether the token indexed by that slot actually match the query: the stored
    /// digest is the query's own 64-bit key, and `key_and_hash` is injective over
    /// words, so equal digests mean equal words - no span load, no memcmp.
    #[inline]
    pub(crate) fn get_bytes(&self, q: &[u8]) -> Option<u32> {
        if self.entries.is_empty() {
            return None;
        }
        let (key, hash) = key_and_hash(q);
        let slot = self.mphf.index(&hash);
        let e = self.entries[slot];
        if e.digest != key {
            return None;
        }
        Some(e.id & VOCAB_ID_MASK)
    }

    #[inline]
    pub(crate) fn get_keyed_foldable(&self, key: u64, hash: u64) -> Option<(u32, bool)> {
        if self.entries.is_empty() {
            return None;
        }
        let slot = self.mphf.index(&hash);
        let e = self.entries[slot];
        // Same check as `get_bytes`, on the key the caller already has.
        (e.digest == key).then_some((e.id & VOCAB_ID_MASK, e.id & FOLD_BIT != 0))
    }

    /// Records that this token folds to itself. Called once per entry at load, after the proof.
    pub(crate) fn set_foldable(&mut self, id: u32) {
        if let Some(&slot) = self.id_to_slot.get(id as usize)
            && slot != u32::MAX
        {
            self.entries[slot as usize].id |= FOLD_BIT;
        }
    }

    #[inline]
    pub(crate) fn token_to_id(&self, s: &str) -> Option<u32> {
        self.get_bytes(s.as_bytes())
    }

    /// `id -> token bytes`, borrowing from the slab (no allocation).
    #[inline]
    pub(crate) fn id_to_token_bytes(&self, id: u32) -> Option<&[u8]> {
        let slot = *self.id_to_slot.get(id as usize)?;
        if slot == u32::MAX {
            return None; // id is within range but absent from the vocab
        }
        let sp = self.spans[slot as usize];
        let start = sp.start as usize;
        self.bytes.get(start..start + sp.len as usize)
    }

    #[inline]
    pub(crate) fn id_to_token(&self, id: u32) -> Option<String> {
        // we are not sure its a valid utf8 so if not, adds replacement char
        self.id_to_token_bytes(id)
            .map(|b| String::from_utf8_lossy(b).into_owned())
    }

    pub(crate) fn len(&self) -> usize {
        self.n
    }

    /// One past the highest id this vocabulary can hold.
    ///
    /// Ids are not dense: a config may leave gaps, so [`Self::len`] counts entries and is *not* an
    /// id bound. Anything that walks ids has to bound itself by this and skip the holes, which
    /// [`Self::id_to_token_bytes`] reports as `None`.
    pub(crate) fn id_space(&self) -> usize {
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
    pub(crate) fn all_foldable(&self) -> bool {
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

    /// convenient when we want to re-build a vocab
    pub(crate) fn byte_content(&self) -> Vec<(Vec<u8>, u32)> {
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
    use crate::utils::perfect_hash::splitmix64;

    // Six keys used to get exactly six slots, and this set has no placement in them on any seed.
    #[test]
    fn six_tokens_build() {
        let tokens = ["<s>", "<unk>", "<bos>", "[CLS]", "[SEP]", "[MASK]"];
        let vocab = BucketVocabStore::build(
            tokens
                .iter()
                .zip(0u32..)
                .map(|(token, id)| (token.as_bytes().to_vec(), id))
                .collect(),
        )
        .unwrap();
        for (token, id) in tokens.iter().zip(0u32..) {
            assert_eq!(vocab.token_to_id(token), Some(id));
        }
    }

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
        let vocab = BucketVocabStore::build(vec![(b"Hel".to_vec(), 0)]).unwrap();
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
        let vocab = BucketVocabStore::build(toks.clone()).unwrap();

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
        ])
        .unwrap();
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
    fn builds_from_no_tokens() {
        let vocab = BucketVocabStore::build(Vec::new()).unwrap();
        assert_eq!(vocab, BucketVocabStore::new());
        assert_eq!(vocab.id_to_token(0), None);
    }

    #[test]
    fn eq_matches_on_dense_content() {
        // Models use dense ids (0..n); equality must reflect the token set on that range.
        let a = BucketVocabStore::build(vec![(b"x".to_vec(), 0), (b"y".to_vec(), 1)]).unwrap();
        let b = BucketVocabStore::build(vec![(b"y".to_vec(), 1), (b"x".to_vec(), 0)]).unwrap();
        let c = BucketVocabStore::build(vec![(b"x".to_vec(), 0), (b"z".to_vec(), 1)]).unwrap();
        let d = BucketVocabStore::build(vec![(b"x".to_vec(), 0)]).unwrap();
        assert_eq!(a, b); // insertion order does not matter
        assert_ne!(a, c); // different token at id 1
        assert_ne!(a, d); // different length
    }

    #[test]
    fn content_views_agree() {
        let vocab =
            BucketVocabStore::build(vec![(b"hi".to_vec(), 0), (b"yo".to_vec(), 1)]).unwrap();
        let mut got = vocab.content();
        got.sort();
        assert_eq!(got, vec![("hi".to_string(), 0), ("yo".to_string(), 1)]);
        let mut bytes = vocab.byte_content();
        bytes.sort();
        assert_eq!(bytes, vec![(b"hi".to_vec(), 0), (b"yo".to_vec(), 1)]);
    }

    #[test]
    fn duplicate_bytes_in_vocab() {
        let vocab =
            BucketVocabStore::build(vec![(b"  ".to_vec(), 245), (b"  ".to_vec(), 50276)]).unwrap();
        assert_eq!(vocab.get_bytes(b"  "), Some(50276));
        assert_eq!(vocab.id_to_token_bytes(245), Some(&b"  "[..]));
        assert_eq!(vocab.id_to_token_bytes(50276), Some(&b"  "[..]));
    }

    #[test]
    fn non_utf8_token_is_lossy_on_string_but_exact_on_bytes() {
        let raw = vec![0xffu8, 0xfe];
        let vocab = BucketVocabStore::build(vec![(raw.clone(), 0)]).unwrap();
        assert_eq!(vocab.get_bytes(&raw), Some(0));
        assert_eq!(vocab.id_to_token_bytes(0), Some(raw.as_slice()));
        assert_eq!(vocab.id_to_token(0), Some("\u{fffd}\u{fffd}".to_string()));
    }

    /// huggingface/tokenizers#2722: a word whose *32-bit* digest matches an unrelated token used to be
    /// accepted as that token, because the query path verified only `digest_of(hash)`.
    ///
    /// This fixture is not searched, it is derived: `mix` is a bijection and `unmix = mix^-1` is
    /// `x ^ (x>>29) ^ (x>>58)`, then `* C^-1 mod 2^64`. So for any token `t` (<= 7 bytes, where
    /// every byte string is a valid key) you pick the high half of `mix(key(t))`, walk the low half
    /// until `unmix` lands on a 7-byte key that the MPHF sends to the same slot, and you have a
    /// collision -- 555 inversions for this vocabulary, and about `256 * vocab_size` for any real
    /// one. That is why a 32-bit digest cannot be the verifier, even if nobody searches 2^32.
    #[test]
    fn forged_32_bit_digest_collision_is_rejected() {
        let toks: [&[u8]; 6] = [b"h", b"e", b"l", b"o", b"hello", b"tokeniz"];
        let vocab = BucketVocabStore::build(
            toks.iter()
                .enumerate()
                .map(|(i, t)| (t.to_vec(), i as u32))
                .collect(),
        )
        .unwrap();

        let query: &[u8] = &[170, 103, 222, 207, 37, 184, 110];
        // The fixture must still collide on what the old digest was -- `digest_of(hash)`, the high
        // half of `mix(key)` -- or this test stops testing anything.
        let (qkey, qhash) = key_and_hash(query);
        let (_, thash) = key_and_hash(b"tokeniz");
        let slot = vocab.mphf.index(&qhash);
        assert_eq!(vocab.mphf.index(&thash), slot);
        assert_eq!(
            (thash >> 32) as u32,
            (qhash >> 32) as u32,
            "fixture no longer collides on the 32-bit digest: regenerating it needs unmix"
        );
        // Same length as the token it collides with, so a length check does not save us either.
        assert_eq!(query.len(), vocab.spans[slot].len as usize);

        assert_eq!(vocab.get_bytes(query), None);
        assert_eq!(vocab.get_keyed_foldable(qkey, qhash), None);
        // ...while the token it used to be confused with still resolves.
        assert_eq!(vocab.get_bytes(b"tokeniz"), Some(5));
    }

    const C: u64 = 0x9E37_79B9_7F4A_7C15;
    /// `C^-1 mod 2^64`, so that `mix` can be undone.
    const C_INV: u64 = 0xf1de_83e1_9937_733d;

    /// `mix` is a *published* bijection: four instructions take a hash back to its key.
    fn unmix(y: u64) -> u64 {
        assert_eq!(C.wrapping_mul(C_INV), 1);
        (y ^ (y >> 29) ^ (y >> 58)).wrapping_mul(C_INV)
    }

    /// The word a hash would have to come from, if that hash is in the packed arm's image.
    fn short_word_from_hash(hash: u64) -> Option<Vec<u8>> {
        let key = unmix(hash);
        let len = (key >> 56) as usize;
        if len > INLINE_KEY_BYTES || key & !(KEY_MASK[len] | LEN_TAG[len]) != 0 {
            return None;
        }
        let raw = key & !LEN_TAG[len];
        let word: Vec<u8> = (0..len).map(|i| (raw >> (8 * i)) as u8).collect();
        (key_and_hash(&word).0 == key).then_some(word)
    }

    /// huggingface/tokenizers#2722, the other half: a 64-bit digest does not by itself reject a forged
    /// query. The packed arm's hashes are `mix(packed key)`, a bijection onto 2^56 of 2^64, so about
    /// one long token in 256 has a hash `unmix` turns into a *different*, <= 7-byte word -- no search.
    /// Comparing the key is what rejects it: the arms' key spaces are disjoint.
    /// The long arm writes the bytes directly, without `Hash for [u8]`'s length prefix. That is only
    /// safe if ahash still separates words that differ by trailing bytes -- a vocabulary holding both
    /// `"foo"` and `"foo\0"` must not look like a hash collision.
    #[test]
    fn the_direct_write_still_separates_trailing_bytes() {
        for w in [
            &b"foo"[..],
            b"the-filler-word",
            b"a-word-past-fifteen-bytes",
        ] {
            let mut padded = w.to_vec();
            for _ in 1..=3 {
                padded.push(0);
                assert_ne!(
                    key_and_hash(w).1,
                    key_and_hash(&padded).1,
                    "{w:?} vs {padded:?}"
                );
            }
            let mut longer = w.to_vec();
            longer.push(b'x');
            assert_ne!(key_and_hash(w).1, key_and_hash(&longer).1);
        }
    }

    #[test]
    fn short_query_cannot_alias_a_long_token() {
        let mut tokens: Vec<Vec<u8>> = Vec::new();
        let mut x = 0x1234_5678_9abc_def0u64;
        for _ in 0..5_000 {
            x = splitmix64(x);
            let len = 8 + (x % 9) as usize;
            let mut y = x;
            let mut t = Vec::new();
            for _ in 0..len {
                y = splitmix64(y);
                t.push(b'a' + (y % 26) as u8);
            }
            tokens.push(t);
        }
        let vocab = BucketVocabStore::build(
            tokens
                .iter()
                .enumerate()
                .map(|(i, t)| (t.clone(), i as u32))
                .collect(),
        )
        .unwrap();

        let mut forged = 0;
        for (id, t) in tokens.iter().enumerate() {
            let (key, hash) = key_and_hash(t);
            // Every design keeps the long arm's top byte >= 8, which packed keys cannot reach.
            assert!(
                key >> 56 >= 8,
                "long key {key:#x} can be spelled by a <= 7-byte word"
            );
            let Some(short) = short_word_from_hash(hash) else {
                continue;
            };
            forged += 1;
            assert_ne!(short, *t);
            assert_eq!(
                vocab.mphf.index(&key_and_hash(&short).1),
                vocab.mphf.index(&hash)
            );
            assert_eq!(
                vocab.get_bytes(&short),
                None,
                "{short:?} aliased {t:?} (id {id})"
            );
        }
        // The aliases are rare but far too common to be a coincidence: 1/256 of long tokens.
        assert!(
            forged > 5,
            "only {forged} aliases derived: the fixture stopped exercising the bug"
        );
    }
}
