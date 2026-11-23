use std::collections::HashMap;

use ahash::RandomState;
use ptr_hash::PtrHashParams;
use std::fmt;

use crate::Result;
use crate::utils::perfect_hash::Mphf;
use crate::utils::signature::Signature;
use crate::utils::uncovered::Uncovered;

static LONG_WORD_HASHER: RandomState = RandomState::with_seeds(
    0x243F_6A88_85A3_08D3,
    0x1319_8A2E_0370_7344,
    0xA409_3822_299F_31D0,
    0x082E_FA98_EC4E_6C89,
);

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

/// The [`Mphf`] key of a word.
///
/// A word its signature holds whole takes [`Signature::vocabulary_key`]. A longer word hashes all
/// of its bytes, because two tokens with one signature would otherwise be one key.
#[inline(always)]
fn mphf_key(word: &[u8], signature: Signature) -> u64 {
    if signature.len > Signature::BYTES {
        hash_long(word)
    } else {
        signature.vocabulary_key()
    }
}

// Out of line so that `mphf_key` inlines without ahash. Not `#[cold]`: for CJK this is the common arm.
#[inline(never)]
fn hash_long(word: &[u8]) -> u64 {
    LONG_WORD_HASHER.hash_one(word)
}

/// One MPHF slot, first half: what turns away a word that is not the slot's token.
///
/// The perfect hash sends any word to some token's slot. Whether the word is that token is a
/// branch the CPU cannot predict, and it waits for this entry to load. So the entry is 8 bytes,
/// in a table a third the size of one that held the token's whole [`Signature`]. A word that
/// passes is confirmed against [`Ends`]: see [`BucketVocabStore::get_foldable`].
#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct Entry {
    /// 24 bits of the token's [`Mphf`] key, then its length in the low 8: see [`Entry::check`].
    check: u32,
    /// The token id in the low 31 bits, [`FOLD_BIT`] in the top.
    id: u32,
}

const _: () = assert!(size_of::<Entry>() == 8);

impl Entry {
    /// The length byte of a slot the build never wrote. No word has it, so no word passes.
    const VACANT_LEN: u32 = 0xFF;
    /// The length byte of every token too long for it. Such a token is longer than its
    /// signature, so the lookup compares its bytes and with them its exact length.
    const LONGEST_LEN: usize = 0xFE;

    const VACANT: Self = Self {
        check: Self::VACANT_LEN,
        id: 0,
    };

    #[inline(always)]
    fn check(key: u64, len: usize) -> u32 {
        ((key >> 40) as u32) << 8 | len.min(Self::LONGEST_LEN) as u32
    }
}

/// One MPHF slot, second half: the first and last eight bytes of its token. With the length in
/// [`Entry`], that is the token's [`Signature`].
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
struct Ends {
    head: u64,
    tail: u64,
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
#[derive(Clone)]
pub(crate) struct BucketVocabStore {
    mphf: Mphf,
    /// All token bytes, concatenated. Ordered by MPHF slot.
    bytes: Box<[u8]>,
    entries: Box<[Entry]>,
    ends: Box<[Ends]>,
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

        let signatures: Vec<Signature> = tokens.iter().map(|(s, _)| Signature::new(s)).collect();
        let keys: Vec<u64> = tokens
            .iter()
            .zip(&signatures)
            .map(|((s, _), &signature)| mphf_key(s, signature))
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
        // build loop stay [`Entry::VACANT`] with a default `Span { len: 0, .. }`, which
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
        let mut entries = vec![Entry::VACANT; n_slots];
        let mut ends = vec![Ends::default(); n_slots];
        let mut spans = vec![Span::default(); n_slots];
        let mut id_to_slot = vec![u32::MAX; id_space];
        for (((s, id), &signature), key) in tokens.iter().zip(&signatures).zip(&keys) {
            assert!(
                s.len() <= u16::MAX as usize,
                "token longer than 65535 bytes"
            );
            assert!(
                *id <= VOCAB_ID_MASK,
                "token id {id} needs bit 31, which holds FOLD_BIT"
            );
            let slot = mphf.index(key);
            entries[slot] = Entry {
                check: Entry::check(*key, s.len()),
                id: *id,
            };
            ends[slot] = Ends {
                head: signature.head,
                tail: signature.tail,
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
            ends: ends.into_boxed_slice(),
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
            ends: Box::new([]),
            spans: Box::new([]),
            id_to_slot: Box::new([]),
            n: 0,
        }
    }

    /// The id of the token equal to `word`, like `get` on a `HashMap`.
    #[inline]
    pub(crate) fn get_bytes(&self, word: &[u8]) -> Option<u32> {
        let (id, _) = self.get_foldable(word, Signature::new(word))?;
        Some(id)
    }

    /// The id of the token equal to `word` and whether it folds, for a caller that already holds
    /// the word's signature.
    ///
    /// The perfect hash returns a slot for any word. The word is that slot's token when it passes
    /// the slot's [`Entry::check`], the two have the same first and last eight bytes and, for a
    /// word its signature does not hold whole, the same bytes in between.
    // Forced: an x86-64 build left it out of line, a call per word the cache does not hold.
    #[inline(always)]
    pub(crate) fn get_foldable(&self, word: &[u8], signature: Signature) -> Option<(u32, bool)> {
        if self.entries.is_empty() {
            return None;
        }
        let key = mphf_key(word, signature);
        let slot = self.mphf.index(&key);
        let entry = self.entries[slot];
        if entry.check != Entry::check(key, signature.len) {
            return None;
        }
        // A word that got here is the token, but for one in 2^24 that is not. The CPU predicts
        // this branch, so the second load does not hold the encode loop back.
        let ends = self.ends[slot];
        (ends.head == signature.head
            && ends.tail == signature.tail
            && self.holds_uncovered(slot, word))
        .then_some((entry.id & VOCAB_ID_MASK, entry.id & FOLD_BIT != 0))
    }

    /// Whether the token at `slot`, whose signature is `word`'s, also has its [`Uncovered`]
    /// bytes.
    #[inline]
    fn holds_uncovered(&self, slot: usize, word: &[u8]) -> bool {
        let Some(uncovered) = Uncovered::of(word) else {
            return true;
        };
        let span = self.spans[slot];
        let token = &self.bytes[span.start as usize..][..span.len as usize];
        Uncovered::of(token) == Some(uncovered)
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

    /// The perfect hash sends every word to some slot, so a word that is not a token lands on a
    /// token's. A 16-byte and a 17-byte word, because the longer one is hashed another way.
    #[test]
    fn a_word_that_is_not_a_token_is_not_found() {
        let tokens: [&[u8]; 3] = [b"h", b"sixteen-byte-tok", b"seventeen-byte-tok"];
        let vocab = BucketVocabStore::build(
            tokens
                .iter()
                .zip(0u32..)
                .map(|(token, id)| (token.to_vec(), id))
                .collect(),
        )
        .unwrap();
        for i in 0..1000u32 {
            for word in [format!("w{i}"), format!("{i:016}"), format!("{i:017}")] {
                assert_eq!(vocab.get_bytes(word.as_bytes()), None, "{word:?}");
            }
        }
    }

    /// A vocabulary of the one `token`, where `word` lands on the token's entry under the word's
    /// own [`Entry::check`]. One word in 2^24 does on a real vocabulary: too rare to find, so
    /// the token is copied to the word's slot.
    fn with_token_on_the_slot_of(token: &[u8], word: &[u8]) -> BucketVocabStore {
        let mut vocab = BucketVocabStore::build(vec![(token.to_vec(), 0)]).unwrap();
        let token_slot = vocab.id_to_slot[0] as usize;
        let key = mphf_key(word, Signature::new(word));
        let word_slot = vocab.mphf.index(&key);
        vocab.entries[word_slot] = Entry {
            check: Entry::check(key, word.len()),
            id: 0,
        };
        vocab.ends[word_slot] = vocab.ends[token_slot];
        vocab.spans[word_slot] = vocab.spans[token_slot];
        vocab
    }

    /// A word past [`Signature::BYTES`] with a token's length, start and end has that token's
    /// signature: only the bytes in between tell the two apart. The word differs from the token
    /// in one of them, each in turn, at the lengths [`Uncovered`] changes shape.
    #[test]
    fn a_word_with_a_tokens_signature_and_check_is_not_found() {
        for len in [17, 23, 24, 32, 33, 48, 49] {
            let token: Vec<u8> = (0..len as u8).collect();
            for at in 8..len - 8 {
                let mut word = token.clone();
                word[at] ^= 0x80;
                assert_eq!(Signature::new(&token), Signature::new(&word));
                let vocab = with_token_on_the_slot_of(&token, &word);
                assert_eq!(vocab.get_bytes(&word), None, "len {len} byte {at}");
            }
        }
    }

    /// [`Entry::check`] holds one length for every token of 254 bytes or more. A word one byte
    /// longer than such a token, with its start and end, is told apart by its bytes alone.
    #[test]
    fn a_word_one_byte_longer_than_a_long_token_is_not_found() {
        let token = [&b"01234567"[..], &[b'm'; 284], b"89abcdef"].concat();
        let word = [&b"01234567"[..], &[b'm'; 285], b"89abcdef"].concat();
        let vocab = BucketVocabStore::build(vec![(token.clone(), 0)]).unwrap();
        assert_eq!(vocab.get_bytes(&token), Some(0));
        // Looked up before the copy: where the word lands on the token's own slot, the copy
        // takes the token's check away.
        let vocab = with_token_on_the_slot_of(&token, &word);
        assert_eq!(vocab.get_bytes(&word), None);
    }

    /// A slot the build never wrote holds no token, and its zeroed [`Ends`] are those of a
    /// word of zero bytes. Its length byte is what keeps every word out, whatever the key.
    #[test]
    fn no_word_passes_the_check_of_a_vacant_slot() {
        for key in [0, 1 << 40, u64::MAX] {
            for len in [
                0,
                1,
                Signature::BYTES,
                Entry::LONGEST_LEN,
                0xFF,
                0x1FF,
                1 << 20,
            ] {
                assert_ne!(
                    Entry::check(key, len),
                    Entry::VACANT.check,
                    "key {key:#x} len {len}"
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
}
