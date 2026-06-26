use std::collections::HashSet;

use ahash::RandomState;
use ptr_hash::bucket_fn::Linear;
use ptr_hash::{PtrHash, PtrHashParams};
use std::fmt;

use crate::models::bpe::Vocab;

type Mphf = PtrHash<u64, Linear>;

// Fixed seeds so a given vocab always hashes identically (the hasher is also stored on the struct,
// so build and query are guaranteed consistent regardless).
const SEEDS: [u64; 4] = [
    0x243F_6A88_85A3_08D3,
    0x1319_8A2E_0370_7344,
    0xA409_3822_299F_31D0,
    0x082E_FA98_EC4E_6C89,
];

#[derive(Clone, Copy, Debug)]
struct Entry {
    start: u32,
    len: u16,
    id: u32,
}

/// The VocabStore optimizes for space and speed. We don't use a HashMap to prevent duplicating the
/// keys. Instead, we just use an `id_to_slot` and `entries` table. When you query bytes, you hash
/// on the fly and get an `index` into the `entries` table. When you query an `id`, you fetch in
/// the `id_to_slot` the same index.
///
/// Entries store start, len and the actual `id` of the token.
/// Example:
///
/// ```
/// use tk-encode::tokenizer::vocab_store::VocabStore;
/// let vocab = VocabStore::build(vec![
///     (b"a".to_vec(), 0),
///     (b"bb".to_vec(), 5),
///     (b"ccc".to_vec(), 100),
/// ]);
/// vocab.token_to_id("a".to_string());
/// vocab.id_to_token(100);
#[derive(Clone)]
pub struct VocabStore {
    mphf: Mphf,
    hasher: RandomState,
    /// All token bytes, concatenated. Ordered by MPHF slot.
    bytes: Box<[u8]>,
    /// `entries[slot]` -> (offset into `bytes`, length, id). Ordered by MPHF slot.
    entries: Box<[Entry]>,
    /// `id_to_slot[id]` -> MPHF slot, or `u32::MAX` for ids absent from the vocab.
    id_to_slot: Box<[u32]>,
}

impl fmt::Debug for VocabStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VocabStore")
            .field("bytes", &self.bytes)
            .field("id_to_slot", &self.id_to_slot)
            .field("entries", &self.entries)
            .finish()
    }
}
impl VocabStore {
    pub fn build(tokens: Vec<(Vec<u8>, u32)>) -> Self {
        let n = tokens.len();

        let hasher = RandomState::with_seeds(SEEDS[0], SEEDS[1], SEEDS[2], SEEDS[3]);

        // 1. Pre-hash token bytes -> u64 keys using near perfect hash func
        let keys: Vec<u64> = tokens
            .iter()
            .map(|(s, _)| hasher.hash_one(s.as_slice()))
            .collect();

        // 2. A minimal perfect hash needs distinct keys. Collisions are astronomically unlikely
        //    (~n^2/2^65); if one ever fires, switch the key type to u128. The byte check below makes
        //    a collision a correct miss at query time, but it would drop a token at build, so guard.
        {
            let mut seen = HashSet::with_capacity(n);
            for k in &keys {
                assert!(
                    seen.insert(*k),
                    "64-bit hash collision in vocab; rebuild with u128 keys"
                );
            }
        }

        // 3. Build the MPHF. `single_part = true` to use the faster `index_single_part` query path.
        let mut params = PtrHashParams::default_fast();
        params.single_part = true;
        let mphf = Mphf::new(&keys, params);

        // 4. Place each token at its MPHF slot; build the slab and the id->slot reverse table.
        let total: usize = tokens.iter().map(|(s, _)| s.len()).sum();
        let max_id = tokens.iter().map(|(_, id)| *id).max().unwrap();
        let mut bytes = Vec::with_capacity(total);
        let mut entries = vec![
            Entry {
                start: 0,
                len: 0,
                id: 0
            };
            n
        ];
        let mut id_to_slot = vec![u32::MAX; max_id as usize + 1];
        for (s, id) in &tokens {
            assert!(
                s.len() <= u16::MAX as usize,
                "token longer than 65535 bytes"
            );
            let slot = mphf.index_single_part(&hasher.hash_one(s.as_slice()));
            entries[slot] = Entry {
                start: bytes.len() as u32,
                len: s.len() as u16,
                id: *id,
            };
            id_to_slot[*id as usize] = slot as u32;
            bytes.extend_from_slice(s);
        }

        Self {
            mphf,
            hasher,
            bytes: bytes.into_boxed_slice(),
            entries: entries.into_boxed_slice(),
            id_to_slot: id_to_slot.into_boxed_slice(),
        }
    }

    #[inline]
    pub fn get_bytes(&self, q: &[u8]) -> Option<u32> {
        let slot = self.mphf.index_single_part(&self.hasher.hash_one(q));
        let e = self.entries[slot];
        let (start, len) = (e.start as usize, e.len as usize);
        // Byte equality: confirms `q` really is the token at this slot (perfect hashing only
        // guarantees a valid slot for in-vocab keys; this rejects collisions and OOV queries).
        if len == q.len() && self.bytes[start..start + len] == *q {
            Some(e.id)
        } else {
            None
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
        self.id_to_token_bytes(id)
            .map(|b| String::from_utf8_lossy(b).into_owned())
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn get_vocab(&self) -> Vec<(String, u32)> {
        self.entries
            .into_iter()
            .filter_map(|m| self.id_to_token(m.id).map(|token| (token, m.id)))
            .collect()
    }

    /// start and end are index into `self.entry`.
    #[inline]
    pub fn match_bytes(&self, bytes: &[u8], start: u32, end: u32) -> Option<u32> {
        let mut i = 0;
        assert!(start < self.entries.len() as u32);
        for i in self.entries[start as usize].start..self.entries[end as usize].start - 1 {
            // we take each entry from byte_offset to end, compare with memcp
            let e = self.entries[i as usize];
            let slice = &self.bytes[e.start as usize..(e.start + e.len as u32) as usize];
            if bytes.starts_with(slice) {
                return Some(e.id);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_token() {
        let vocab = VocabStore::build(vec![(b"Hel".to_vec(), 0)]);
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
        let vocab = VocabStore::build(toks.clone());

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
        let vocab = VocabStore::build(vec![
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
    fn test_match_bytes() {
        let vocab = VocabStore::build(vec![
            (b"ccci".to_vec(), 0),
            (b"cc".to_vec(), 5),
            (b"isn".to_vec(), 100),
        ]);
        assert!(vocab.match_bytes("cccisnot the best".as_bytes(), 0, 2) == Some(0));
        assert!(vocab.match_bytes("snot the best".as_bytes(), 0, 2) == None);
    }
}
