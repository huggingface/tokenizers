use hashbrown::HashMap;

use crate::token::TokenId;

/// We [`Box`] here to avoid the overhead of a [`Vec`]. Data is immutable and static once loaded, so a
/// Box is sufficient
struct VocabStore {
    /// full byte content of the vocab
    blob: Box<[u8]>,
    /// token i's bytes defined by blob[offsets[i]..offsets[i+1]]
    offsets: Box<[u32]>,
    /// byte -> id
    lookup: HashMap<Box<[u8]>, TokenId>,
}

impl VocabStore {
    pub fn id_to_bytes(&self, id: TokenId) -> Option<&[u8]> {
        let id = id as usize;
        let (s, e) = (self.offsets[id] as usize, self.offsets[id + 1] as usize);
        self.blob.get(s..e)
    }

    pub fn token_to_id(&self, token: &[u8]) -> Option<TokenId> {
        self.lookup.get(token).copied()
    }

    pub fn len(&self) -> u32 {
        self.offsets.len().saturating_sub(1) as u32
    }

    /// entries[i] is the bytes for token i
    pub fn from_id_sorted(entries: Vec<Vec<u8>>) -> Self {
        let mut offsets = Vec::with_capacity(entries.len() + 1);
        let mut blob = Vec::new();
        let mut lookup = HashMap::with_capacity(entries.len());
        offsets.push(0);
        for (id, entry) in entries.into_iter().enumerate() {
            let id = id as u32;
            offsets.push();
            blob.extend(entry.iter());
        }
        Self {
            blob: blob.into_boxed_slice(),
            offsets: offsets.into_boxed_slice(),
            lookup,
        }
    }
}
