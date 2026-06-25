use ptr_hash::{PtrHash, PtrHashParams};
#[derive(Clone, Copy)]
struct Entry {
    start: u32,
    len: u16,
    id: u32,
}

struct VocabStore {
    mphf: PtrHash<Vec<u8>>,
    bytes: Box<[u8]>,
    entries: Box<[Entry]>,
}

impl VocabStore {
    // we build with vec of u8 (bytes)
    fn build(tokens: Vec<(Vec<u8>, u32)>) -> Self {
        let keys: Vec<Vec<u8>> = tokens.iter().map(|(s, _)| s.clone()).collect();

        let mphf: PtrHash<Vec<u8>> = PtrHash::new(&keys, PtrHashParams::default());
        let mut bytes = Vec::new();
        let mut entries = vec![
            Entry {
                start: 0,
                len: 0,
                id: 0,
            };
            keys.len()
        ];

        for (s, id) in tokens {
            let slot = mphf.index(&s);

            let start = bytes.len() as u32;
            let len = s.len() as u16;

            bytes.extend_from_slice(&s);

            entries[slot] = Entry { start, len, id };
        }

        Self {
            mphf,
            bytes: bytes.into_boxed_slice(),
            entries: entries.into_boxed_slice(),
        }
    }

    #[inline]
    fn get_bytes(&self, q: &[u8]) -> Option<u32> {
        // This part depends on whether ptr_hash accepts borrowed &[u8]
        // for a PtrHash<Vec<u8>>. If not, see note below.
        let slot = self.mphf.index(&q.to_vec());

        let e = self.entries[slot];

        if e.len as usize != q.len() {
            return None;
        }

        let start = e.start as usize;
        let end = start + e.len as usize;

        if &self.bytes[start..end] == q {
            Some(e.id)
        } else {
            None
        }
    }

    #[inline]
    fn token_to_id(&self, s: &str) -> Option<u32> {
        self.get_bytes(s.as_bytes())
    }

    #[inline]
    fn id_to_token(&self, i: u32) -> Option<String> {
        let entry = self.entries[i as usize];
        Some(
            String::from_utf8_lossy(
                &self.bytes[entry.start as usize..(entry.start + entry.len as u32) as usize],
            )
            .into_owned(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab_store::VocabStore;
    #[test]
    fn test_vocab_score() {
        let vocab = VocabStore::build(vec![("Hel".to_string().as_bytes().into(), 0)]);
        assert!(vocab.token_to_id("Hel") == Some(0));
        assert!(vocab.token_to_id("lo") == None);

        assert!(vocab.id_to_token(0) == Some("Hey".to_string()));
        assert!(vocab.id_to_token(1000) == None);
    }
}
