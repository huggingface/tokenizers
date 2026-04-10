/// A compact, cache-friendly id→token store.
///
/// All token strings are concatenated into a single contiguous byte buffer and
/// indexed by a dense array of `u32` byte offsets.  Reverse lookup (id → token)
/// is two array reads — no hash-table indirection, no per-string heap
/// allocation — and sequential id scans (serialization, iteration) stay in the
/// same cache lines.
///
/// # Layout
///
/// ```text
/// data:    [ h e l l o w o r l d ]
/// offsets: [ 0, 5, 10 ]          (len = vocab_size + 1)
///                                  id=0 → data[0..5]  = "hello"
///                                  id=1 → data[5..10] = "world"
/// ```
///
/// # Dense ids
/// Ids must form a range `0..N` but **gaps are allowed**: a missing id is
/// represented by an empty slice (`offsets[i] == offsets[i+1]`) and returns
/// `None` from [`get`].  An empty-string token and a gap are therefore
/// indistinguishable — avoid inserting empty tokens.
#[derive(Clone, Default, Debug)]
pub struct CompactVocab {
    /// Concatenated UTF-8 bytes of every token, in ascending id order.
    data: Vec<u8>,
    /// `offsets[i]` = byte start of token `i`; `offsets[i+1]` = exclusive end.
    /// `offsets.len() == max_id + 2` (or 0 for an empty vocab).
    offsets: Vec<u32>,
}

impl CompactVocab {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build from an iterator of `(token, id)` pairs.
    ///
    /// The pairs can arrive in any order and may contain gaps.  Only one pass
    /// over the iterator is made; the resulting buffer is allocated exactly
    /// once.
    pub fn from_vocab(iter: impl IntoIterator<Item = (impl AsRef<str>, u32)>) -> Self {
        let mut sorted: Vec<(u32, String)> = iter
            .into_iter()
            .map(|(s, id)| (id, s.as_ref().to_owned()))
            .collect();

        if sorted.is_empty() {
            return Self::default();
        }
        sorted.sort_unstable_by_key(|(id, _)| *id);

        let max_id = sorted.last().unwrap().0 as usize;
        let n = max_id + 1; // number of slots (including gaps)

        // Pre-calculate total data size to avoid reallocations.
        let total_bytes: usize = sorted.iter().map(|(_, s)| s.len()).sum();
        let mut data = Vec::with_capacity(total_bytes);
        let mut offsets = Vec::with_capacity(n + 1);

        let mut sorted_iter = sorted.into_iter().peekable();

        for i in 0..n {
            offsets.push(data.len() as u32);
            if sorted_iter.peek().map(|(id, _)| *id as usize) == Some(i) {
                let (_, token) = sorted_iter.next().unwrap();
                data.extend_from_slice(token.as_bytes());
            }
            // gap → no bytes written → offsets[i] == offsets[i+1] (filled next iteration)
        }
        offsets.push(data.len() as u32); // sentinel

        Self { data, offsets }
    }

    /// Return the token string for `id`, or `None` for an unknown / gap id.
    ///
    /// This is two array reads — no hash lookup.
    #[inline]
    pub fn get(&self, id: u32) -> Option<&str> {
        let i = id as usize;
        let (&start, &end) = (self.offsets.get(i)?, self.offsets.get(i + 1)?);
        if start == end {
            return None; // gap — id was never inserted
        }
        // SAFETY: `data` only ever receives bytes from valid `String` / `&str` values.
        Some(unsafe { std::str::from_utf8_unchecked(&self.data[start as usize..end as usize]) })
    }

    /// Number of id slots (including gaps); equals `max_id + 1`.
    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    pub fn is_empty(&self) -> bool {
        self.offsets.len() <= 1
    }

    /// Iterate `(token, id)` pairs in ascending id order, skipping gaps.
    pub fn iter(&self) -> impl Iterator<Item = (&str, u32)> + '_ {
        let data = &self.data;
        let offsets = &self.offsets;
        let n = self.len() as u32;
        (0..n).filter_map(move |id| {
            let i = id as usize;
            let (&start, &end) = (offsets.get(i)?, offsets.get(i + 1)?);
            if start == end {
                return None;
            }
            // SAFETY: data only receives bytes from valid String / &str values.
            let s = unsafe {
                std::str::from_utf8_unchecked(&data[start as usize..end as usize])
            };
            Some((s, id))
        })
    }
}

impl PartialEq for CompactVocab {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.offsets == other.offsets
    }
}

// ---------------------------------------------------------------------------
// Serde — JSON object {"token": id} in ascending id order, same format as
// OrderedVocabIter so existing tokenizer files remain compatible.
// ---------------------------------------------------------------------------

use serde::{
    de::Deserializer,
    ser::{SerializeMap, Serializer},
    Deserialize, Serialize,
};

impl Serialize for CompactVocab {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.len()))?;
        for (token, id) in self.iter() {
            map.serialize_entry(token, &id)?;
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for CompactVocab {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Deserialize as the same {"token": id} map that the BPE JSON uses.
        let raw: std::collections::HashMap<String, u32> =
            Deserialize::deserialize(deserializer)?;
        Ok(Self::from_vocab(raw))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_dense() {
        let pairs = vec![
            ("hello".to_string(), 0u32),
            ("world".to_string(), 1),
            ("foo".to_string(), 2),
        ];
        let cv = CompactVocab::from_vocab(pairs);

        assert_eq!(cv.get(0), Some("hello"));
        assert_eq!(cv.get(1), Some("world"));
        assert_eq!(cv.get(2), Some("foo"));
        assert_eq!(cv.get(3), None);
        assert_eq!(cv.len(), 3);
    }

    #[test]
    fn round_trip_with_gaps() {
        let pairs = vec![("a".to_string(), 0u32), ("c".to_string(), 2)];
        let cv = CompactVocab::from_vocab(pairs);

        assert_eq!(cv.get(0), Some("a"));
        assert_eq!(cv.get(1), None); // gap
        assert_eq!(cv.get(2), Some("c"));
        assert_eq!(cv.len(), 3); // slots 0, 1, 2
    }

    #[test]
    fn iter_skips_gaps() {
        let pairs = vec![("a".to_string(), 0u32), ("c".to_string(), 2)];
        let cv = CompactVocab::from_vocab(pairs);
        let collected: Vec<(&str, u32)> = cv.iter().collect();
        assert_eq!(collected, vec![("a", 0), ("c", 2)]);
    }

    #[test]
    fn serde_round_trip() {
        let pairs = vec![
            ("hello".to_string(), 0u32),
            ("world".to_string(), 1),
        ];
        let cv = CompactVocab::from_vocab(pairs);
        let json = serde_json::to_string(&cv).unwrap();
        assert_eq!(json, r#"{"hello":0,"world":1}"#);
        let cv2: CompactVocab = serde_json::from_str(&json).unwrap();
        assert_eq!(cv, cv2);
    }

    #[test]
    fn unordered_input() {
        // Input in reverse id order — should still reconstruct correctly.
        let pairs = vec![("c".to_string(), 2u32), ("a".to_string(), 0), ("b".to_string(), 1)];
        let cv = CompactVocab::from_vocab(pairs);
        assert_eq!(cv.get(0), Some("a"));
        assert_eq!(cv.get(1), Some("b"));
        assert_eq!(cv.get(2), Some("c"));
    }

    #[test]
    fn single_contiguous_allocation() {
        let pairs = vec![("ab".to_string(), 0u32), ("cd".to_string(), 1)];
        let cv = CompactVocab::from_vocab(pairs);
        // All 4 bytes live in one Vec.
        assert_eq!(cv.data.len(), 4);
        assert_eq!(&cv.data, b"abcd");
    }
}
