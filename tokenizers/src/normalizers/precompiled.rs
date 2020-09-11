use crate::tokenizer::{NormalizedString, Normalizer, Result};
use nom::{number::complete::le_u32, number::Endianness, IResult, ToUsize};
use serde::{
    de::{Error, Visitor},
    ser::SerializeStruct,
    Deserialize, Deserializer, Serialize, Serializer,
};
use std::fmt;

/// This struct is specifically done to be compatible with SentencePiece
/// SentencePiece models embed their Normalizer within a `precompiled_charsmap`
/// that both represents a Trie, and embedded rewrite rules.
/// In order to be 100% compliant we need to interpret that binary format too.
///
#[derive(Default, Clone, Debug, PartialEq)]
pub struct Precompiled {
    precompiled_charsmap: Vec<u8>,
    pub(super) normalized: String,
    pub(super) trie: DoubleArray,
}

impl Serialize for Precompiled {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("Precompiled", 1)?;
        state.serialize_field("precompiled_charsmap", &self.precompiled_charsmap)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for Precompiled {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(PrecompiledVisitor)
    }
}
struct PrecompiledVisitor;
impl<'de> Visitor<'de> for PrecompiledVisitor {
    type Value = Precompiled;
    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Precompiled")
    }

    fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: serde::de::MapAccess<'de>,
    {
        let maybe_type = map.next_entry::<String, Vec<u8>>()?;
        let maybe_type_str = maybe_type.as_ref().map(|(k, v)| (k.as_str(), v));
        match maybe_type_str {
            Some(("precompiled_charsmap", value)) => Ok(Precompiled::from(&value)
                .map_err(|_| Error::custom("Cannot read `precompiled` string"))?),
            _ => Err(Error::custom("Expected precompiled value, got {:?}")),
        }
    }
}

named!(be<u32>, u32!(Endianness::Little));

pub type ArrayUnit = usize;

trait ArrayUnitTrait {
    fn has_leaf(&self) -> bool;
    fn value(&self) -> isize;
    fn label(&self) -> usize;
    fn offset(&self) -> usize;
}

impl ArrayUnitTrait for ArrayUnit {
    fn has_leaf(&self) -> bool {
        (self >> 8) & 1 == 1
    }

    fn value(&self) -> isize {
        (self & ((1usize << 31) - 1)) as isize
    }

    fn label(&self) -> usize {
        self & ((1usize << 31) | 0xFF)
    }

    fn offset(&self) -> usize {
        (self >> 10) << ((self & (1usize << 9)) >> 6)
    }
}

type Array = Vec<ArrayUnit>;

#[derive(Default, Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DoubleArray {
    array: Array,
}

impl DoubleArray {
    fn from(array: Array) -> Self {
        Self { array }
    }

    pub fn common_prefix_search(&self, key: &[u8]) -> Vec<isize> {
        let mut node_pos = 0;
        let mut results = vec![];

        let mut unit = self.array[node_pos];
        node_pos ^= unit.offset();
        for c in key {
            if *c == 0u8 {
                break;
            }
            node_pos ^= *c as usize;
            unit = self.array[node_pos];
            if unit.label() != *c as usize {
                return results;
            }
            node_pos ^= unit.offset();
            if unit.has_leaf() {
                results.push(self.array[node_pos].value());
            }
        }
        results
    }
}

fn parse(precompiled_charsmap: &[u8]) -> IResult<&[u8], Array> {
    let n = precompiled_charsmap.len();
    let (mut rest, trie_size) = le_u32(precompiled_charsmap)?;
    assert_eq!(trie_size % 4, 0);
    // u8 to u32.
    let trie_char_size = trie_size / 4;
    let mut trie_blob = Vec::with_capacity(trie_char_size as usize);
    for _ in 0..trie_char_size {
        let (rest2, n) = le_u32(rest)?;
        rest = rest2;
        trie_blob.push(n.to_usize());
    }
    let normalized_blob = rest;
    assert_eq!(rest.len() + trie_size as usize + 4, n);
    Ok((normalized_blob, trie_blob))
}

impl Precompiled {
    pub fn from(precompiled_charsmap: &[u8]) -> Result<Precompiled> {
        let (normalized_blob, trie_blob) = parse(&precompiled_charsmap).unwrap();
        let normalized = String::from_utf8(normalized_blob.to_vec())?;
        let trie = DoubleArray::from(trie_blob);
        let precompiled = Precompiled {
            precompiled_charsmap: precompiled_charsmap.to_vec(),
            normalized,
            trie,
        };
        Ok(precompiled)
    }

    pub(super) fn transform(&self, chunk: &str) -> Option<&str> {
        let results = self.trie.common_prefix_search(&chunk.as_bytes());
        if results.is_empty() {
            None
        } else {
            let index = results[0] as usize;
            let mut index2 = index;
            while index2 < self.normalized.len() {
                if self.normalized.bytes().nth(index2)? == 0u8 {
                    break;
                }
                index2 += 1;
            }
            Some(&self.normalized[index..index2])
        }
    }
}

impl Normalizer for Precompiled {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        let mut normalized_string = Vec::with_capacity(normalized.get().len());
        normalized.get().char_indices().for_each(|(index, c)| {
            let source = &normalized.get()[index..index + c.len_utf8()];
            if let Some(normalized) = self.transform(source) {
                for (i, c) in normalized.chars().enumerate() {
                    normalized_string.push((c, i as isize));
                }
            } else {
                normalized_string.push((c, 0));
            }
        });
        normalized.transform(normalized_string.into_iter(), 0);
        Ok(())
    }
}
