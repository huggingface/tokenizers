use crate::tokenizer::{NormalizedString, Normalizer, Result};
use nom::{bytes::complete::take, number::Endianness, IResult};
use serde::{Deserialize, Serialize};

/// This struct is specifically done to be compatible with SentencePiece
/// SentencePiece models embed their Normalizer within a `precompiled_charsmap`
/// that both represents a Trie, and embedded rewrite rules.
/// In order to be 100% compliant we need to interpret that binary format too.
///
#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct Precompiled {
    normalized: String,
    trie: DoubleArray,
}

named!(be<u32>, u32!(Endianness::Little));

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
struct DoubleArray {
    array: String,
}

impl DoubleArray {
    fn from(array: String) -> Self {
        Self { array }
    }
}

fn parse(precompiled_charsmap: &[u8], n: usize) -> IResult<&[u8], &[u8]> {
    println!("Precompiled {:?}", precompiled_charsmap.len());
    let (rest, trie_size) = be(precompiled_charsmap)?;
    let (normalized_blob, trie_blob) = take(trie_size as usize)(rest)?;
    Ok((trie_blob, normalized_blob))
}

impl Precompiled {
    pub fn from(precompiled_charsmap: &[u8]) -> Result<Precompiled> {
        let (trie_blob, normalized_blob) =
            parse(&precompiled_charsmap, precompiled_charsmap.len()).unwrap();
        let normalized = String::from_utf8(normalized_blob.to_vec())?;
        let array = String::from_utf8(trie_blob.to_vec())?;
        let trie = DoubleArray::from(array);
        let precompiled = Precompiled { normalized, trie };
        Ok(precompiled)
    }
}

impl Normalizer for Precompiled {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        Ok(())
    }
}
