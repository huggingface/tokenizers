use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::tokenizer::{pattern::Pattern, PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior};

/// Pre-tokenizes text by splitting it into random-length chunks.
///
/// This allows tokenization across traditional whitespace boundaries, enabling BPE
/// to learn multi-word expressions as a single token.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub struct RandomChunkSplit {
    /// Minimum chunk length (in characters)
    pub min_length: usize,
    /// Maximum chunk length (in characters)
    pub max_length: usize,
}

impl RandomChunkSplit {
    /// Create a new `RandomChunkSplit` with the given min and max chunk lengths.
    pub fn new(min_length: usize, max_length: usize) -> Self {
        // Ensure min_length and max_length are valid
        let min_length = min_length.max(1);
        let max_length = max_length.max(min_length);

        Self {
            min_length,
            max_length,
        }
    }
}

/// Split pattern that creates chunks of random lengths
struct RandomChunkPattern<'a> {
    min_length: usize,
    max_length: usize,
    chars: &'a [char],
    current_pos: usize,
}

impl<'a> RandomChunkPattern<'a> {
    fn new(chars: &'a [char], min_length: usize, max_length: usize) -> Self {
        Self {
            min_length,
            max_length,
            chars,
            current_pos: 0,
        }
    }
}

impl<'a> Pattern for RandomChunkPattern<'a> {
    fn find_matches(&self, _text: &str) -> Result<Vec<((usize, usize), bool)>> {
        let mut result = Vec::new();
        let mut current_pos = self.current_pos;
        let chars = self.chars;
        let mut char_start_byte = 0;
        
        // Get byte offset of current_pos
        for i in 0..current_pos {
            char_start_byte += chars[i].len_utf8();
        }
        
        while current_pos < chars.len() {
            // Calculate remaining characters
            let remaining = chars.len() - current_pos;
            
            // Calculate effective max length (limited by remaining chars)
            let effective_max = self.max_length.min(remaining);
            
            // If we can't satisfy minimum length, just take all remaining chars
            if effective_max < self.min_length {
                let chunk_len = remaining;
                let mut chunk_bytes = 0;
                for i in 0..chunk_len {
                    chunk_bytes += chars[current_pos + i].len_utf8();
                }
                
                result.push(((char_start_byte, char_start_byte + chunk_bytes), false));
                break;
            }
            
            // Generate random chunk length between min_length and effective_max
            let chunk_len = if self.min_length == effective_max {
                self.min_length
            } else {
                let mut rng = rand::thread_rng();
                rng.gen_range(self.min_length..=effective_max)
            };
            
            // Calculate byte length of this chunk
            let mut chunk_bytes = 0;
            for i in 0..chunk_len {
                chunk_bytes += chars[current_pos + i].len_utf8();
            }
            
            // Add segment
            result.push(((char_start_byte, char_start_byte + chunk_bytes), false));
            
            // Update positions
            current_pos += chunk_len;
            char_start_byte += chunk_bytes;
        }
        
        Ok(result)
    }
}

impl PreTokenizer for RandomChunkSplit {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, normalized| {
            let chars: Vec<char> = normalized.get().chars().collect();
            let pattern = RandomChunkPattern::new(&chars, self.min_length, self.max_length);
            normalized.split(pattern, SplitDelimiterBehavior::Isolated)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OffsetReferential, OffsetType};

    #[test]
    fn test_empty_string() {
        let pretok = RandomChunkSplit::new(1, 5);
        let mut pretokenized = PreTokenizedString::from("");
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        
        let splits = pretokenized.get_splits(OffsetReferential::Original, OffsetType::Byte);
        assert_eq!(splits.len(), 0);
    }

    #[test]
    fn test_deterministic_chunks() {
        // With min_length = max_length, the chunking should be deterministic
        let pretok = RandomChunkSplit::new(3, 3);
        let s = "Hello world!";
        let mut pretokenized = PreTokenizedString::from(s);
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        
        let splits = pretokenized
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(s, o, _)| (s, o))
            .collect::<Vec<_>>();
            
        assert_eq!(
            splits,
            vec![
                ("Hel", (0, 3)),
                ("lo ", (3, 6)),
                ("wor", (6, 9)),
                ("ld!", (9, 12)),
            ]
        );
    }

    #[test]
    fn test_unicode_handling() {
        // Ensure proper handling of multi-byte Unicode characters
        let pretok = RandomChunkSplit::new(1, 1);
        let s = "こんにちは";  // "Hello" in Japanese
        let mut pretokenized = PreTokenizedString::from(s);
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        
        let splits = pretokenized
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(s, o, _)| (s, o))
            .collect::<Vec<_>>();
            
        assert_eq!(splits.len(), 5);  // 5 characters
        assert_eq!(splits[0].0, "こ");
        assert_eq!(splits[1].0, "ん");
        assert_eq!(splits[2].0, "に");
        assert_eq!(splits[3].0, "ち");
        assert_eq!(splits[4].0, "は");
    }

    #[test]
    fn test_min_max_validation() {
        // If min > max, it should be corrected
        let pretok = RandomChunkSplit::new(5, 3);
        assert_eq!(pretok.min_length, 5);
        assert_eq!(pretok.max_length, 5);
        
        // Min can't be 0
        let pretok = RandomChunkSplit::new(0, 5);
        assert_eq!(pretok.min_length, 1);
        assert_eq!(pretok.max_length, 5);
    }

    #[test]
    fn test_random_chunks() {
        // Test with a range of chunk sizes
        let pretok = RandomChunkSplit::new(1, 5);
        let s = "The quick brown fox jumps over the lazy dog.";
        let mut pretokenized = PreTokenizedString::from(s);
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        
        let splits = pretokenized
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(s, o, _)| (s, o))
            .collect::<Vec<_>>();
            
        // Ensure all characters are accounted for
        let joined: String = splits.iter().map(|(s, _)| s.to_string()).collect();
        assert_eq!(joined, s);
        
        // Verify that each chunk has a length within the specified range
        for (chunk, _) in &splits {
            let chunk_chars = chunk.chars().count();
            assert!(chunk_chars >= 1 && chunk_chars <= 5);
        }
    }
}