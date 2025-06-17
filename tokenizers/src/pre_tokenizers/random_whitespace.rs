use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::tokenizer::{PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior};

/// Pre-tokenizes text by deciding at each whitespace character whether to split
/// or continue based on a given probability.
///
/// This pre-tokenizer is similar to `WhitespaceSplit` but randomly decides for
/// each whitespace character whether to split at that position. This allows
/// tokenization to occasionally span across whitespace boundaries, enabling BPE
/// to learn multi-word expressions as single tokens.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub struct RandomWhitespaceSplit {
    /// Probability of splitting at each whitespace character (0.0-1.0)
    pub split_probability: f32,
    /// When true, uses deterministic behavior instead of random decisions (for inference)
    #[serde(default)]
    pub deterministic: bool,
}

impl RandomWhitespaceSplit {
    /// Create a new `RandomWhitespaceSplit` with the given probability.
    ///
    /// The `split_probability` determines how likely the tokenizer is to split
    /// at each whitespace character. Higher values (closer to 1.0) make the behavior
    /// more similar to traditional `WhitespaceSplit`, while lower values encourage
    /// more multi-word tokens.
    pub fn new(split_probability: f32) -> Self {
        // Ensure probability is within valid range
        let split_probability = split_probability.min(1.0).max(0.0);

        Self { 
            split_probability,
            deterministic: false,
        }
    }
    
    /// Sets the deterministic mode for inference
    ///
    /// When deterministic is true, a consistent behavior will be used for
    /// whitespace splitting. In deterministic mode:
    /// - If split_probability > 0.5, all whitespace is split (like WhitespaceSplit)
    /// - If split_probability <= 0.5, no whitespace is split (preserving multi-word tokens)
    /// 
    /// This provides consistent tokenization at inference time.
    pub fn with_deterministic(mut self, deterministic: bool) -> Self {
        self.deterministic = deterministic;
        self
    }
}

impl PreTokenizer for RandomWhitespaceSplit {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        // This function captures the split probability and deterministic mode
        let split_prob = self.split_probability;
        let deterministic = self.deterministic;
        
        pretokenized.split(|_, normalized| {
            // Create a pattern closure that decides whether to split at whitespace
            let whitespace_pattern = |c: char| {
                if c.is_whitespace() {
                    if deterministic {
                        // In deterministic mode, make a consistent decision based on the probability
                        // If split_prob > 0.5, always split (like WhitespaceSplit)
                        // If split_prob <= 0.5, never split (preserve multi-word tokens)
                        split_prob > 0.5
                    } else {
                        // In random mode, use randomness to decide
                        let mut rng = rand::thread_rng();
                        rng.gen::<f32>() < split_prob
                    }
                } else {
                    false
                }
            };
            
            // Use the pattern with the normalized string
            normalized.split(whitespace_pattern, SplitDelimiterBehavior::Isolated)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OffsetReferential, OffsetType};

    #[test]
    fn test_empty_string() {
        let pretok = RandomWhitespaceSplit::new(0.5);
        let mut pretokenized = PreTokenizedString::from("");
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        
        let splits = pretokenized.get_splits(OffsetReferential::Original, OffsetType::Byte);
        assert_eq!(splits.len(), 0);
    }

    #[test]
    fn test_full_split_probability() {
        // With split_probability = 1.0, should behave like WhitespaceSplit
        let pretok = RandomWhitespaceSplit::new(1.0);
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
                ("Hello", (0, 5)),
                (" ", (5, 6)),
                ("world!", (6, 12)),
            ]
        );
    }

    #[test]
    fn test_zero_split_probability() {
        // With split_probability = 0.0, should not split at all
        let pretok = RandomWhitespaceSplit::new(0.0);
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
                ("Hello world!", (0, 12)),
            ]
        );
    }

    #[test]
    fn test_multiple_whitespaces() {
        // Test with multiple whitespaces and full split probability
        let pretok = RandomWhitespaceSplit::new(1.0);
        let s = "Hello  world!\nTest";
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
                ("Hello", (0, 5)),
                (" ", (5, 6)),
                (" ", (6, 7)),
                ("world!", (7, 13)),
                ("\n", (13, 14)),
                ("Test", (14, 18)),
            ]
        );
    }
    
    #[test]
    fn test_deterministic_mode() {
        let s = "Hello world! How are you?";
        
        // Test with high probability (>0.5) in deterministic mode
        // Should behave like WhitespaceSplit (all whitespace split)
        let high_prob_pretok = RandomWhitespaceSplit::new(0.7).with_deterministic(true);
        let mut high_prob_pretokenized1 = PreTokenizedString::from(s);
        high_prob_pretok.pre_tokenize(&mut high_prob_pretokenized1).unwrap();
        
        // Run it again to verify consistency
        let mut high_prob_pretokenized2 = PreTokenizedString::from(s);
        high_prob_pretok.pre_tokenize(&mut high_prob_pretokenized2).unwrap();
        
        // Get the splits from both runs
        let high_prob_splits1 = high_prob_pretokenized1
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(s, o, _)| (s, o))
            .collect::<Vec<_>>();
            
        let high_prob_splits2 = high_prob_pretokenized2
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(s, o, _)| (s, o))
            .collect::<Vec<_>>();
        
        // The two runs should produce identical results in deterministic mode
        assert_eq!(high_prob_splits1, high_prob_splits2);
        
        // With high probability, should behave like WhitespaceSplit (everything split)
        // Expected: ["Hello", " ", "world!", " ", "How", " ", "are", " ", "you?"]
        assert_eq!(high_prob_splits1.len(), 9);
        assert_eq!(high_prob_splits1[0].0, "Hello");
        assert_eq!(high_prob_splits1[1].0, " ");
        assert_eq!(high_prob_splits1[2].0, "world!");
        
        // Test with low probability (<=0.5) in deterministic mode
        // Should never split on whitespace
        let low_prob_pretok = RandomWhitespaceSplit::new(0.3).with_deterministic(true);
        let mut low_prob_pretokenized = PreTokenizedString::from(s);
        low_prob_pretok.pre_tokenize(&mut low_prob_pretokenized).unwrap();
        
        let low_prob_splits = low_prob_pretokenized
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(s, o, _)| (s, o))
            .collect::<Vec<_>>();
        
        // With low probability, no whitespace should be split at all
        assert_eq!(low_prob_splits.len(), 1);
        assert_eq!(low_prob_splits[0].0, s);
    }
}