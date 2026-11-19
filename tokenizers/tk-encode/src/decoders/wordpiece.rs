use crate::tokenizer::{Decode, Result};

#[derive(Clone, Debug)]
/// The WordPiece decoder takes care of decoding a list of wordpiece tokens
/// back into a readable string.
#[non_exhaustive]
pub struct WordPiece {
    /// The prefix to be used for continuing subwords
    pub prefix: String,
    /// Whether to cleanup some tokenization artifacts (spaces before punctuation, ...)
    pub cleanup: bool,
}

impl WordPiece {
    pub fn new(prefix: String, cleanup: bool) -> Self {
        Self { prefix, cleanup }
    }
}

impl Default for WordPiece {
    fn default() -> Self {
        Self {
            prefix: "##".to_owned(),
            cleanup: true,
        }
    }
}
pub fn cleanup(dirty_input: &str) -> String {
    dirty_input
        .replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" do not", " don't")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
}

impl Decode for WordPiece {
    fn decode_chain(&self, mut tokens: Vec<String>) -> Result<Vec<String>> {
        for (i, token) in tokens.iter_mut().enumerate() {
            if i != 0 {
                if let Some(tk) = token.strip_prefix(&self.prefix) {
                    *token = tk.to_string();
                } else {
                    *token = format!(" {token}");
                }
            }
            // Keep these byte pairs in sync with the patterns in cleanup().
            if self.cleanup
                && token.as_bytes().windows(2).any(|pair| {
                    pair[0] == b' '
                        && matches!(pair[1], b'.' | b'?' | b'!' | b',' | b'\'' | b'n' | b'd')
                })
            {
                *token = cleanup(token);
            }
        }
        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wordpiece_decoder() {
        let decoder = WordPiece::new("##".to_string(), false);

        assert_eq!(
            decoder
                .decode(vec![
                    "##uelo".to_string(),
                    "Ara".to_string(),
                    "##új".to_string(),
                    "##o".to_string(),
                    "No".to_string(),
                    "##guera".to_string()
                ])
                .unwrap(),
            "##uelo Araújo Noguera"
        );
    }

    #[test]
    fn wordpiece_cleanup_preserves_all_replacements() {
        let decoder = WordPiece::new("##".to_string(), true);
        for (input, expected) in [
            ("hello .", "hello."),
            ("hello ?", "hello?"),
            ("hello !", "hello!"),
            ("hello ,", "hello,"),
            ("a ' b", "a'b"),
            ("ca n't", "can't"),
            ("i 'm", "i'm"),
            ("i do not", "i don't"),
            ("it 's", "it's"),
            ("i 've", "i've"),
            ("we 're", "we're"),
            ("你好 世界", "你好 世界"),
        ] {
            assert_eq!(decoder.decode(vec![input.to_owned()]).unwrap(), expected);
        }
        assert_eq!(
            decoder
                .decode(["hello", "##s", ".", "world"].map(String::from).to_vec())
                .unwrap(),
            "hellos. world"
        );
    }

    #[test]
    fn wordpiece_cleanup_matches_reference_for_short_inputs() {
        let decoder = WordPiece::new("##".to_string(), true);
        let alphabet = [' ', '.', '?', '!', ',', '\'', 'n', 'd', 'a', '你'];
        for a in alphabet {
            for b in alphabet {
                for c in alphabet {
                    let input: String = [a, b, c].into_iter().collect();
                    assert_eq!(
                        decoder.decode(vec![input.clone()]).unwrap(),
                        cleanup(&input),
                        "input: {input:?}"
                    );
                }
            }
        }
    }
}
