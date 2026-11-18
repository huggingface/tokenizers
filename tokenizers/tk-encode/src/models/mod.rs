//! Popular tokenizer models.

pub mod bpe;
#[cfg(feature = "unigram")]
pub mod unigram;
#[cfg(feature = "wordlevel")]
pub mod wordlevel;
#[cfg(feature = "wordpiece")]
pub mod wordpiece;

/// Characters whose UTF-8 encodings, taken together, contain every byte that valid UTF-8 can hold.
#[cfg(test)]
pub(crate) fn chars_covering_every_utf8_byte() -> Vec<char> {
    let mut seen = [false; 256];
    let mut chars = Vec::new();
    for c in char::MIN..=char::MAX {
        let mut buffer = [0; 4];
        let bytes = c.encode_utf8(&mut buffer).as_bytes();
        if bytes.iter().any(|&byte| !seen[byte as usize]) {
            for &byte in bytes {
                seen[byte as usize] = true;
            }
            chars.push(c);
        }
    }
    chars
}
