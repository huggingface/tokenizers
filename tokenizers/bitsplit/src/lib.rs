use std::fmt;

/// Minimal Parabix impementation
/// 8 basis bit streams.
/// ∧ is and
/// ¬ is not
/// ('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)
/// This is the first full regex we will implement. It needs various bitstreams. We'll start with
/// `'(s|t|re|ve|m|ll|d)`. b'\''
pub struct BasisBitStream {
    stream: [u64; 8],
}
impl BasisBitStream {
    pub fn new(bytes: &Vec<u8>) -> Self {
        // 1. Naive implementation:

        let mut stream = [0u64; 8];
        for (idx, b) in bytes.iter().enumerate() {
            for n in 0..8 {
                stream[n] |= (((*b >> n) & 1) as u64) << idx;
            }
        }
        Self { stream }
    } // This operation can be done using SIMD! ~1 cycle / byte.
    // Now let's try a SIMD implementation?
}

impl fmt::Display for BasisBitStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut output = "".to_string();
        for n in 0..8 {
            output += &format!("b{n}:{:64b}\n", self.stream[n]).to_string();
        }
        write!(f, "{}", output)
    }
}
pub struct CharacterBitStream {}
pub struct LexicalBitStream {}

pub fn advance(cursor_bits: BasisBitStream) {} // Advances cursor bits forward by one.
pub fn scan_thru(cursor_bits: BasisBitStream, markers: BasisBitStream) {} // cursor positions and marked lexical positions. Computes (c + m) ∧¬m.
pub fn bitstream_inverse_transpose() {} // Same, should be SIMD.

#[cfg(test)]
pub mod test {
    use crate::BasisBitStream;

    #[test]
    pub fn test_new() {
        let input_str = "Hey how are you doing sir?";
        // To give an idea of the first runs:
        // idx   char   ASCII     binary ASCII   & 0x01   contributes
        // ───   ────   ─────     ────────────   ──────   ───────────
        //  0     H      0x48      01001000         0       0 << 0    -> 0u64
        //  1     e      0x65      01100101         1       1 << 1    -> 1u64
        //  2     y      0x79      01111001         1       1 << 2    -> 1u64 + 1u64 << 1
        //  3    ' '     0x20      00100000         0       0 << 3
        let stream = BasisBitStream::new(&input_str.as_bytes().to_vec());
        assert_eq!(stream.stream[0], 0b10110101100111010101100110 as u64);
        // let expected_splits = vec![3, 7, 11, 15, 21];
    }
    #[test]
    pub fn test_appostrophe() {
        let input = 0x27;
        let cc = vec![input; 64];
        let input_cc = BasisBitStream::new(&cc);
        println!("{:b}", input);
        println!("{}", input_cc);
        println!("{:?}", cc);
        let text = "Hey I'll have your's";
        let text_stream = BasisBitStream::new(&text.as_bytes().to_vec());
        println!("{:}", text_stream);
        let mut markers = text_stream.stream[0]
            & text_stream.stream[1]
            & text_stream.stream[2]
            & !text_stream.stream[3]
            & !text_stream.stream[4]
            & text_stream.stream[5]
            & !text_stream.stream[6]
            & !text_stream.stream[7];
        println!("ms:{:64b}", markers);
        let first_match = markers.trailing_zeros();
        markers = markers & (markers.saturating_sub(1u64));
        println!("ms:{:64b}", markers);
        let second_match = markers.trailing_zeros();
        assert_eq!(first_match, 5);
        assert_eq!(second_match, 18);
    }
}
