/// Parabix implementation
/// 8 basis bit streams.
/// ∧ is and
/// ¬ is not
pub struct BasisBitStream {
    b0: u64,
    b1: u64,
    b2: u64,
    b3: u64,
    b4: u64,
    b5: u64,
    b6: u64,
    b7: u64,
}
impl BasisBitStream {
    pub fn new(bytes: &Vec<u8>) -> Self {
        // 1. Naive implementation:
        let mut b0: u64 = 0u64;
        let mut b1: u64 = 0u64;
        let mut b2: u64 = 0u64;
        let mut b3: u64 = 0u64;
        let mut b4: u64 = 0u64;
        let mut b5: u64 = 0u64;
        let mut b6: u64 = 0u64;
        let mut b7: u64 = 0u64;
        for (idx, b) in bytes.iter().enumerate() {
            b0 |= ((*b & 0x01) as u64) << idx;
            b1 |= ((*b & 0x02) as u64) << idx;
            b2 |= ((*b & 0x04) as u64) << idx;
            b3 |= ((*b & 0x08) as u64) << idx;
            b4 |= ((*b & 0x10) as u64) << idx;
            b5 |= ((*b & 0x20) as u64) << idx;
            b6 |= ((*b & 0x40) as u64) << idx;
            b7 |= ((*b & 0x80) as u64) << idx;
        }
        Self {
            b0,
            b1,
            b2,
            b3,
            b4,
            b5,
            b6,
            b7,
        }
    } // This operation can be done using SIMD! ~1 cycle / byte.
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
    pub fn test_whitespace_split() {
        let input_str = "Hey how are you doing sir?";
        // To give an idea of the first runs:
        // idx   char   ASCII     binary ASCII   & 0x01   contributes
        // ───   ────   ─────     ────────────   ──────   ───────────
        //  0     H      0x48      01001000         0       0 << 0    -> 0u64
        //  1     e      0x65      01100101         1       1 << 1    -> 1u64
        //  2     y      0x79      01111001         1       1 << 2    -> 1u64 + 1u64 << 1
        //  3    ' '     0x20      00100000         0       0 << 3
        let stream = BasisBitStream::new(&input_str.as_bytes().to_vec());
        assert_eq!(stream.b0, 0b10110101100111010101100110 as u64);
        // let expected_splits = vec![3, 7, 11, 15, 21];
    }
}
