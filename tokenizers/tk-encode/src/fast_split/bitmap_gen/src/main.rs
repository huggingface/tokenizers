/// This will store what we need to build the unicode free pretokenization tables.
/// iterates codepoints in the 2-byte (U+0080–07FF) and 3-byte (U+0800–FFFF) ranges,
/// tests membership, sets bits, and emits static LETTER2: [u64;32], LETTER3: [u64;1024],
/// NUM2/NUM3, … into a committed unicode_tables.rs.
/// Index scheme like the POC: LETTER3[((lead-0xE0)<<6)|(b2-0x80)] >> (b3&0x3F).
/// 4-byte (astral) letters are rare → a small fallback fn, not a giant table.
/// This is only used in fallback / non SIMD path.

const LETTER_RANGES: &[(u32, u32)] = &[(0x41, 0x5A), (0x61, 0x7A), (0xC0, 0xD6)]; // \p{L}, pinned to onig's Unicode ver
const fn is_letter(cp: u32) -> bool {
    unsafe {
        if let Some(c) = char::from_u32(cp) {
            c.is_ascii_alphabetic()
        } else {
            false
        }
    }
}
const LETTER2: [u64; 32] = {
    // we compute the 32 different values for the 32 starting bytes of 2-byte unicode
    // the value is a bitmap. Since there are 64 possible continuation values, for the
    // bitmap just says for each on of them 1 if its a letter 0 otherwise.
    //                                                        utf8       utf8
    //                                                        ———        ——
    // When you check say 0xCEA4 for example. You take CEL 0b[110.01110][10.100100]
    //                                                          byte1      byt2
    // you take the leading non utf8 value (0xCE - 0xC0) which gives you the index in the table.
    // Then you just look at the value stored, shiffted by the byt2 interpreded as u64:
    // byte2 & 0x3F. This just isolates the value at the bitmap. Now you just do & 1.
    // For xCEA4, the index is 12, the bit shift is 36, it points to the x below.
    //                                10011010000011101010110011000110110x1001100011111101010100000000
    let mut t = [0u64; 32];
    let mut i = 0u32;
    while i < 32 {
        let mut bitmap = 0u64;
        let mut byte2 = 0u32;
        while byte2 < 64 {
            let codepoint = i << 6 | byte2;
            if is_letter(codepoint) {
                bitmap |= 1u64 << byte2
            }
            byte2 += 1;
        }
        t[i as usize] = bitmap;
        i += 1;
    }
    t
};

const LETTER3: [u64; 1024] = {
    let mut t = [0u64; 1024];
    let mut i = 0;
    while i < 1024 {
        let mut bitmap = 0u64;
        let mut byte3 = 0u32;
        while byte3 < 64 {
            let codepoint = i << 5 | byte3;
            if is_letter(codepoint) {
                bitmap |= 1u64 << byte3
            }
            byte3 += 1;
        }
        t[i as usize] = bitmap;
        i += 1;
    }
    t
};
// const NUMBER2: [u64; 32] = { todo!() };
// const NUMBER3: [u64; 1024] = { todo!() };
//
// TODO: now that I got the algo, I need to test every single codepoint?
fn main() {
    println!("Generating bitmap table for fast nibble based splitting");
    let letter = "é".as_bytes();
    let lead = letter[0] - 0xC0;
    let shift = letter[1] & 0x3F;
    assert_eq!(LETTER2[lead as usize] >> shift & 1, 1);

    let no_letter = "|".as_bytes();
    let lead = no_letter[0] - 0xC0;
    let shift = no_letter[1] & 0x3F;
    assert_eq!(LETTER2[lead as usize] >> shift & 1, 0);
    let three_byte_letter = "ত".as_bytes();
    let lead = three_byte_letter[0] - 0xE0 << 6;
    let second = three_byte_letter[2] - 0x3F;
    let shift = three_byte_letter[3] & 0x3F;
    assert_eq!(LETTER3[(lead | second) as usize] >> shift & 1, 0)
}
