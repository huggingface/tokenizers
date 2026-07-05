use std::io::Write;

/// This will store what we need to build the unicode free pretokenization tables.
/// iterates codepoints in the 2-byte (U+0080–07FF) and 3-byte (U+0800–FFFF) ranges,
/// tests membership, sets bits, and emits static LETTER2: [u64;32], LETTER3: [u64;1024],
/// NUM2/NUM3, … into a committed unicode_tables.rs.
/// Index scheme like the POC: LETTER3[((lead-0xE0)<<6)|(b2-0x80)] >> (b3&0x3F).
/// 4-byte (astral) letters are rare → a small fallback fn, not a giant table.
/// This is only used in fallback / non SIMD path.
const fn is_letter(cp: u32) -> bool {
    if let Some(c) = char::from_u32(cp) {
        c.is_ascii_alphabetic()
    } else {
        false
    }
}
const fn is_digit(cp: u32) -> bool {
    if let Some(c) = char::from_u32(cp) {
        c.is_ascii_digit()
    } else {
        false
    }
}
const NUMBER2: [u64; 32] = {
    let mut t = [0u64; 32];
    let mut i = 0u32;
    while i < 32 {
        let mut bitmap = 0u64;
        let mut byte2 = 0u32;
        while byte2 < 64 {
            let codepoint = i << 6 | byte2;
            if is_digit(codepoint) {
                bitmap |= 1u64 << byte2
            }
            byte2 += 1;
        }
        t[i as usize] = bitmap;
        i += 1;
    }
    t
};

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

const NUMBER3: [u64; 1024] = {
    let mut t = [0u64; 1024];
    let mut i = 0;
    while i < 1024 {
        let mut bitmap = 0u64;
        let mut byte3 = 0u32;
        while byte3 < 64 {
            let codepoint = i << 5 | byte3;
            if is_digit(codepoint) {
                bitmap |= 1u64 << byte3
            }
            byte3 += 1;
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

const fn bitmap_is_letter(cp: u32) -> bool {
    if cp < 0x80 {
        // ASCII \p{L} == A-Za-z
        (cp >= 'a' as u32 && cp <= 'z' as u32) || (cp >= 'A' as u32 && cp <= 'Z' as u32)
    } else if cp <= 0x7FF {
        (LETTER2[(cp >> 6) as usize] >> (cp & 0x3F)) & 1 == 1
    } else if cp <= 0xFFFF {
        (LETTER3[(cp >> 6) as usize] >> (cp & 0x3F)) & 1 == 1
    } else {
        // 4-byte, we fallback
        false
    }
}

const fn number2_to_hit(l: u8, b2: u8) -> bool {
    (NUMBER2[(l & 0x1F) as usize] >> (b2 & 0x3F)) & 1 == 1
}
const fn letter2_hit(l: u8, b2: u8) -> bool {
    (LETTER2[(l & 0x1F) as usize] >> (b2 & 0x3F)) & 1 == 1
}
// TODO: now that I got the algo, I need to test every single codepoint?
fn main() {
    println!("Generating bitmap table for fast nibble based splitting");
    let letter = "é".as_bytes();
    let lead = letter[0] - 0xC0;
    let shift = letter[1] & 0x3F;
    assert_eq!(LETTER2[lead as usize] >> shift & 1, 1);
    use std::fs::File;
    use std::io::IoSlice;
    use std::path::Path;
    let no_letter = "|".as_bytes();
    let lead = no_letter[0] - 0xC0;
    let shift = no_letter[1] & 0x3F;
    assert_eq!(LETTER2[lead as usize] >> shift & 1, 0);
    assert_eq!(letter2_hit(no_letter[0], no_letter[1]), false);
    let three_byte_letter = "ত".as_bytes();
    let lead = three_byte_letter[0] - 0xE0 << 6;
    let second = three_byte_letter[2] - 0x3F;
    let shift = three_byte_letter[3] & 0x3F;
    assert_eq!(LETTER3[(lead | second) as usize] >> shift & 1, 0);
    // TEST ON ALL UNICODE:
    for cp in 0x80u32..=0xFFFF {
        let expect = is_letter(cp);
        let got = bitmap_is_letter(cp); // encode cp to bytes, index LETTER2/3
        assert_eq!(got, expect, "cp {cp:#06x}");
    }
    let path = Path::new("../../src/unicode_ranges.rs");
    let display = path.display();
    let mut file = match File::create(&path) {
        Err(why) => panic!("couldn't create {}: {}", display, why),
        Ok(file) => file,
    };
    let mut data = String::new();
    data.push_str(&format!("const LETTER2: [u64, 32] = ").to_string());
    match file.write_all(data.as_bytes()) {
        Err(why) => panic!("couldn't write to {}: {}", display, why),
        Ok(_) => println!("successfully wrote to {}", display),
    }
}
