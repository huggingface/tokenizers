use super::unicode_tables::{LETTER2, LETTER3, NUMBER2, NUMBER3};
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

pub const fn number2_to_hit(l: u8, b2: u8) -> bool {
    (NUMBER2[(l & 0x1F) as usize] >> (b2 & 0x3F)) & 1 == 1
}
pub const fn letter2_hit(l: u8, b2: u8) -> bool {
    (LETTER2[(l & 0x1F) as usize] >> (b2 & 0x3F)) & 1 == 1
}
pub const fn letter3_hit(l: u8, b2: u8, b3: u8) -> bool {
    (LETTER2[((l & 0x0F) << 6 | (b2 & 0x3F)) as usize] >> (b3 & 0x3F)) & 1 == 1
}
pub const fn number3_hit(l: u8, b2: u8, b3: u8) -> bool {
    (NUMBER3[((l & 0x0F) << 6 | (b2 & 0x3F)) as usize] >> (b3 & 0x3F)) & 1 == 1
}
#[cfg(test)]
pub mod test {
    use super::*;
    use unicode_categories::*;
    pub fn _simple_test() {
        let letter = "é".as_bytes();
        let lead = letter[0] & 0x1F;
        let shift = letter[1] & 0x3F;

        assert_eq!(letter2_hit(letter[0], letter[1]), true);
        assert_eq!((LETTER2[lead as usize] >> shift) & 1, 1);
        let no_letter = "×".as_bytes();
        let lead = no_letter[0] & 0x1F;
        let shift = no_letter[1] & 0x3F;
        assert_eq!(LETTER2[lead as usize] >> shift & 1, 0);
        assert_eq!(letter2_hit(no_letter[0], no_letter[1]), false);
        let three_byte_letter = "ত".as_bytes();
        let lead = three_byte_letter[0] - 0xE0 << 6;
        let second = three_byte_letter[1] - 0x3F;
        let shift = three_byte_letter[2] & 0x3F;
        assert_eq!((LETTER3[(lead | second) as usize] >> shift) & 1, 0);
        // TEST ON ALL UNICODE:
        for cp in 0x80u32..=0xFFFF {
            let expect = char::from_u32(cp).is_some_and(|c| c.is_letter());
            let got = bitmap_is_letter(cp); // encode cp to bytes, index LETTER2/3
            assert_eq!(got, expect, "cp {cp:#06x}");
        }
    }
}
