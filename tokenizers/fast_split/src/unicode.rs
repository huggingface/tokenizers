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
const fn bitmap_is_number(cp: u32) -> bool {
    if cp < 0x80 {
        // ASCII \p{N} == 0-9
        cp >= '0' as u32 && cp <= '9' as u32
    } else if cp <= 0x7FF {
        (NUMBER2[(cp >> 6) as usize] >> (cp & 0x3F)) & 1 == 1
    } else if cp <= 0xFFFF {
        (NUMBER3[(cp >> 6) as usize] >> (cp & 0x3F)) & 1 == 1
    } else {
        // 4-byte, we fallback
        false
    }
}

pub const fn number2_hit(l: u8, b2: u8) -> bool {
    (NUMBER2[(l & 0x1F) as usize] >> (b2 & 0x3F)) & 1 == 1
}
pub const fn letter2_hit(l: u8, b2: u8) -> bool {
    (LETTER2[(l & 0x1F) as usize] >> (b2 & 0x3F)) & 1 == 1
}
pub const fn letter3_hit(l: u8, b2: u8, b3: u8) -> bool {
    let row = ((l & 0x0F) as usize) << 6 | (b2 & 0x3F) as usize;
    (LETTER3[row] >> (b3 & 0x3F)) & 1 == 1
}
pub const fn number3_hit(l: u8, b2: u8, b3: u8) -> bool {
    let row = ((l & 0x0F) as usize) << 6 | (b2 & 0x3F) as usize;
    (NUMBER3[row] >> (b3 & 0x3F)) & 1 == 1
}
#[cfg(all(test, feature = "unicode"))]
mod test {
    use super::*;
    use unicode_categories::UnicodeCategories;

    #[test]
    fn hit_helpers_spot_checks() {
        let e = "é".as_bytes(); // U+00E9, 2-byte letter
        assert!(letter2_hit(e[0], e[1]));
        assert!(!number2_hit(e[0], e[1]));
        let x = "×".as_bytes(); // U+00D7, 2-byte symbol (not a letter)
        assert!(!letter2_hit(x[0], x[1]));
        let t = "ত".as_bytes(); // U+09A4, 3-byte Bengali letter
        assert!(letter3_hit(t[0], t[1], t[2]));
    }

    #[test]
    fn readers_match_unicode_categories() {
        let mut buf = [0u8; 4];
        for cp in 0u32..=0xFFFF {
            let Some(c) = char::from_u32(cp) else {
                continue;
            };
            let (want_l, want_n) = (c.is_letter(), c.is_number());
            assert_eq!(bitmap_is_letter(cp), want_l, "bitmap letter {cp:#06x}");
            assert_eq!(bitmap_is_number(cp), want_n, "bitmap number {cp:#06x}");
            let b = c.encode_utf8(&mut buf).as_bytes();
            let (got_l, got_n) = match b.len() {
                2 => (letter2_hit(b[0], b[1]), number2_hit(b[0], b[1])),
                3 => (letter3_hit(b[0], b[1], b[2]), number3_hit(b[0], b[1], b[2])),
                _ => continue, // ASCII covered by bitmap_is_* above; 4-byte not tabled
            };
            assert_eq!(got_l, want_l, "hit letter {cp:#06x}");
            assert_eq!(got_n, want_n, "hit number {cp:#06x}");
        }
    }
}
