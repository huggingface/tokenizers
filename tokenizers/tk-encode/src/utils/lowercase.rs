//! Lowercasing with the Unicode 17 data of the 0.23 wheels, which were built with rustc 1.92 to 1.98.
//! `char::to_lowercase` would follow whatever toolchain builds this crate.

/// The lowercase mappings of Unicode 17 that are a single char, as runs `(first, last, step, delta)`:
/// every `step`th code point from `first` to `last` maps to itself plus `delta`. In code point order.
///
/// Rebuilt from std by `table_matches_std_on_unicode_17`.
#[rustfmt::skip]
static LOWERCASE: [(char, char, u8, i32); 185] = [
    ('\u{0041}', '\u{005A}', 1, 32), ('\u{00C0}', '\u{00D6}', 1, 32), ('\u{00D8}', '\u{00DE}', 1, 32),
    ('\u{0100}', '\u{012E}', 2, 1), ('\u{0132}', '\u{0136}', 2, 1), ('\u{0139}', '\u{0147}', 2, 1),
    ('\u{014A}', '\u{0176}', 2, 1), ('\u{0178}', '\u{0178}', 1, -121), ('\u{0179}', '\u{017D}', 2, 1),
    ('\u{0181}', '\u{0181}', 1, 210), ('\u{0182}', '\u{0184}', 2, 1), ('\u{0186}', '\u{0186}', 1, 206),
    ('\u{0187}', '\u{0187}', 1, 1), ('\u{0189}', '\u{018A}', 1, 205), ('\u{018B}', '\u{018B}', 1, 1),
    ('\u{018E}', '\u{018E}', 1, 79), ('\u{018F}', '\u{018F}', 1, 202), ('\u{0190}', '\u{0190}', 1, 203),
    ('\u{0191}', '\u{0191}', 1, 1), ('\u{0193}', '\u{0193}', 1, 205), ('\u{0194}', '\u{0194}', 1, 207),
    ('\u{0196}', '\u{0196}', 1, 211), ('\u{0197}', '\u{0197}', 1, 209), ('\u{0198}', '\u{0198}', 1, 1),
    ('\u{019C}', '\u{019C}', 1, 211), ('\u{019D}', '\u{019D}', 1, 213), ('\u{019F}', '\u{019F}', 1, 214),
    ('\u{01A0}', '\u{01A4}', 2, 1), ('\u{01A6}', '\u{01A6}', 1, 218), ('\u{01A7}', '\u{01A7}', 1, 1),
    ('\u{01A9}', '\u{01A9}', 1, 218), ('\u{01AC}', '\u{01AC}', 1, 1), ('\u{01AE}', '\u{01AE}', 1, 218),
    ('\u{01AF}', '\u{01AF}', 1, 1), ('\u{01B1}', '\u{01B2}', 1, 217), ('\u{01B3}', '\u{01B5}', 2, 1),
    ('\u{01B7}', '\u{01B7}', 1, 219), ('\u{01B8}', '\u{01B8}', 1, 1), ('\u{01BC}', '\u{01BC}', 1, 1),
    ('\u{01C4}', '\u{01C4}', 1, 2), ('\u{01C5}', '\u{01C5}', 1, 1), ('\u{01C7}', '\u{01C7}', 1, 2),
    ('\u{01C8}', '\u{01C8}', 1, 1), ('\u{01CA}', '\u{01CA}', 1, 2), ('\u{01CB}', '\u{01DB}', 2, 1),
    ('\u{01DE}', '\u{01EE}', 2, 1), ('\u{01F1}', '\u{01F1}', 1, 2), ('\u{01F2}', '\u{01F4}', 2, 1),
    ('\u{01F6}', '\u{01F6}', 1, -97), ('\u{01F7}', '\u{01F7}', 1, -56), ('\u{01F8}', '\u{021E}', 2, 1),
    ('\u{0220}', '\u{0220}', 1, -130), ('\u{0222}', '\u{0232}', 2, 1), ('\u{023A}', '\u{023A}', 1, 10795),
    ('\u{023B}', '\u{023B}', 1, 1), ('\u{023D}', '\u{023D}', 1, -163), ('\u{023E}', '\u{023E}', 1, 10792),
    ('\u{0241}', '\u{0241}', 1, 1), ('\u{0243}', '\u{0243}', 1, -195), ('\u{0244}', '\u{0244}', 1, 69),
    ('\u{0245}', '\u{0245}', 1, 71), ('\u{0246}', '\u{024E}', 2, 1), ('\u{0370}', '\u{0372}', 2, 1),
    ('\u{0376}', '\u{0376}', 1, 1), ('\u{037F}', '\u{037F}', 1, 116), ('\u{0386}', '\u{0386}', 1, 38),
    ('\u{0388}', '\u{038A}', 1, 37), ('\u{038C}', '\u{038C}', 1, 64), ('\u{038E}', '\u{038F}', 1, 63),
    ('\u{0391}', '\u{03A1}', 1, 32), ('\u{03A3}', '\u{03AB}', 1, 32), ('\u{03CF}', '\u{03CF}', 1, 8),
    ('\u{03D8}', '\u{03EE}', 2, 1), ('\u{03F4}', '\u{03F4}', 1, -60), ('\u{03F7}', '\u{03F7}', 1, 1),
    ('\u{03F9}', '\u{03F9}', 1, -7), ('\u{03FA}', '\u{03FA}', 1, 1), ('\u{03FD}', '\u{03FF}', 1, -130),
    ('\u{0400}', '\u{040F}', 1, 80), ('\u{0410}', '\u{042F}', 1, 32), ('\u{0460}', '\u{0480}', 2, 1),
    ('\u{048A}', '\u{04BE}', 2, 1), ('\u{04C0}', '\u{04C0}', 1, 15), ('\u{04C1}', '\u{04CD}', 2, 1),
    ('\u{04D0}', '\u{052E}', 2, 1), ('\u{0531}', '\u{0556}', 1, 48), ('\u{10A0}', '\u{10C5}', 1, 7264),
    ('\u{10C7}', '\u{10C7}', 1, 7264), ('\u{10CD}', '\u{10CD}', 1, 7264), ('\u{13A0}', '\u{13EF}', 1, 38864),
    ('\u{13F0}', '\u{13F5}', 1, 8), ('\u{1C89}', '\u{1C89}', 1, 1), ('\u{1C90}', '\u{1CBA}', 1, -3008),
    ('\u{1CBD}', '\u{1CBF}', 1, -3008), ('\u{1E00}', '\u{1E94}', 2, 1), ('\u{1E9E}', '\u{1E9E}', 1, -7615),
    ('\u{1EA0}', '\u{1EFE}', 2, 1), ('\u{1F08}', '\u{1F0F}', 1, -8), ('\u{1F18}', '\u{1F1D}', 1, -8),
    ('\u{1F28}', '\u{1F2F}', 1, -8), ('\u{1F38}', '\u{1F3F}', 1, -8), ('\u{1F48}', '\u{1F4D}', 1, -8),
    ('\u{1F59}', '\u{1F5F}', 2, -8), ('\u{1F68}', '\u{1F6F}', 1, -8), ('\u{1F88}', '\u{1F8F}', 1, -8),
    ('\u{1F98}', '\u{1F9F}', 1, -8), ('\u{1FA8}', '\u{1FAF}', 1, -8), ('\u{1FB8}', '\u{1FB9}', 1, -8),
    ('\u{1FBA}', '\u{1FBB}', 1, -74), ('\u{1FBC}', '\u{1FBC}', 1, -9), ('\u{1FC8}', '\u{1FCB}', 1, -86),
    ('\u{1FCC}', '\u{1FCC}', 1, -9), ('\u{1FD8}', '\u{1FD9}', 1, -8), ('\u{1FDA}', '\u{1FDB}', 1, -100),
    ('\u{1FE8}', '\u{1FE9}', 1, -8), ('\u{1FEA}', '\u{1FEB}', 1, -112), ('\u{1FEC}', '\u{1FEC}', 1, -7),
    ('\u{1FF8}', '\u{1FF9}', 1, -128), ('\u{1FFA}', '\u{1FFB}', 1, -126), ('\u{1FFC}', '\u{1FFC}', 1, -9),
    ('\u{2126}', '\u{2126}', 1, -7517), ('\u{212A}', '\u{212A}', 1, -8383), ('\u{212B}', '\u{212B}', 1, -8262),
    ('\u{2132}', '\u{2132}', 1, 28), ('\u{2160}', '\u{216F}', 1, 16), ('\u{2183}', '\u{2183}', 1, 1),
    ('\u{24B6}', '\u{24CF}', 1, 26), ('\u{2C00}', '\u{2C2F}', 1, 48), ('\u{2C60}', '\u{2C60}', 1, 1),
    ('\u{2C62}', '\u{2C62}', 1, -10743), ('\u{2C63}', '\u{2C63}', 1, -3814), ('\u{2C64}', '\u{2C64}', 1, -10727),
    ('\u{2C67}', '\u{2C6B}', 2, 1), ('\u{2C6D}', '\u{2C6D}', 1, -10780), ('\u{2C6E}', '\u{2C6E}', 1, -10749),
    ('\u{2C6F}', '\u{2C6F}', 1, -10783), ('\u{2C70}', '\u{2C70}', 1, -10782), ('\u{2C72}', '\u{2C72}', 1, 1),
    ('\u{2C75}', '\u{2C75}', 1, 1), ('\u{2C7E}', '\u{2C7F}', 1, -10815), ('\u{2C80}', '\u{2CE2}', 2, 1),
    ('\u{2CEB}', '\u{2CED}', 2, 1), ('\u{2CF2}', '\u{2CF2}', 1, 1), ('\u{A640}', '\u{A66C}', 2, 1),
    ('\u{A680}', '\u{A69A}', 2, 1), ('\u{A722}', '\u{A72E}', 2, 1), ('\u{A732}', '\u{A76E}', 2, 1),
    ('\u{A779}', '\u{A77B}', 2, 1), ('\u{A77D}', '\u{A77D}', 1, -35332), ('\u{A77E}', '\u{A786}', 2, 1),
    ('\u{A78B}', '\u{A78B}', 1, 1), ('\u{A78D}', '\u{A78D}', 1, -42280), ('\u{A790}', '\u{A792}', 2, 1),
    ('\u{A796}', '\u{A7A8}', 2, 1), ('\u{A7AA}', '\u{A7AA}', 1, -42308), ('\u{A7AB}', '\u{A7AB}', 1, -42319),
    ('\u{A7AC}', '\u{A7AC}', 1, -42315), ('\u{A7AD}', '\u{A7AD}', 1, -42305), ('\u{A7AE}', '\u{A7AE}', 1, -42308),
    ('\u{A7B0}', '\u{A7B0}', 1, -42258), ('\u{A7B1}', '\u{A7B1}', 1, -42282), ('\u{A7B2}', '\u{A7B2}', 1, -42261),
    ('\u{A7B3}', '\u{A7B3}', 1, 928), ('\u{A7B4}', '\u{A7C2}', 2, 1), ('\u{A7C4}', '\u{A7C4}', 1, -48),
    ('\u{A7C5}', '\u{A7C5}', 1, -42307), ('\u{A7C6}', '\u{A7C6}', 1, -35384), ('\u{A7C7}', '\u{A7C9}', 2, 1),
    ('\u{A7CB}', '\u{A7CB}', 1, -42343), ('\u{A7CC}', '\u{A7DA}', 2, 1), ('\u{A7DC}', '\u{A7DC}', 1, -42561),
    ('\u{A7F5}', '\u{A7F5}', 1, 1), ('\u{FF21}', '\u{FF3A}', 1, 32), ('\u{10400}', '\u{10427}', 1, 40),
    ('\u{104B0}', '\u{104D3}', 1, 40), ('\u{10570}', '\u{1057A}', 1, 39), ('\u{1057C}', '\u{1058A}', 1, 39),
    ('\u{1058C}', '\u{10592}', 1, 39), ('\u{10594}', '\u{10595}', 1, 39), ('\u{10C80}', '\u{10CB2}', 1, 64),
    ('\u{10D50}', '\u{10D65}', 1, 32), ('\u{118A0}', '\u{118BF}', 1, 32), ('\u{16E40}', '\u{16E5F}', 1, 32),
    ('\u{16EA0}', '\u{16EB8}', 1, 27), ('\u{1E900}', '\u{1E921}', 1, 34),
];

/// The one code point that lowercases to two chars.
const DOTTED_CAPITAL_I: char = '\u{130}';

fn lowercase(c: char) -> (char, Option<char>) {
    if c.is_ascii() {
        return (c.to_ascii_lowercase(), None);
    }
    if c == DOTTED_CAPITAL_I {
        return ('i', Some('\u{307}'));
    }
    let i = LOWERCASE.partition_point(|&(first, ..)| first <= c);
    if let Some(&(first, last, step, delta)) = i.checked_sub(1).map(|i| &LOWERCASE[i])
        && c <= last
        && (c as u32 - first as u32).is_multiple_of(u32::from(step))
    {
        let lower = char::from_u32((c as u32).wrapping_add_signed(delta));
        return (lower.expect("the table maps to valid chars"), None);
    }
    (c, None)
}

/// `char::to_lowercase`, on Unicode 17 whatever the toolchain.
pub(crate) fn to_lowercase(c: char) -> impl Iterator<Item = char> {
    let (first, second) = lowercase(c);
    std::iter::once(first).chain(second)
}

/// Whether lowercasing `c` leaves it unchanged (a single, identical char)
pub(crate) fn lowercases_to_self(c: char) -> bool {
    lowercase(c) == (c, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rebuilds [`LOWERCASE`] from std, which only gives the 0.23 data on Unicode 17. When it fails, paste
    /// the table it prints.
    ///
    ///   cargo +1.98.0 test -p tk-encode --lib table_matches_std_on_unicode_17 -- --ignored
    #[test]
    #[ignore = "needs a toolchain on Unicode 17 (rustc 1.92 to 1.99)"]
    fn table_matches_std_on_unicode_17() {
        assert_eq!(
            char::UNICODE_VERSION,
            (17, 0, 0),
            "run it with rustc 1.92 to 1.99"
        );
        let mut runs: Vec<(char, char, u8, i32)> = Vec::new();
        for c in (0..=char::MAX as u32).filter_map(char::from_u32) {
            let mut lower = c.to_lowercase();
            let (Some(l), None) = (lower.next(), lower.next()) else {
                assert_eq!(c, DOTTED_CAPITAL_I, "a new multi-char lowercase mapping");
                continue;
            };
            if l == c {
                continue;
            }
            let delta = l as i32 - c as i32;
            if let Some((first, last, step, d)) = runs.last_mut() {
                let gap = c as u32 - *last as u32;
                if *d == delta && (gap == u32::from(*step) || (*first == *last && gap <= 2)) {
                    *step = gap as u8;
                    *last = c;
                    continue;
                }
            }
            runs.push((c, c, 1, delta));
        }
        if runs != LOWERCASE {
            let rows: Vec<String> = runs
                .iter()
                .map(|&(first, last, step, delta)| {
                    format!(
                        "('\\u{{{:04X}}}', '\\u{{{:04X}}}', {step}, {delta})",
                        first as u32, last as u32
                    )
                })
                .collect();
            panic!("{} runs: [\n{}\n]", rows.len(), rows.join(",\n"));
        }
    }
}
