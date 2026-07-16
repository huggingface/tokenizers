//! Byte-exactness of the scan normalizers against verbatim replicas of the legacy tk-encode
//! implementations (same crates, same predicates), on every codepoint in context — both paths.
use std::borrow::Cow;
use unicode_categories::UnicodeCategories;
use unicode_normalization::UnicodeNormalization;

fn lowercases_to_self(c: char) -> bool {
    let mut it = c.to_lowercase();
    matches!((it.next(), it.next()), (Some(first), None) if first == c)
}

fn ref_lowercase(input: &str) -> String {
    input.chars().flat_map(|c| c.to_lowercase()).collect()
}

fn ref_strip_accents(input: &str) -> String {
    input
        .chars()
        .filter(|&c| !unicode_normalization_alignments::char::is_combining_mark(c))
        .collect()
}

fn nmt_removes(c: char) -> bool {
    matches!(c as u32,
        0x0001..=0x0008 | 0x000B | 0x000E..=0x001F | 0x007F | 0x008F | 0x009F)
}
fn nmt_to_space(c: char) -> char {
    match c as u32 {
        0x0009
        | 0x000A
        | 0x000C
        | 0x000D
        | 0x1680
        | 0x200B..=0x200F
        | 0x2028
        | 0x2029
        | 0x2581
        | 0xFEFF
        | 0xFFFD => ' ',
        _ => c,
    }
}
fn ref_nmt(input: &str) -> String {
    input
        .chars()
        .filter(|&c| !nmt_removes(c))
        .map(nmt_to_space)
        .collect()
}

// the legacy BertNormalizer pipeline chain, verbatim
fn is_whitespace(c: char) -> bool {
    match c {
        '\t' | '\n' | '\r' => true,
        _ => c.is_whitespace(),
    }
}
fn is_control(c: char) -> bool {
    match c {
        '\t' | '\n' | '\r' => false,
        _ => c.is_other(),
    }
}
fn clean_text_removes(c: char) -> bool {
    c == '\0' || c == '\u{fffd}' || is_control(c)
}
fn clean_text_map(c: char) -> char {
    if is_whitespace(c) { ' ' } else { c }
}
fn is_chinese_char(c: char) -> bool {
    matches!(c as usize,
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2A6DF | 0x2A700..=0x2B73F |
        0x2B740..=0x2B81F | 0x2B920..=0x2CEAF | 0xF900..=0xFAFF | 0x2F800..=0x2FA1F)
}
fn ref_bert(input: &str, clean: bool, chinese: bool, strip: bool, lower: bool) -> String {
    let cleaned = input
        .chars()
        .filter(|&c| !(clean && clean_text_removes(c)))
        .flat_map(|c| {
            let c = if clean { clean_text_map(c) } else { c };
            if chinese && is_chinese_char(c) {
                [Some(' '), Some(c), Some(' ')]
            } else {
                [Some(c), None, None]
            }
        })
        .flatten();
    let mut normalized = String::with_capacity(input.len());
    match (strip, lower) {
        (true, true) => normalized.extend(
            cleaned
                .nfd()
                .filter(|c| !c.is_mark_nonspacing())
                .flat_map(char::to_lowercase),
        ),
        (true, false) => normalized.extend(cleaned.nfd().filter(|c| !c.is_mark_nonspacing())),
        (false, true) => normalized.extend(cleaned.flat_map(char::to_lowercase)),
        (false, false) => normalized.extend(cleaned),
    }
    normalized
}

fn check(what: &str, input: &str, expect: &str, simd: Cow<str>, sclr: Cow<str>) {
    assert_eq!(&*simd, expect, "{what} (simd) diverged on {input:?}");
    assert_eq!(&*sclr, expect, "{what} (scalar) diverged on {input:?}");
}

#[test]
fn exhaustive_simple() {
    let mut buf = String::new();
    for cp in 0u32..0x110000 {
        let Some(c) = char::from_u32(cp) else {
            continue;
        };
        for ctx in [0, 1] {
            buf.clear();
            if ctx == 0 {
                buf.push(c);
            } else {
                buf.push('A');
                buf.push(c);
                buf.push('\u{0301}');
                buf.push('b');
            }
            check(
                "lowercase",
                &buf,
                &ref_lowercase(&buf),
                atomnorm::lowercase(&buf),
                atomnorm::scalar::lowercase(&buf),
            );
            check(
                "strip_accents",
                &buf,
                &ref_strip_accents(&buf),
                atomnorm::strip_accents(&buf),
                atomnorm::scalar::strip_accents(&buf),
            );
            check(
                "nmt",
                &buf,
                &ref_nmt(&buf),
                atomnorm::nmt(&buf),
                atomnorm::scalar::nmt(&buf),
            );
        }
    }
}

#[test]
fn exhaustive_bert() {
    // every codepoint inside a context exercising case, cluster adjacency, ASCII, and CJK — under
    // the 4 flag corners that route through different fused paths
    let combos = [
        (true, true, true, true),
        (true, false, false, false),
        (false, false, true, true),
        (false, true, false, true),
    ];
    let mut buf = String::new();
    for cp in 0u32..0x110000 {
        let Some(c) = char::from_u32(cp) else {
            continue;
        };
        buf.clear();
        buf.push('a');
        buf.push(c);
        buf.push('\u{0301}');
        buf.push('B');
        buf.push('中');
        for (ct, cc, sa, lc) in combos {
            let expect = ref_bert(&buf, ct, cc, sa, lc);
            check(
                "bert",
                &buf,
                &expect,
                atomnorm::bert(&buf, ct, cc, sa, lc),
                atomnorm::scalar::bert(&buf, ct, cc, sa, lc),
            );
        }
    }
}

#[test]
fn bert_all_flag_combos() {
    // all 16 configs over inputs covering every rule interaction, including the adversarial ones:
    // cross-char reorder after decomposition, clean-removed chars transparent to the nfd cluster
    let inputs = [
        "Héllo World",
        "中文字 test",
        "a中b文c",
        "  spaced\tout  ",
        "MiXeD Café ÀÉÎ",
        "e\u{0301}\u{0323}",         // marks out of canonical order
        "É\u{0323}x",                // decomposition + following mark must reorder as one cluster
        "e\u{0301}\u{200D}\u{0323}", // ZWJ (clean-removed) inside a mark cluster
        "null\0here \u{fffd} ctrl\u{0007}",
        "İstanbul ǅungla straße",
        "한국어 テスト ﬁ ½",
        "�ievề Việt Nam ế ᾈ",
        "\u{1D400}\u{1D160}\u{10400}𝒜", // astral: math bold, music (astral marks), Deseret upper
        "\u{2F800}\u{20000}",           // astral CJK: compat ideo (decomposes) + ext-B
        "\t\n\r\u{000B}\u{2028}\u{00A0}",
        "",
    ];
    for input in inputs {
        for k in 0u32..16 {
            let (ct, cc, sa, lc) = (k & 1 != 0, k & 2 != 0, k & 4 != 0, k & 8 != 0);
            let expect = ref_bert(input, ct, cc, sa, lc);
            check(
                &format!("bert[{k:04b}]"),
                input,
                &expect,
                atomnorm::bert(input, ct, cc, sa, lc),
                atomnorm::scalar::bert(input, ct, cc, sa, lc),
            );
        }
    }
}

#[test]
fn long_inputs_scan() {
    let cases: Vec<String> = vec![
        "The Quick Brown Fox JUMPS over the lazy dog 0123456789. ".repeat(40),
        "Voix ambiguë d'un CŒUR qui, au zéphyr, préfère les jattes de KIWIS. ".repeat(30),
        "СЪЕШЬ ещё этих мягких французских булок, да выпей же чаю. ".repeat(30),
        "Ξεσκεπάζω τὴν ΨΥΧΟΦΘΌΡΑ βδελυγμία. ".repeat(30),
        "עִבְרִית עם נִקּוּד וּטְעָמִים ".repeat(40),
        "देवनागरी क्षत्रिय संस्कृतम् ".repeat(40),
        "中文测试，这是一个句子。English MIXED in 中间。".repeat(30),
        "한국어 테스트 문장입니다 Awesome ".repeat(30),
        "\tTabs\tand\nnewlines\r\nand \u{200B}zero\u{FEFF}width\u{2028}breaks ".repeat(30),
        "café ﬁnance ǅungla İstanbul STRASSE µ-metre ".repeat(30),
    ];
    for (k, s) in cases.iter().enumerate() {
        check(
            &format!("lowercase case {k}"),
            "…",
            &ref_lowercase(s),
            atomnorm::lowercase(s),
            atomnorm::scalar::lowercase(s),
        );
        check(
            &format!("strip_accents case {k}"),
            "…",
            &ref_strip_accents(s),
            atomnorm::strip_accents(s),
            atomnorm::scalar::strip_accents(s),
        );
        check(
            &format!("nmt case {k}"),
            "…",
            &ref_nmt(s),
            atomnorm::nmt(s),
            atomnorm::scalar::nmt(s),
        );
        for k2 in 0u32..16 {
            let (ct, cc, sa, lc) = (k2 & 1 != 0, k2 & 2 != 0, k2 & 4 != 0, k2 & 8 != 0);
            check(
                &format!("bert case {k} [{k2:04b}]"),
                "…",
                &ref_bert(s, ct, cc, sa, lc),
                atomnorm::bert(s, ct, cc, sa, lc),
                atomnorm::scalar::bert(s, ct, cc, sa, lc),
            );
        }
    }
}

#[test]
fn scan_borrows_when_noop() {
    for (name, r) in [
        (
            "lower",
            atomnorm::lowercase("already lowercase ascii 123") as Cow<str>,
        ),
        (
            "strip",
            atomnorm::strip_accents("no marks here, café is fine") as Cow<str>,
        ),
        ("nmt", atomnorm::nmt("plain text with spaces") as Cow<str>),
        (
            "bert",
            atomnorm::bert("plain lowercase, no cjk", true, true, true, true),
        ),
    ] {
        assert!(matches!(r, Cow::Borrowed(_)), "{name} should borrow");
    }
    // owned when a rule fires
    assert!(matches!(atomnorm::lowercase("Caps"), Cow::Owned(_)));
    assert!(matches!(
        atomnorm::bert("中", true, true, true, true),
        Cow::Owned(_)
    ));
    // sanity vs the reference gate: all-noop iff every char lowercases to itself
    assert!("already lowercase ascii".chars().all(lowercases_to_self));
}
