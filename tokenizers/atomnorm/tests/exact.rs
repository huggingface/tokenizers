//! Byte-exactness gates: every codepoint × mark suffixes × 4 forms vs `unicode-normalization`,
//! plus long synthetic paragraphs that exercise the SIMD kernels (dispatchless — atomnorm has no
//! size gates, every input takes the same path).
use unicode_normalization::UnicodeNormalization;

#[test]
fn exhaustive() {
    let mut buf = String::new();
    for cp in 0u32..0x30000 {
        let Some(c) = char::from_u32(cp) else { continue };
        for suffix in ["", "\u{0301}", "\u{0323}\u{0301}", "\u{0301}\u{0323}", "\u{05B4}"] {
            buf.clear();
            buf.push(c);
            buf.push_str(suffix);
            assert_eq!(atomnorm::nfd(&buf), buf.nfd().collect::<String>(), "NFD {cp:#x} {suffix:?}");
            assert_eq!(atomnorm::nfkd(&buf), buf.nfkd().collect::<String>(), "NFKD {cp:#x} {suffix:?}");
            assert_eq!(atomnorm::nfc(&buf), buf.nfc().collect::<String>(), "NFC {cp:#x} {suffix:?}");
            assert_eq!(atomnorm::nfkc(&buf), buf.nfkc().collect::<String>(), "NFKC {cp:#x} {suffix:?}");
        }
    }
}

#[test]
fn long_inputs() {
    let cases: Vec<String> = vec![
        "hello world, plain ascii that should borrow untouched. ".repeat(20),
        "café crème déjà-vu à côté ".repeat(30),
        "Ἀρχαία ἑλληνικά κείμενα ".repeat(30),
        "한국어 테스트 문장입니다 ".repeat(30),
        "\u{1100}\u{1161}\u{11A8}".repeat(100),                    // isolated jamo → compose to syllables
        "これは日本語のテキストです。がぎぐげご ".repeat(20),
        "สวัสดีครับ กุ้ง ".repeat(40),
        "\u{0E01}\u{0E49}\u{0E38}".repeat(80),                     // Thai marks out of canonical order
        "देवनागरी क्षत्रिय ".repeat(30),
        "עברית עם נִקּוּד ".repeat(30),
        "ﬁ ² Ⅻ ㍿ ， 　 fullwidth ｈｅｌｌｏ ".repeat(20),
        "e\u{0301}\u{0323}x".repeat(60),
        "a\u{0323}\u{0301}x".repeat(60),
        "mixed 世界 café Москва テスト 한글 ￦ ".repeat(20),
        "\u{2F800}\u{2F801} astral compat ".repeat(20),
        "𝅗\u{1D165}\u{1D16E} musical ".repeat(20),
    ];
    for (k, s) in cases.iter().enumerate() {
        assert_eq!(atomnorm::nfd(s.as_str()), s.nfd().collect::<String>(), "NFD case {k}");
        assert_eq!(atomnorm::nfkd(s.as_str()), s.nfkd().collect::<String>(), "NFKD case {k}");
        assert_eq!(atomnorm::nfc(s.as_str()), s.nfc().collect::<String>(), "NFC case {k}");
        assert_eq!(atomnorm::nfkc(s.as_str()), s.nfkc().collect::<String>(), "NFKC case {k}");
    }
}

#[test]
fn borrows_when_normalized() {
    use std::borrow::Cow;
    assert!(matches!(atomnorm::nfd("plain ascii"), Cow::Borrowed(_)));
    assert!(matches!(atomnorm::nfc("café precomposed"), Cow::Borrowed(_)));
    assert!(matches!(atomnorm::nfd("这是中文"), Cow::Borrowed(_)));
    assert!(matches!(atomnorm::nfc("한국어"), Cow::Borrowed(_)));
}
