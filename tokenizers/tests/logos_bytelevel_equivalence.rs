//! Proves the logos-backed ByteLevel `Pattern` implementation produces
//! byte-for-byte identical splits to the legacy `SysRegex` pattern.
//!
//! Runs only when the `logos-pretok` feature is active, which is the default.
//! The comparison drives both `Pattern` impls directly on the same input
//! string so the check is independent of the rest of the pre-tokenization
//! pipeline.
//!
//! Corpus: `data/big.txt` line by line (≈ 130k lines, mix of English prose)
//! plus an inline adversarial table covering the edge cases the
//! `\s+(?!\S)` lookahead interacts with.

#![cfg(feature = "logos-pretok")]

use std::sync::LazyLock;
use tokenizers::pre_tokenizers::byte_level::LogosByteLevel;
use tokenizers::tokenizer::pattern::Pattern;
use tokenizers::utils::SysRegex;

static LEGACY_RE: LazyLock<SysRegex> = LazyLock::new(|| {
    SysRegex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
        .unwrap()
});

fn assert_equivalent(input: &str) {
    let sys = &*LEGACY_RE;
    let logos = &LogosByteLevel;
    let legacy = sys.find_matches(input).unwrap();
    let new = logos.find_matches(input).unwrap();
    assert_eq!(
        legacy, new,
        "split mismatch on input {:?}\n legacy={:?}\n logos ={:?}",
        input, legacy, new
    );
}

#[test]
fn adversarial_table() {
    let inputs: &[&str] = &[
        // Empty & trivial
        "",
        " ",
        "\n",
        "\t",
        "a",
        // Whitespace lookahead edge cases — the whole point of `\s+(?!\S)`
        "a b",         // single ws between letters (lookahead fails → `\s+` wins, but ` ?\p{L}+` consumes it first)
        "a  b",        // two ws — backtrack leaves 1 for next token's prefix
        "a   b",       // three ws — backtrack leaves 1 for next token's prefix
        "a       b",   // long ws run
        "a ",          // trailing single ws at EOF
        "a  ",         // trailing multi ws at EOF (no backtrack — no next token)
        "a \n b",      // mixed ws kinds
        "\n\n\n",      // only ws
        "   hello",    // leading ws
        // Contractions
        "don't",
        "it's",
        "we're",
        "I've",
        "I'm",
        "I'll",
        "I'd",
        "he'd've",
        // Numbers
        "123",
        "a 123",
        "12 34 56",
        "mix3d",       // digits inside letters — regex splits
        "123abc",      // digits then letters
        // Punctuation / "other"
        "hello, world",
        "a...b",
        "--",
        "!@#$%",
        // Unicode — letters & numbers via \p{L} \p{N}
        "café",
        "naïve",
        "𝔾𝕠𝕠𝕕 𝕞𝕠𝕣𝕟𝕚𝕟𝕘",
        "日本語テスト",
        "한국어",
        "עברית",
        "मुझे",       // Devanagari with combining marks
        "i⭢j",
        // Mixed Unicode + ws lookahead
        "café   latte",
        "123    日本",
        // GPT-2 canonical examples
        "Hello my friend, how is your day going?",
        "Hello there\nHello there",
        "Hello there       dear",
    ];
    for s in inputs {
        assert_equivalent(s);
    }
}

#[test]
fn big_txt_corpus() {
    let path = std::path::Path::new("data/big.txt");
    if !path.exists() {
        // Skip gracefully when the corpus isn't present (e.g. lightweight CI).
        eprintln!("data/big.txt not found — skipping corpus sweep");
        return;
    }
    let data = std::fs::read_to_string(path).expect("read big.txt");
    for (i, line) in data.lines().enumerate() {
        let sys = &*LEGACY_RE;
        let logos = &LogosByteLevel;
        let legacy = sys.find_matches(line).unwrap();
        let new = logos.find_matches(line).unwrap();
        assert_eq!(
            legacy, new,
            "big.txt line {} {:?}",
            i, line
        );
    }
}
