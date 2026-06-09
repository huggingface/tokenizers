//! Proves the logos cl100k `Pattern` impl is byte-for-byte equivalent to
//! the `SysRegex` implementation of the same pattern used by Llama-3 and
//! tiktoken-family tokenizers.
//!
//! Pattern under test (verbatim from `data/llama-3-tokenizer.json`):
//! ```text
//! (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//! ```

#![cfg(feature = "logos-pretok")]

use std::sync::LazyLock;
use tokenizers::pre_tokenizers::logos_tiktoken::{LogosCl100k, CL100K_PATTERN};
use tokenizers::tokenizer::pattern::Pattern;
use tokenizers::utils::SysRegex;

static LEGACY_RE: LazyLock<SysRegex> = LazyLock::new(|| SysRegex::new(CL100K_PATTERN).unwrap());

fn assert_equivalent(input: &str) {
    let sys = &*LEGACY_RE;
    let logos = &LogosCl100k;
    let legacy = sys.find_matches(input).unwrap();
    let new = logos.find_matches(input).unwrap();
    assert_eq!(
        legacy, new,
        "split mismatch on {:?}\n legacy={:?}\n logos ={:?}",
        input, legacy, new
    );
}

#[test]
fn adversarial_table() {
    let inputs: &[&str] = &[
        "",
        " ",
        "\n",
        "hello",
        "hello world",
        "hello  world",
        "  hello",
        "hello  ",
        "a  'tis b",
        "don't",
        "DON'T",           // case-insensitive contraction
        "  (hello",        // non-ws prefix into Letters
        "  123",           // ws + Numbers (no prefix)
        "   123",
        "12345",           // Numbers{1,3} splits
        "1",
        "12",
        "1234",
        "123 456",
        "\r\n",
        "a\r\nb",
        "a\n\n\nb",
        "  \nhello",
        "\n   \n",
        "She's",
        "SHE'S",
        "hello, world",
        "--hello",
        "$100",
        "café",
        "日本語",
        "  日本語  テスト",
        "I'm going to the store, don't you think?",
    ];
    for s in inputs {
        assert_equivalent(s);
    }
}

#[test]
fn big_txt_corpus() {
    let path = std::path::Path::new("data/big.txt");
    if !path.exists() {
        eprintln!("data/big.txt not found — skipping corpus sweep");
        return;
    }
    let data = std::fs::read_to_string(path).expect("read big.txt");
    let sys = &*LEGACY_RE;
    let logos = &LogosCl100k;
    let mut first_fail: Option<(usize, String)> = None;
    for (i, line) in data.lines().enumerate() {
        let legacy = sys.find_matches(line).unwrap();
        let new = logos.find_matches(line).unwrap();
        if legacy != new {
            first_fail = Some((i, line.to_string()));
            // Print context and diff for the first failing line
            let idx = legacy
                .iter()
                .zip(new.iter())
                .position(|(a, b)| a != b)
                .unwrap();
            let pos = legacy[idx].0 .0.min(new[idx].0 .0);
            let start = pos.saturating_sub(20);
            let end = (pos + 20).min(line.len());
            eprintln!(
                "line {} mismatch at byte {}\n  context: {:?}\n  legacy: {:?}\n  logos:  {:?}",
                i,
                pos,
                line.get(start..end).unwrap_or("<cut>"),
                &legacy[idx..(idx + 3).min(legacy.len())],
                &new[idx..(idx + 3).min(new.len())],
            );
            break;
        }
    }
    if let Some((i, line)) = first_fail {
        panic!("first mismatch on line {}: {:?}", i, line);
    }
}
