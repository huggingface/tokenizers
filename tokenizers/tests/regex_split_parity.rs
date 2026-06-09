use tokenizers::Tokenizer;

#[cfg(feature = "pcre2")]
const BACKEND: &str = "pcre2";
#[cfg(all(feature = "onig", not(feature = "pcre2")))]
const BACKEND: &str = "onig";
#[cfg(all(
    feature = "fancy-regex",
    not(any(feature = "onig", feature = "pcre2"))
))]
const BACKEND: &str = "fancy-regex";

struct Case {
    input: &'static str,
    tokens: &'static [&'static str],
    offsets: &'static [(usize, usize)],
}

fn run_cases(tokenizer_path: &str, cases: &[Case]) {
    let tokenizer = Tokenizer::from_file(tokenizer_path).expect("load tokenizer fixture");
    for Case { input, tokens, offsets } in cases {
        let encoding = tokenizer.encode(*input, false).expect("encode");
        let actual_tokens: Vec<&str> = encoding.get_tokens().iter().map(String::as_str).collect();
        assert_eq!(actual_tokens, *tokens, "backend={BACKEND} input={input:?}");
        assert_eq!(
            encoding.get_offsets(),
            *offsets,
            "backend={BACKEND} input={input:?}"
        );
    }
}

#[test]
fn bytelevel_gpt2_split_parity() {
    let cases = [
        Case {
            input: "Hello, world! It's 2026.",
            tokens: &[
                "Ġ", "H", "e", "l", "l", "o", ",", "Ġw", "o", "r", "l", "d", "!", "Ġ", "I",
                "t", "'", "s", "Ġ", "2", "0", "2", "6", ".",
            ],
            offsets: &[
                (0, 0),
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 13),
                (14, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 18),
                (19, 19),
                (19, 20),
                (20, 21),
                (21, 22),
                (22, 23),
                (23, 24),
            ],
        },
        Case {
            input: "I love \u{1f355} and \u{1f602}",
            tokens: &["Ġ", "I", "Ġ", "l", "o", "v", "e", "Ġ", "Ġa", "nd", "Ġ"],
            offsets: &[
                (0, 0),
                (0, 1),
                (2, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (7, 7),
                (12, 13),
                (13, 15),
                (16, 16),
            ],
        },
        Case {
            input: "你好，世界",
            tokens: &["Ġ"],
            offsets: &[(0, 2)],
        },
        Case {
            input: "line1\nline2\n\nline3",
            tokens: &[
                "Ġ", "l", "in", "e", "1", "l", "in", "e", "2", "l", "in", "e",
            ],
            offsets: &[
                (0, 0),
                (0, 1),
                (1, 3),
                (3, 4),
                (4, 5),
                (6, 7),
                (7, 9),
                (9, 10),
                (10, 11),
                (13, 14),
                (14, 16),
                (16, 17),
            ],
        },
        Case {
            input: "foo-bar_baz",
            tokens: &["Ġ", "f", "o", "o", "-", "b", "a", "r", "b", "a", "z"],
            offsets: &[
                (0, 0),
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (8, 9),
                (9, 10),
                (10, 11),
            ],
        },
    ];

    run_cases("data/tokenizer.json", &cases);
}

#[test]
fn llama3_split_parity() {
    let cases = [
        Case {
            input: "Hello, world! It's 2026.",
            tokens: &[
                "Hello", ",", "Ġworld", "!", "ĠIt", "'s", "Ġ", "202", "6", ".",
            ],
            offsets: &[
                (0, 5),
                (5, 6),
                (6, 12),
                (12, 13),
                (13, 16),
                (16, 18),
                (18, 19),
                (19, 22),
                (22, 23),
                (23, 24),
            ],
        },
        Case {
            input: "I love \u{1f355} and \u{1f602}",
            tokens: &["I", "Ġlove", "ĠðŁ", "į", "ķ", "Ġand", "ĠðŁĺ", "Ĥ"],
            offsets: &[
                (0, 1),
                (1, 6),
                (6, 11),
                (7, 11),
                (7, 11),
                (11, 15),
                (15, 20),
                (16, 20),
            ],
        },
        Case {
            input: "你好，世界",
            tokens: &["ä½ł", "å¥½", "ï¼Į", "ä¸ĸçķĮ"],
            offsets: &[(0, 3), (3, 6), (6, 9), (9, 15)],
        },
        Case {
            input: "line1\nline2\n\nline3",
            tokens: &["line", "1", "Ċ", "line", "2", "ĊĊ", "line", "3"],
            offsets: &[
                (0, 4),
                (4, 5),
                (5, 6),
                (6, 10),
                (10, 11),
                (11, 13),
                (13, 17),
                (17, 18),
            ],
        },
        Case {
            input: "foo-bar_baz",
            tokens: &["foo", "-bar", "_b", "az"],
            offsets: &[(0, 3), (3, 7), (7, 9), (9, 11)],
        },
    ];

    run_cases("data/llama-3-tokenizer.json", &cases);
}
