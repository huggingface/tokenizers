//! A large input encoded in cut chunks must match the serial encode (`encode_into`).

use tk_encode::pipeline::{EncodeOptions, PARALLEL_MIN_BYTES, PipelineTokenizer};

const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../data");

fn load(file: &str) -> PipelineTokenizer {
    let canonical = tk_convert::canonicalize_file(format!("{DATA}/{file}")).unwrap();
    tk_serialize::from_json(&canonical).unwrap()
}

/// About `len` bytes of `source`, repeated, cut at a char boundary.
fn sized(source: &str, len: usize) -> String {
    let mut out = String::with_capacity(len + source.len());
    while out.len() < len {
        out.push_str(source);
    }
    let mut end = len;
    while !out.is_char_boundary(end) {
        end -= 1;
    }
    out.truncate(end);
    out
}

fn base64(len: usize) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ALPHABET[(state >> 58) as usize] as char
        })
        .collect()
}

fn texts(special: &str) -> Vec<(&'static str, String)> {
    let len = 16 * PARALLEL_MIN_BYTES;
    let english = std::fs::read_to_string(format!("{DATA}/big.txt")).unwrap();
    let cjk = std::fs::read_to_string(format!("{DATA}/unigram_wagahaiwa_nekodearu.txt")).unwrap();
    let xnli = std::fs::read_to_string(format!("{DATA}/xnli.txt")).unwrap();
    let code = include_str!("../../tk-encode/src/tokenizer/pipeline/parallel.rs");
    let mut with_specials = sized(&english[..len], len);
    for at in (0..8).map(|i| i * len / 8) {
        let mut at = at;
        while !with_specials.is_char_boundary(at) {
            at += 1;
        }
        with_specials.insert_str(at, special);
    }
    vec![
        ("english", sized(&english, len)),
        ("cjk", sized(&cjk, len)),
        ("multilingual", sized(&xnli, len)),
        ("code", sized(code, len)),
        ("base64", base64(len)),
        (
            "whitespace runs",
            sized("a  b\t\t c\n\n\n  d   \r\n e      ", len),
        ),
        ("digits", sized("abc 1234567 x9y 3.14159 ", len)),
        (
            "contractions",
            sized("don't we'll they've I'M it's y'all'd've ", len),
        ),
        ("specials", with_specials),
    ]
}

fn assert_cut_encode_matches_serial(file: &str, special: &str) {
    let tokenizer = load(file);
    let options = EncodeOptions::no_specials();
    for (name, text) in texts(special) {
        let mut serial = Vec::new();
        tokenizer.encode_into(&text, &options, &mut serial).unwrap();
        let parallel = tokenizer.encode(text.as_str(), &options).wait().unwrap();
        assert!(
            parallel[0].ids() == serial.as_slice(),
            "{file}/{name}: {} parallel ids vs {} serial",
            parallel[0].len(),
            serial.len()
        );
    }
}

#[test]
fn llama3_cut_encode_matches_serial() {
    assert_cut_encode_matches_serial("llama-3-tokenizer.json", "<|reserved_special_token_0|>");
}

#[test]
fn roberta_cut_encode_matches_serial() {
    assert_cut_encode_matches_serial("roberta.json", "<|endoftext|>");
}

/// Control: not cut (normalized `<|endoftext|>`).
#[test]
fn gpt2_encode_matches_serial() {
    assert_cut_encode_matches_serial("gpt2.json", "<|endoftext|>");
}

/// Control: not cut (no proven deepseek cut, has a normalizer).
#[test]
fn deepseek_encode_matches_serial() {
    assert_cut_encode_matches_serial("deepseek-v3-tokenizer.json", "<｜User｜>");
}

/// With `encode_special_tokens`, specials stay in the text: a cut inside one must not expose a
/// shorter added token that the serial scan skips over along with the special.
#[test]
fn cuts_never_split_specials_left_in_the_text() {
    let mut json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{DATA}/gpt2.json")).unwrap())
            .unwrap();
    let token = |id: u32, content: &str, special: bool| {
        serde_json::json!({"id": id, "content": content, "special": special, "single_word": false,
            "lstrip": false, "rstrip": false, "normalized": false})
    };
    json["added_tokens"] = serde_json::json!([
        token(50256, "<|endoftext|>", true),
        token(50257, "<|endof", false)
    ]);
    let canonical = tk_convert::canonicalize_str(&json.to_string()).unwrap();
    let tokenizer = tk_serialize::from_json(&canonical).unwrap();
    let options = EncodeOptions {
        encode_special_tokens: true,
        ..EncodeOptions::no_specials()
    };
    let text = "<|endoftext|>".repeat(4 * PARALLEL_MIN_BYTES / 13);
    let mut serial = Vec::new();
    tokenizer.encode_into(&text, &options, &mut serial).unwrap();
    let parallel = tokenizer.encode(text.as_str(), &options).wait().unwrap();
    assert!(parallel[0].ids() == serial.as_slice());
}
