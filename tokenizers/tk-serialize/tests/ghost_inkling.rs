//! `prefix_atoms` on the Inkling-2 vocab: tokens the merge list cannot build are emitted whole
//! at a pretoken start, and nothing else moves. Skipped when the fixture is absent.
//!
//! The fixture is `tokenizer.json` with `"prefix_atoms": true` added to the model.

use std::path::Path;

use tk_encode::pipeline::{EncodeOptions, Tokenizer};

const BASE: &str = "/tmp/inkling/tokenizer.json";
const ATOMS: &str = "/tmp/inkling/tokenizer_prefix.json";

fn load(path: &str) -> Tokenizer {
    let canonical = tk_convert::canonicalize_file(path).unwrap();
    tk_serialize::from_json(&canonical).unwrap()
}

fn ids(pipe: &Tokenizer, text: &str) -> Vec<u32> {
    pipe.encode(text, &EncodeOptions::no_specials())
        .wait()
        .unwrap()
        .remove(0)
        .ids()
        .iter()
        .map(|t| t.id())
        .collect()
}

#[test]
fn unreachable_tokens_are_atoms_at_a_pretoken_start() {
    if !Path::new(BASE).exists() || !Path::new(ATOMS).exists() {
        return;
    }
    let base = load(BASE);
    let atoms = load(ATOMS);

    assert_eq!(ids(&atoms, "Klebsiella"), vec![53960]);
    assert_eq!(ids(&atoms, "Klebsiellae"), vec![53960, 68]);
    assert_eq!(ids(&atoms, "についてください"), vec![91149, 103413]);
    assert_eq!(ids(&atoms, "Rijksmuseum"), vec![55562, 32876]);

    // Every unreachable token is emitted when it is a prefix of a pretoken.
    if let Ok(raw) = std::fs::read_to_string("/tmp/inkling/targets.json") {
        let targets: Vec<(u32, String)> = serde_json::from_str(&raw).unwrap();
        for (id, token) in targets {
            let text = format!("{token}x");
            let got = ids(&atoms, &text);
            assert_eq!(got.first(), Some(&id), "not an atom: {token:?} -> {got:?}");
        }
    }

    // Not at a pretoken start, so nothing fires: byte-identical to the base model.
    for text in [
        "hello world",
        "my Klebsiella",
        "my Klebsiellae",
        "xKlebsiella",
        "The quick brown fox jumps over the lazy dog.",
        "abcdefghij",
        "Darts-clone を MeCab や ChaSen で利用する方法を紹介します．",
        "品詞や読みに割り当てられている領域の方がずっと大きいようで，",
        "def main():\n    return 0\n",
    ] {
        assert_eq!(ids(&atoms, text), ids(&base, text), "moved: {text:?}");
    }
}
