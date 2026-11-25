//! The ghost-merge path on the Inkling-2 vocab: roots encode whole at a pretoken start and
//! nothing else moves. Skipped when the fixture is absent (it is not committed).
//!
//! Regenerate with `python /tmp/inkling/make_ghosts.py tokenizer.json tokenizer_ghost.json`.

use std::path::Path;

use tk_encode::pipeline::{EncodeOptions, Tokenizer};

const BASE: &str = "/tmp/inkling/tokenizer.json";
const GHOST: &str = "/tmp/inkling/tokenizer_ghost.json";

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
fn roots_encode_whole_and_nothing_else_moves() {
    if !Path::new(BASE).exists() || !Path::new(GHOST).exists() {
        return;
    }
    let base = load(BASE);
    let ghost = load(GHOST);

    // Roots: the tokens the vocab could not build. The reference emits them whole.
    assert_eq!(ids(&ghost, "Klebsiella"), vec![53960]);
    assert_eq!(ids(&ghost, "Klebsiellae"), vec![53960, 68]);
    assert_eq!(ids(&ghost, "についてください"), vec![91149, 103413]);
    // Another root mid-word: "Rijks" is one of the unreachable tokens.
    assert_eq!(ids(&ghost, "Rijksmuseum"), vec![55562, 32876]);

    // Every unreachable token is now a root: as a prefix of a larger word it is emitted whole.
    if let Ok(raw) = std::fs::read_to_string("/tmp/inkling/targets.json") {
        let targets: Vec<(u32, String)> = serde_json::from_str(&raw).unwrap();
        for (id, token) in targets {
            let text = format!("{token}x");
            let got = ids(&ghost, &text);
            assert_eq!(
                got.first(),
                Some(&id),
                "root not emitted for {token:?}: {got:?}"
            );
        }
    }

    // Not at a pretoken start, so the ghost never fires: byte-identical to the base model.
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
        assert_eq!(ids(&ghost, text), ids(&base, text), "moved: {text:?}");
    }
}
