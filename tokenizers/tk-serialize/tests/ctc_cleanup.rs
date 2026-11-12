#![cfg(feature = "wordlevel")]

use tk_encode::pipeline::PipelineTokenizer;
use tk_serialize::from_json;

fn ctc_tokenizer(cleanup: bool, delimiter: &str) -> PipelineTokenizer {
    let tokens = ["<unk>", "<pad>", "a", "b", delimiter, " .", "你", "好"];
    let vocab: serde_json::Map<String, serde_json::Value> = tokens
        .iter()
        .enumerate()
        .map(|(id, token)| (token.to_string(), serde_json::json!(id)))
        .collect();
    let json = serde_json::json!({
        "version": "2.0",
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "padding": null,
        "model": {"type": "WordLevel", "unk_token": "<unk>", "vocab": vocab},
        "decoder": {
            "type": "CTC",
            "pad_token": "<pad>",
            "word_delimiter_token": delimiter,
            "cleanup": cleanup
        }
    });
    from_json(&json.to_string()).unwrap()
}

#[test]
fn ctc_word_boundaries_do_not_depend_on_cleanup() {
    for delimiter in ["|", "<space>", "▁"] {
        for cleanup in [false, true] {
            let tokenizer = ctc_tokenizer(cleanup, delimiter);
            // Frame predictions: a a blank a delimiter delimiter b b blank b.
            assert_eq!(
                tokenizer.decode(&[2, 2, 1, 2, 4, 4, 3, 3, 1, 3], false).unwrap(),
                "aa bb"
            );
        }
    }
}

#[test]
fn ctc_cleanup_still_controls_punctuation() {
    let ids = [2, 5, 4, 3];
    assert_eq!(ctc_tokenizer(false, "|").decode(&ids, false).unwrap(), "a . b");
    assert_eq!(ctc_tokenizer(true, "|").decode(&ids, false).unwrap(), "a. b");
}

#[test]
fn ctc_unicode_words_keep_their_boundary_without_cleanup() {
    assert_eq!(
        ctc_tokenizer(false, "▁").decode(&[6, 6, 4, 7, 7], false).unwrap(),
        "你 好"
    );
}

#[test]
fn ctc_blank_and_repetition_controls_are_unchanged() {
    for cleanup in [false, true] {
        let tokenizer = ctc_tokenizer(cleanup, "|");
        assert_eq!(tokenizer.decode(&[2, 2, 1, 2, 3, 3], false).unwrap(), "aab");
        assert_eq!(tokenizer.decode(&[1, 1], false).unwrap(), "");
        assert_eq!(tokenizer.decode(&[], false).unwrap(), "");
    }
}
