//! Uses the same downloaded GPT-2 fixture as fold_parity.rs.
// These are lists of policy ranges, never attempts to collect a range's elements.
#![allow(clippy::single_range_in_vec_init)]

use tk_encode::pipeline::SafeRangeInput;

fn tokenizer() -> tk_encode::pipeline::PipelineTokenizer {
    let json = tk_convert::canonicalize_file("../data/gpt2.json").unwrap();
    tk_serialize::from_json(&json).unwrap()
}

#[test]
fn safe_ranges_gpt2_preserves_literal_unicode_and_only_recognizes_trusted_specials() {
    let tk = tokenizer();
    let user_text = "你好 café 👋 <|endoftext|>\n\0<script>\0";
    let rendered = format!("{user_text}<|endoftext|>");
    let safe = user_text.len()..rendered.len();
    let encoded = tk
        .encode(SafeRangeInput::new(&rendered, [safe]).unwrap(), false)
        .wait()
        .unwrap();
    let ids: Vec<_> = encoded[0].ids().iter().map(|id| id.id()).collect();
    assert_eq!(ids.last(), Some(&50256));
    assert_eq!(ids.iter().filter(|&&id| id == 50256).count(), 1);
    assert_eq!(tk.decode(&ids, false).unwrap(), rendered);
    assert_eq!(tk.decode(&ids, true).unwrap(), user_text);
    assert_eq!(
        tk.encode("<|endoftext|>", false).wait().unwrap()[0].ids(),
        &[50256]
    );
}

#[test]
fn safe_ranges_gpt2_cannot_join_partial_control_markers_across_a_boundary() {
    let tk = tokenizer();
    let text = "<|endoftext|>";
    for split in 1..text.len() {
        for range in [0..split, split..text.len()] {
            let encoded = tk
                .encode(SafeRangeInput::new(text, [range]).unwrap(), false)
                .wait()
                .unwrap();
            let ids: Vec<_> = encoded[0].ids().iter().map(|id| id.id()).collect();
            assert!(!ids.contains(&50256));
            assert_eq!(tk.decode(&ids, false).unwrap(), text);
        }
    }
}

#[test]
fn safe_ranges_gpt2_large_batches_match_single_calls_and_clone_reuse() {
    let tk = tokenizer();
    let text = "hello 世界 <|endoftext|> ".repeat(1000);
    let rendered = format!("{text}<|endoftext|>");
    let batch = vec![
        SafeRangeInput::new(&rendered, [text.len()..rendered.len()]).unwrap(),
        SafeRangeInput::new(&text, []).unwrap(),
        SafeRangeInput::new("", []).unwrap(),
    ];
    let encodings = tk.encode(batch.clone(), false).wait().unwrap();
    for (input, encoding) in batch.into_iter().zip(&encodings) {
        assert_eq!(
            &tk.clone().encode(input, false).wait().unwrap()[0],
            encoding
        );
    }
    assert_eq!(
        encodings[0].ids().iter().filter(|id| **id == 50256).count(),
        1
    );
    assert!(!encodings[1].ids().iter().any(|id| *id == 50256));
}

#[test]
#[cfg(feature = "serialize")]
fn safe_ranges_json_config_roundtrip_does_not_serialize_input_policy() {
    let config = r#"{
        "version":"2.0", "model":{"type":"BPE", "byte_level":false,
        "vocab":{"a":0,"b":1,"ab":2},"merges":[["a","b"]]},
        "added_tokens":[{"id":3,"content":"<s>","special":true,"normalized":false,
        "single_word":false,"lstrip":false,"rstrip":false}],
        "normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null
    }"#;
    let tk = tk_serialize::from_json(config).unwrap();
    let before = tk_serialize::to_json(&tk).unwrap();
    let input = SafeRangeInput::new("ab<s><s>", [5..8]).unwrap();
    assert_eq!(tk.encode(input, false).wait().unwrap()[0].ids(), &[2, 3]);
    assert_eq!(tk_serialize::to_json(&tk).unwrap(), before);
    let reloaded = tk_serialize::from_json(&before).unwrap();
    assert_eq!(
        reloaded.encode("ab<s><s>", false).wait().unwrap()[0].ids(),
        &[2, 3, 3]
    );
}
