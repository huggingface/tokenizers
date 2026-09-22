//! Uses the same downloaded GPT-2 fixture as fold_parity.rs.
use tk_encode::pipeline::EncodeSegment;

fn tokenizer() -> tk_encode::pipeline::PipelineTokenizer {
    let json = tk_convert::canonicalize_file("../data/gpt2.json").unwrap();
    tk_serialize::from_json(&json).unwrap()
}

#[test]
fn structured_gpt2_preserves_literal_text_and_only_emits_the_explicit_special() {
    let tk = tokenizer();
    let text = "你好 café 👋 <|endoftext|>\n";
    let segments = [
        EncodeSegment::text(text),
        EncodeSegment::special("<|endoftext|>"),
    ];
    let encoded = tk.encode_segments(&segments, false).wait().unwrap();
    let ids: Vec<_> = encoded[0].ids().iter().map(|id| id.id()).collect();
    assert_eq!(ids.last(), Some(&50256));
    assert_eq!(ids.iter().filter(|&&id| id == 50256).count(), 1);
    assert_eq!(
        tk.decode(&ids, false).unwrap(),
        format!("{text}<|endoftext|>")
    );
    assert_eq!(tk.decode(&ids, true).unwrap(), text);
    // Legacy encoding still recognizes the literal spelling after the structured call.
    let legacy = tk.encode("<|endoftext|>", false).wait().unwrap();
    assert_eq!(legacy[0].ids(), &[50256]);
}

#[test]
fn structured_gpt2_batch_matches_single_calls_across_large_inputs_and_clone_reuse() {
    let tk = tokenizer();
    let text = "hello 世界 <|endoftext|> ".repeat(1000);
    let batch = vec![
        vec![
            EncodeSegment::text(&text),
            EncodeSegment::special("<|endoftext|>"),
        ],
        vec![],
        vec![EncodeSegment::text("hel"), EncodeSegment::text("lo")],
    ];
    let encodings = tk.encode_segments_batch(&batch, false).wait().unwrap();
    for (segments, encoding) in batch.iter().zip(encodings) {
        assert_eq!(
            tk.clone().encode_segments(segments, false).wait().unwrap()[0],
            encoding
        );
    }
}
