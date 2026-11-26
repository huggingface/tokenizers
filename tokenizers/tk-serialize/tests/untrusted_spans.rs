//! The bytes a caller wrote must not become an added token.
//!
//! A chat template renders a caller's text into the same string as its own markers, so a user who
//! writes `<|endoftext|>` is offered to the added-token matcher as one. Masking the bytes they
//! wrote has to leave every other id alone: same batch, same text, same everything apart from
//! that one marker.

use tk_encode::pipeline::EncodeOptions;
use tk_encode::pipeline::TrustMasks;
use tk_encode::pipeline::Tokenizer;

/// `data/gpt2.json` is still version `1.0`, so run the upgrade pass first -- the reader only
/// accepts canonical `2.0`. Same as `benches/encode.rs`.
fn load() -> Tokenizer {
    let canonical = tk_convert::canonicalize_file("../data/gpt2.json").unwrap();
    tk_serialize::from_json(&canonical).unwrap()
}

const EOT: u32 = 50256;
const MARKER: &str = "<|endoftext|>";

/// The whole caller side of the contract: `push` one mask per sequence, in order, and reuse the
/// buffers. Everything else -- which bytes those are -- comes from whoever assembled the text.
fn masks(texts: &[&str], untrusted: &[&[std::ops::Range<usize>]]) -> TrustMasks {
    let mut masks = TrustMasks::default();
    for (text, spans) in texts.iter().zip(untrusted) {
        masks.push(text, spans);
    }
    masks
}

fn ids(tokenizer: &Tokenizer, text: &str, spans: Option<&[std::ops::Range<usize>]>) -> Vec<u32> {
    let handle = match spans {
        Some(spans) => tokenizer.encode_with_trust_mask(
            text,
            &masks(&[text], &[spans]),
            &EncodeOptions::default(),
        ),
        None => tokenizer.encode(text, &EncodeOptions::default()),
    };
    handle
        .wait()
        .unwrap()
        .into_iter()
        .flat_map(|encoding| encoding.ids().iter().map(|token| token.id()).collect::<Vec<u32>>())
        .collect()
}

fn ids_batch(
    tokenizer: &Tokenizer,
    texts: &[&str],
    spans: &[&[std::ops::Range<usize>]],
) -> Vec<Vec<u32>> {
    let inputs: Vec<String> = texts.iter().map(|text| text.to_string()).collect();
    tokenizer
        .encode_with_trust_mask(inputs, &masks(texts, spans), &EncodeOptions::default())
        .wait()
        .unwrap()
        .into_iter()
        .map(|encoding| encoding.ids().iter().map(|token| token.id()).collect::<Vec<u32>>())
        .collect()
}

#[test]
fn a_marker_in_masked_text_is_not_a_token() {
    let tokenizer = load();
    let text = "a<|endoftext|>b";
    let start = text.find(MARKER).unwrap();

    assert!(ids(&tokenizer, text, None).contains(&EOT), "the fixture has an added token here");
    let masked = ids(&tokenizer, text, Some(&[start..start + MARKER.len()]));
    assert!(!masked.contains(&EOT), "\"a{MARKER}b\" written by the caller must stay text");
    // Not a lossy rewrite: the bytes are still there, they just tokenize as text.
    assert_eq!(tokenizer.decode(&masked, false).unwrap(), text);
}

#[test]
fn masking_part_of_the_marker_still_refuses_it() {
    let tokenizer = load();
    let text = "a<|endoftext|>b";
    let start = text.find(MARKER).unwrap();
    // Two bytes of the marker. A match over any 0 byte is refused whole, because half a token is
    // not a token.
    let masked = ids(&tokenizer, text, Some(&[start + 2..start + 4]));
    assert!(!masked.contains(&EOT));
}

#[test]
fn text_without_added_tokens_encodes_byte_identically() {
    let tokenizer = load();
    let text = "the quick brown fox jumps over the lazy dog";
    let plain = ids(&tokenizer, text, None);
    assert_eq!(ids(&tokenizer, text, Some(&[0..text.len()])), plain);
}

#[test]
fn each_sequence_carries_its_own_mask() {
    let tokenizer = load();
    let a = "x<|endoftext|>y";
    let b = "p<|endoftext|>q";
    let start = a.find(MARKER).unwrap();

    let out = ids_batch(&tokenizer, &[a, b], &[&[start..start + MARKER.len()], &[]]);
    assert!(!out[0].contains(&EOT), "the first sequence is masked");
    assert!(out[1].contains(&EOT), "the second has no mask and finds its marker");

    // A batch nobody masks is the plain encode, id for id.
    let plain = ids_batch(&tokenizer, &[a, b], &[&[], &[]]);
    assert_eq!(plain, [ids(&tokenizer, a, None), ids(&tokenizer, b, None)]);
}

#[test]
fn a_long_input_goes_through_the_parallel_planner_and_still_refuses_the_marker() {
    let tokenizer = load();
    // Past `PARALLEL_MIN_BYTES`, so the sequence is chunked across workers. That is the path that
    // would silently lose the mask if it were not sliced along with the chunk's range.
    let text = format!(
        "{} <|endoftext|> {}",
        "lorem ipsum ".repeat(2_000),
        "dolor sit ".repeat(2_000)
    );
    let start = text.find(MARKER).unwrap();
    assert!(text.len() > 8 * 1024);

    assert!(ids(&tokenizer, &text, None).contains(&EOT));
    let masked = ids(&tokenizer, &text, Some(&[start..start + MARKER.len()]));
    assert!(!masked.contains(&EOT));
}

#[test]
fn a_long_batch_goes_through_the_parallel_planner_with_and_without_masks() {
    let tokenizer = load();
    let long = format!("{} <|endoftext|> {}", "lorem ipsum ".repeat(1_000), "dolor sit ".repeat(1_000));
    let texts = [long.as_str(), long.as_str()];
    let start = long.find(MARKER).unwrap();

    // A batch with no masks at all is the plain encode, on the parallel path.
    let plain = ids_batch(&tokenizer, &texts, &[&[], &[]]);
    assert!(plain[0].contains(&EOT) && plain[1].contains(&EOT));

    // One masked, one not, still on the parallel path.
    let out = ids_batch(&tokenizer, &texts, &[&[start..start + MARKER.len()], &[]]);
    assert!(!out[0].contains(&EOT), "the first sequence is masked");
    assert!(out[1].contains(&EOT), "the second has no mask");
}

#[test]
fn a_mask_that_does_not_line_up_is_rejected() {
    let tokenizer = load();
    let text = "aé<|endoftext|>";

    // One mask per input, or the masks have no owner.
    let two: Vec<String> = vec![text.to_string(), text.to_string()];
    assert!(
        tokenizer
            .encode_with_trust_mask(two, &masks(&[text], &[&[]]), &EncodeOptions::default())
            .wait()
            .is_err()
    );

    // One byte per text byte. A mask is a mask, not a list of ranges: it cannot be out of order,
    // overlap, or split a character, because none of that can be expressed.
    let mut wrong = TrustMasks::default();
    wrong.push(&text[..text.len() - 1], &[]);
    assert!(
        tokenizer
            .encode_with_trust_mask(text, &wrong, &EncodeOptions::default())
            .wait()
            .is_err()
    );

    // A pair is two sequences, and one mask cannot say which.
    assert!(
        tokenizer
            .encode_with_trust_mask(
                (text.to_string(), text.to_string()),
                &masks(&[text], &[&[]]),
                &EncodeOptions::default()
            )
            .wait()
            .is_err()
    );
}
