//! Render a minimal chat, preserving trusted ranges separately from user text.
//! cargo run --example safe_ranges_encode -- tokenizer.json USER_TEXT
use std::env;
use tokenizers::{canonicalize_file, from_json, pipeline::SafeRangeInput};

fn main() -> tokenizers::Result<()> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 3 {
        return Err("usage: safe_ranges_encode TOKENIZER_JSON USER_TEXT".into());
    }
    let tokenizer = from_json(&canonicalize_file(&args[1])?)?;
    // GPT-2's registered control marker. Real renderers choose the model's template markers.
    let control = "<|endoftext|>";
    if !tokenizer.get_added_vocabulary().is_special_token(control) {
        return Err("this example requires a registered <|endoftext|> special token".into());
    }
    let mut rendered = String::from(control);
    let prefix = 0..rendered.len();
    rendered.push_str(&args[2]);
    let suffix_start = rendered.len();
    rendered.push_str(control);
    let suffix = suffix_start..rendered.len();
    let input = SafeRangeInput::new(rendered, [prefix, suffix])?;
    // The renderer already supplied its control markers, so do not add a second template.
    let encodings = tokenizer.encode(input, false).wait()?;
    let ids: Vec<_> = encodings[0].ids().iter().map(|token| token.id()).collect();
    println!("{}", serde_json::to_string(&ids)?);
    Ok(())
}
