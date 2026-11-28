use std::error::Error;
use std::fs::File;
use std::io::Write;
use tk_encode::models::unigram::Unigram;
use tk_train::{Trainer, UnigramTrainer};

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    // Arguments: sentence_count output_path
    let sentence_count: usize = std::env::args()
        .nth(1)
        .map(|v| v.parse().unwrap_or(5_000))
        .unwrap_or(5_000);
    let output_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "unigram_vocab.json".to_string());

    let templates = [
        "the quick brown fox jumps over the lazy dog",
        "a tokenizer converts text into useful pieces",
        "reusable buffers reduce repeated allocation costs",
        "language models learn common patterns from data",
        "unicode text includes characters beyond ascii",
    ];
    let corpus: Vec<String> = (0..sentence_count)
        .map(|index| {
            format!(
                "{} sample{:08} {}",
                templates[index % templates.len()],
                index,
                templates[(index / templates.len()) % templates.len()]
            )
        })
        .collect();

    let mut trainer = UnigramTrainer::builder()
        .show_progress(false)
        .vocab_size(256)
        .n_sub_iterations(2)
        .seed_size(50_000)
        .build()?;
    trainer.feed(corpus.iter().map(String::as_str), |sentence| {
        Ok(vec![sentence.to_owned()])
    })?;

    let mut model = Unigram::default();
    trainer.train(&mut model)?;

    // Dump vocab as JSON (matching tokenizer.json Unigram model format)
    let mut file = File::create(&output_path)?;
    writeln!(file, "{{")?;
    writeln!(file, "  \"type\": \"Unigram\",")?;
    match model.unk_id() {
        Some(id) => writeln!(file, "  \"unk_id\": {},", id)?,
        None => writeln!(file, "  \"unk_id\": null,")?,
    }
    writeln!(file, "  \"vocab\": [")?;
    let vocab: Vec<_> = model.vocab().to_vec();
    for (i, (token, score)) in vocab.iter().enumerate() {
        let comma = if i + 1 < vocab.len() { "," } else { "" };
        // JSON-escape the token string
        let escaped = token
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t");
        writeln!(file, "    [\"{}\", {}]{}", escaped, score, comma)?;
    }
    writeln!(file, "  ],")?;
    writeln!(file, "  \"byte_fallback\": {}", model.byte_fallback())?;
    writeln!(file, "}}")?;

    println!("Wrote {} tokens to {}", vocab.len(), output_path);
    Ok(())
}
