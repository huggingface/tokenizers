use std::error::Error;
use std::time::{Duration, Instant};
use tk_encode::models::unigram::Unigram;
use tk_train::{Trainer, UnigramTrainer};

fn argument(index: usize, default: usize) -> Result<usize, Box<dyn Error + Send + Sync>> {
    std::env::args()
        .nth(index)
        .map(|value| value.parse::<usize>().map_err(Into::into))
        .unwrap_or(Ok(default))
}

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let sentence_count = argument(1, 5_000)?;
    let run_count = argument(2, 3)?;
    if sentence_count == 0 || run_count == 0 {
        return Err("sentence count and run count must be greater than zero".into());
    }

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

    let mut total = Duration::ZERO;
    for run in 1..=run_count {
        let mut model = Unigram::default();
        let start = Instant::now();
        trainer.train(&mut model)?;
        let elapsed = start.elapsed();
        total += elapsed;
        println!("run {run}: {:.3}s", elapsed.as_secs_f64());
    }

    let average = total / run_count as u32;
    println!(
        "{sentence_count} unique sentences; average training time: {:.3}s",
        average.as_secs_f64()
    );
    Ok(())
}
