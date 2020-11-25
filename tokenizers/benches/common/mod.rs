use std::time::{Duration, Instant};

use criterion::black_box;

use tokenizers::{
    Decoder, EncodeInput, Model, Normalizer, PostProcessor, PreTokenizer, TokenizerImpl, Trainer,
};

pub fn iter_bench_encode<M, N, PT, PP, D>(
    iters: u64,
    tokenizer: &TokenizerImpl<M, N, PT, PP, D>,
    lines: &[EncodeInput],
) -> Duration
where
    M: Model,
    N: Normalizer,
    PT: PreTokenizer,
    PP: PostProcessor,
    D: Decoder,
{
    let mut duration = Duration::new(0, 0);
    let mut line_index: usize = 0;
    for _i in 0..iters {
        if line_index >= lines.len() {
            line_index = 0;
        }
        let input = lines[line_index].clone();
        let start = Instant::now();
        let _ = black_box(tokenizer.encode(input, false));
        duration = duration.checked_add(start.elapsed()).unwrap();
    }
    duration
}

pub fn iter_bench_encode_batch<M, N, PT, PP, D>(
    iters: u64,
    tokenizer: &TokenizerImpl<M, N, PT, PP, D>,
    batches: &[Vec<EncodeInput>],
) -> Duration
where
    M: Model + Send + Sync,
    N: Normalizer + Send + Sync,
    PT: PreTokenizer + Send + Sync,
    PP: PostProcessor + Send + Sync,
    D: Decoder + Send + Sync,
{
    let mut duration = Duration::new(0, 0);
    let mut batch_index: usize = 0;
    for _i in 0..iters {
        if batch_index >= batches.len() {
            batch_index = 0;
        }
        let batch = batches[batch_index].clone();
        let start = Instant::now();
        let _ = black_box(tokenizer.encode_batch(batch, false));
        duration = duration.checked_add(start.elapsed()).unwrap();
    }
    duration
}

pub fn iter_bench_train<T, M, N, PT, PP, D>(
    iters: u64,
    tokenizer: &mut TokenizerImpl<M, N, PT, PP, D>,
    trainer: &mut T,
    files: Vec<String>,
) -> Duration
where
    T: Trainer<Model = M> + Sync,
    M: Model + Send + Sync,
    N: Normalizer + Send + Sync,
    PT: PreTokenizer + Send + Sync,
    PP: PostProcessor + Send + Sync,
    D: Decoder + Send + Sync,
{
    let mut duration = Duration::new(0, 0);
    for _i in 0..iters {
        let start = Instant::now();
        tokenizer.train_from_files(trainer, files.clone()).unwrap();
        duration = duration.checked_add(start.elapsed()).unwrap();
    }
    duration
}
