#[macro_use]
extern crate criterion;

use criterion::{black_box, Criterion};
use tokenizers::{FromPretrainedParameters, Tokenizer};

static BATCH_SIZE: usize = 256;
static SEQ_LEN: usize = 1024;

fn bench_decode_starcoder(c: &mut Criterion) {
    let tokenizer = Tokenizer::from_pretrained(
        "bigcode/starcoder",
        Some(FromPretrainedParameters {
            auth_token: Some("hf_app_AKMictvLliBCQBKbznrqyXnBJUCnEfKQ".to_string()),
            ..Default::default()
        }),
    )
    .unwrap();

    c.bench_function(&format!("decoding {BATCH_SIZE}x{SEQ_LEN}"), |b| {
        b.iter(|| {
            (0..BATCH_SIZE).for_each(|_| {
                let input_ids = vec![423; SEQ_LEN];
                black_box(tokenizer.decode(&input_ids, false).unwrap());
            })
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(20);
    targets = bench_decode_starcoder
}
criterion_main!(benches);
