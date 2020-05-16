use serde_json;
use tokenizers::models::bpe::BPE;

#[test]
fn bpe_serde() {
    let bpe = BPE::from_files("data/gpt2-vocab.json", "data/gpt2-merges.txt")
        .build()
        .expect("Files not found, run `make test` to download these files");
    let ser = serde_json::to_string(&bpe).unwrap();
    let de: BPE = serde_json::from_str(&ser).unwrap();
    assert_eq!(bpe, de);
}
