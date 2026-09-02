//! `encode_batch_flat` must agree with `encode`, document for document.
//!
//! The flat path skips the planner, the completion queue and the per-document `Encoding`, so it is
//! a second implementation of the same contract rather than a wrapper over the first. This is what
//! keeps the two from drifting.
//!
//! Run `make data/gpt2.json` first -- the fixture is fetched, not committed.

use tk_encode::pipeline::PipelineTokenizer;

fn corpus() -> Vec<String> {
    let mut v = Vec::new();
    for i in 0..2000 {
        v.push(format!("the quick brown fox {i} jumps over the lazy dog"));
        v.push(format!("  leading and trailing   {i}  "));
        v.push(format!("语言模型 {i} mixed with ASCII and ελληνικά"));
        v.push(String::new());
        v.push(format!("<|endoftext|> {i} in the middle <|endoftext|>"));
    }
    v
}

fn check(tok: &PipelineTokenizer, add_special: bool) {
    let owned = corpus();
    let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
    let flat = tok.encode_batch_flat(&refs, add_special).unwrap();
    let one_by_one = tok.encode(owned.clone(), add_special).wait().unwrap();

    assert_eq!(flat.rows(), one_by_one.len(), "row count");
    for (i, enc) in one_by_one.iter().enumerate() {
        let want: Vec<u32> = enc.ids().iter().map(|t| t.id()).collect();
        let got: Vec<u32> = flat.row(i).unwrap().iter().map(|t| t.id()).collect();
        assert_eq!(want, got, "row {i} differs for {:?}", owned[i]);
    }
}

#[test]
fn flat_matches_encode() {
    let canonical = tk_convert::canonicalize_file("../data/gpt2.json").unwrap();
    let tok: PipelineTokenizer = tk_serialize::from_json(&canonical).unwrap();
    check(&tok, false);
    check(&tok, true);
}
