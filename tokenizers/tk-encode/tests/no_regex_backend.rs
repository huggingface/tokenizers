//! Every model the CI benchmark reports on must load with no regex backend.
//!
//! `fancy-regex` is optional. Without it there is no engine for a user-supplied regex, so a config
//! needing one fails to load — and the benchmark renders that model as an error card instead of
//! benching it. The benchmark is built without the feature on purpose (see `bench-baseline` in
//! Cargo.toml), so that the throughput and memory numbers describe the configuration tk-encode
//! actually ships, the same one the binary-size numbers describe.
//!
//! This test walks the benchmark's own model list and fails if any entry stops loading, which is what
//! adding a model with, say, a `Replace` normalizer holding a real regex would do. It is the guard on
//! that build choice. Run in the default build to mean anything; with `fancy-regex` on it passes
//! trivially, since then every pattern has an engine.

use std::convert::TryFrom;
use std::path::Path;

use serde_json::Value;
use tk_encode::Tokenizer;
use tk_encode::pipeline::PipelineTokenizer;

const MANIFEST: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/bench_models.json");
const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../data");
const PROBE: &str = "The quick brown fox jumps 123 中文 don't!";

#[test]
fn every_benchmarked_model_loads_without_a_regex_backend() {
    let manifest: Vec<Value> =
        serde_json::from_str(&std::fs::read_to_string(MANIFEST).unwrap()).unwrap();
    assert!(!manifest.is_empty(), "empty model manifest");

    let (mut checked, mut failed) = (0, Vec::new());
    for entry in &manifest {
        let name = entry["name"].as_str().unwrap();
        let file = entry["file"]
            .as_str()
            .map_or_else(|| format!("{name}.json"), str::to_string);
        let path = Path::new(DATA).join(&file);
        if !path.exists() {
            eprintln!("skip {name}: {file} not fetched (`make bench-models`)");
            continue;
        }
        checked += 1;
        match Tokenizer::from_file(&path) {
            // A model the pipeline cannot build or encode yet is a separate, tracked gap (the
            // benchmark reports those as roadmap cards); only the *load* is pinned here.
            Ok(tok) => {
                if let Ok(pipeline) = PipelineTokenizer::try_from(&tok)
                    && let Ok(tokens) = pipeline.encode(PROBE, false)
                {
                    assert!(!tokens.is_empty(), "{name}: encoded to nothing");
                }
            }
            Err(e) => failed.push(format!("{name}: {e}")),
        }
    }
    assert!(
        checked > 0,
        "no model tokenizers present — run `make bench-models`"
    );
    assert!(
        failed.is_empty(),
        "{checked} models checked, these need a regex backend to load:\n  {}",
        failed.join("\n  ")
    );
}
