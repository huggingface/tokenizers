//! Byte-exactness gate for the slim JSON read path.
//!
//! Every model in `examples/bench_models.json` (10) against every fixture the `Makefile` defines
//! (9 languages + 6 modalities = 15), for both `add_special_tokens` settings: **300 id-stream
//! comparisons**.
//!
//! Two modes, because the slim reader is being built against this gate rather than after it:
//!
//! - `record` — encode through the config path (`Tokenizer::from_file` →
//!   `PipelineTokenizer::try_from`) and write a digest per (model, fixture, add_special) triple to
//!   `examples/json_oracle.digests.json` (`data/` is gitignored, and this baseline must be tracked:
//!   it is what proves moving the config layer to another crate changed no ids). Run it on a commit
//!   that predates the slim reader.
//! - `check` — recompute and compare against that file. Any drift fails, so this also catches the
//!   case where *both* paths change together (which an in-process A/B could not see).
//!
//! Once `PipelineTokenizer::from_json` exists, `check` additionally compares it against the config
//! path in the same process, which is the comparison that actually matters. Until then it reports
//! the slim column as `skipped` rather than silently passing.
//!
//! The digest is FNV-1a over the little-endian id bytes plus the id count: a mismatch in either
//! the ids or their number changes it, and it is stable across platforms (no hashing of pointers,
//! no `AHash` randomisation).
//!
//! ```text
//! cargo run --release -p tk-convert --features fancy-regex --example json_oracle -- record
//! cargo run --release -p tk-convert --features fancy-regex --example json_oracle -- check
//! ```
//!
//! `fancy-regex` is needed to *record*, because the config path compiles a real regex for any
//! pattern `gpt_fsm` does not recognise. The slim path must not need it — that is part of the point
//! — so `check` reports which features it ran under.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::time::Instant;

use tk_convert::{AddedToken, Tokenizer};
use tk_encode::pipeline::PipelineTokenizer;

const DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../data");
const MANIFEST: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/bench_models.json");
const DIGESTS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/examples/json_oracle.digests.json"
);

/// Keep in sync with `Makefile`'s `FIXTURE_LANGS`.
const FIXTURE_LANGS: &[&str] = &[
    "amh_Ethi", "arb_Arab", "cmn_Hani", "eng_Latn", "hin_Deva", "jpn_Jpan", "rus_Cyrl", "tam_Taml",
    "tha_Thai",
];
/// Keep in sync with `Makefile`'s `FIXTURE_MODALITIES`.
const FIXTURE_MODALITIES: &[&str] = &[
    "agentic_swe",
    "agentic-traces",
    "code_mixed",
    "math_latex",
    "added_special_sparse",
    "added_normalized_sparse",
];

// The same added tokens `fixture_bench` injects, so the `added_*` fixtures actually exercise the
// added-token split whichever model is loaded. `ADDED_SPECIAL` is matched on the raw pass,
// `ADDED_NORMALIZED` on the normalized one.
const ADDED_SPECIAL: &[&str] = &["<|xs0|>", "<|xs1|>", "<|xs2|>", "<|xs3|>", "<|xs4|>"];
const ADDED_NORMALIZED: &[&str] = &[
    "widgetron",
    "flibberjast",
    "zorptastic",
    "quibblenaut",
    "snorlaxian",
];

/// FNV-1a over the ids' LE bytes, salted with the count so a truncated stream cannot collide.
fn digest(ids: &[u32]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |b: u8| {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    for byte in (ids.len() as u64).to_le_bytes() {
        mix(byte);
    }
    for id in ids {
        for byte in id.to_le_bytes() {
            mix(byte);
        }
    }
    format!("{h:016x}")
}

fn fixtures() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for (group, names) in [("lang", FIXTURE_LANGS), ("modalities", FIXTURE_MODALITIES)] {
        for name in names {
            let path = format!("{DATA_DIR}/fixtures/{group}/{name}.txt");
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("missing fixture {path}: {e}\nrun `make fixtures`"));
            out.push(((*name).to_string(), text));
        }
    }
    out
}

fn models() -> Vec<(String, String)> {
    let manifest = std::fs::read_to_string(MANIFEST).expect("manifest");
    // Two fields, fixed shape, and this example must build without serde_json once the slim path is
    // the default — so scan for them rather than pulling in a JSON parser.
    let mut out = Vec::new();
    for entry in manifest.split('{').skip(1) {
        let field = |key: &str| -> Option<String> {
            let at = entry.find(&format!("\"{key}\""))? + key.len() + 2;
            let rest = &entry[at..];
            let open = rest.find('"')? + 1;
            let close = rest[open..].find('"')? + open;
            Some(rest[open..close].to_string())
        };
        if let (Some(name), Some(file)) = (field("name"), field("file")) {
            out.push((name, file));
        }
    }
    assert!(!out.is_empty(), "parsed no models out of {MANIFEST}");
    out
}

/// Load through the config path *without* injecting anything: the slim reader only sees the file,
/// so this is the fair comparand for the A/B. The injected variant below is what the recorded
/// digests pin, so the `added_*` fixtures still exercise the added-token split.
fn load_raw(file: &str) -> Result<Tokenizer, String> {
    let path = format!("{DATA_DIR}/{file}");
    Tokenizer::from_file(&path).map_err(|e| format!("from_file: {e}"))
}

/// Load through the config path and inject the shared added tokens.
fn load(file: &str) -> Result<Tokenizer, String> {
    let path = format!("{DATA_DIR}/{file}");
    let mut tok = Tokenizer::from_file(&path).map_err(|e| format!("from_file: {e}"))?;
    tok.add_special_tokens(ADDED_SPECIAL.iter().map(|s| AddedToken::from(*s, true)))
        .map_err(|e| format!("add_special_tokens: {e}"))?;
    tok.add_tokens(
        ADDED_NORMALIZED
            .iter()
            .map(|s| AddedToken::from(*s, false).normalized(true)),
    )
    .map_err(|e| format!("add_tokens: {e}"))?;
    Ok(tok)
}

fn ids(pipeline: &PipelineTokenizer, text: &str, add_special_tokens: bool) -> Vec<u32> {
    pipeline
        .encode(text, add_special_tokens)
        .wait()
        .expect("encode")
        .iter()
        .flat_map(|e| e.ids())
        .map(|t| t.id())
        .collect()
}

/// The slim path: `tk-convert`'s canonicalizer, then the serde-free reader. No injected tokens.
///
/// Routed *through* `canonicalize_file` rather than reading the fixture directly, because that is
/// the shipping arrangement: `tk-convert` owns every legacy shape and hands the reader a canonical
/// file. Reading the raw fixture here would instead test `tk-serialize`'s own backwards
/// compatibility, which is the thing being moved out — and it would leave the converter with no
/// gate on it at all. This way all 300 comparisons exercise the converter, and every legacy shape
/// in `data/` (7 configs with an untyped model, 7 with space-joined merges, 4 array-shaped Unigram
/// vocabs, 4 legacy Metaspace spellings) is one it has to get right.
///
/// `None` when either half refuses the config — the reader does not cover every model yet, and a
/// refusal is reported as `unsupported` rather than counted as a pass or a failure.
fn slim_pipeline(file: &str) -> Result<PipelineTokenizer, String> {
    let canonical =
        tk_convert::canonicalize_file(format!("{DATA_DIR}/{file}")).map_err(|e| e.to_string())?;
    tk_serialize::from_json(&canonical).map_err(|e| e.to_string())
}

fn read_digests() -> BTreeMap<String, String> {
    let raw = match std::fs::read_to_string(DIGESTS) {
        Ok(raw) => raw,
        Err(e) => {
            eprintln!(
                "cannot read {DIGESTS}: {e}\nrun `... --example json_oracle -- record` first"
            );
            std::process::exit(2);
        }
    };
    let mut out = BTreeMap::new();
    for line in raw.lines() {
        let line = line.trim().trim_end_matches(',');
        if let Some((k, v)) = line.split_once("\": \"") {
            out.insert(
                k.trim_start_matches('"').to_string(),
                v.trim_end_matches('"').to_string(),
            );
        }
    }
    out
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "check".into());
    if mode != "record" && mode != "check" {
        eprintln!("usage: json_oracle [record|check]");
        std::process::exit(2);
    }

    let models = models();
    let fixtures = fixtures();
    let total = models.len() * fixtures.len() * 2;
    println!(
        "json_oracle {mode}: {} models x {} fixtures x 2 add_special = {total} comparisons",
        models.len(),
        fixtures.len()
    );
    println!(
        "features: fancy-regex={} parallelism={}",
        cfg!(feature = "fancy-regex"),
        cfg!(feature = "parallelism")
    );

    let baseline = if mode == "check" {
        read_digests()
    } else {
        BTreeMap::new()
    };

    let start = Instant::now();
    let mut recorded = BTreeMap::new();
    let (mut done, mut ok, mut failed, mut skipped_models) = (0usize, 0usize, 0usize, 0usize);
    let mut slim_checked = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for (name, file) in &models {
        let tok = match load(file) {
            Ok(tok) => tok,
            Err(e) => {
                println!("  SKIP {name}: {e}");
                skipped_models += 1;
                continue;
            }
        };
        let pipeline = match PipelineTokenizer::try_from(&tok) {
            Ok(p) => p,
            Err(e) => {
                println!("  SKIP {name}: pipeline build: {e}");
                skipped_models += 1;
                continue;
            }
        };

        // The A/B pair: the same file through the config path with nothing injected, and through the
        // slim reader. A reader refusal is reported, not counted as pass or fail.
        let raw = load_raw(file)
            .ok()
            .and_then(|t| PipelineTokenizer::try_from(&t).ok());
        let slim = match slim_pipeline(file) {
            Ok(s) => Some(s),
            Err(e) => {
                println!("  slim reader refuses {name}: {e}");
                None
            }
        };

        for (fixture, text) in &fixtures {
            for add_special in [false, true] {
                let key = format!("{name}|{fixture}|{add_special}");
                let got = ids(&pipeline, text, add_special);
                let d = digest(&got);
                done += 1;

                let mut verdict = String::new();
                let mut bad = false;

                if mode == "record" {
                    recorded.insert(key.clone(), d.clone());
                    verdict.push_str("recorded");
                } else {
                    match baseline.get(&key) {
                        Some(want) if *want == d => verdict.push_str("ok"),
                        Some(want) => {
                            bad = true;
                            let _ = write!(verdict, "DRIFT want={want} got={d}");
                        }
                        None => {
                            bad = true;
                            verdict.push_str("MISSING from digest file");
                        }
                    }
                    match (&slim, &raw) {
                        (Some(slim), Some(raw)) => {
                            slim_checked += 1;
                            let a = ids(slim, text, add_special);
                            let b = ids(raw, text, add_special);
                            if a == b {
                                verdict.push_str(" | slim ok");
                            } else {
                                bad = true;
                                let first = a.iter().zip(&b).position(|(x, y)| x != y).map_or_else(
                                    || "length only".to_string(),
                                    |i| format!("index {i}"),
                                );
                                let _ = write!(
                                    verdict,
                                    " | SLIM MISMATCH ({} vs {} ids, first diff at {first})",
                                    a.len(),
                                    b.len()
                                );
                            }
                        }
                        _ => verdict.push_str(" | slim unsupported"),
                    }
                }

                if bad {
                    failed += 1;
                    failures.push(format!("{key}: {verdict}"));
                } else {
                    ok += 1;
                }

                let elapsed = start.elapsed().as_secs_f64();
                let eta = if done > 0 {
                    elapsed / done as f64 * (total - done) as f64
                } else {
                    0.0
                };
                println!(
                    "[{done}/{total}] {name} x {fixture} special={add_special:<5} {:>9} ids  {verdict}  | elapsed {elapsed:.0}s | eta ~{eta:.0}s",
                    got.len()
                );
            }
        }
    }

    if mode == "record" {
        let mut out = String::from("{\n");
        let n = recorded.len();
        for (i, (k, v)) in recorded.iter().enumerate() {
            let comma = if i + 1 == n { "" } else { "," };
            let _ = writeln!(out, "  \"{k}\": \"{v}\"{comma}");
        }
        out.push_str("}\n");
        std::fs::write(DIGESTS, out).expect("write digests");
        println!("\nwrote {n} digests to {DIGESTS}");
    }

    println!(
        "\n{ok}/{done} ok, {failed} failed, {skipped_models} model(s) skipped, {slim_checked} slim comparison(s) in {:.0}s",
        start.elapsed().as_secs_f64()
    );
    if !failures.is_empty() {
        println!("\nfailures:");
        for f in &failures {
            println!("  {f}");
        }
    }
    if failed > 0 {
        std::process::exit(1);
    }
    if mode == "check" && skipped_models > 0 {
        // A model that stops loading is a regression, not a pass.
        println!("FAIL: {skipped_models} model(s) skipped during check");
        std::process::exit(1);
    }
}
