#[cfg(not(debug_assertions))]
use assert_approx_eq::assert_approx_eq;
#[cfg(not(debug_assertions))]
use std::collections::HashMap;
#[cfg(not(debug_assertions))]
use std::fs::read_to_string;
use std::path::Path;
#[cfg(not(debug_assertions))]
use tokenizers::models::unigram::Lattice;
use tokenizers::models::unigram::Unigram;
use tokenizers::tokenizer::Model;

#[test]
fn test_unigram_from_file() {
    let model = Unigram::load(Path::new("data/unigram.json")).unwrap();
    let string = "吾輩《わがはい》は猫である。名前はまだ無い。";
    assert_eq!(
        model
            .tokenize(string)
            .unwrap()
            .iter()
            .map(|tok| tok.value.clone())
            .collect::<Vec<_>>(),
        vec![
            "吾輩",
            "《",
            "わが",
            "はい",
            "》",
            "は",
            "猫",
            "である",
            "。",
            "名前",
            "はまだ",
            "無い",
            "。"
        ]
    );
}

#[cfg(not(debug_assertions))]
#[test]
fn test_sample() {
    let mut lattice = Lattice::from("ABC", 0, 1, 2);
    lattice.insert(0, 1, 1.0, 3); // A
    lattice.insert(1, 1, 1.2, 4); // B
    lattice.insert(2, 1, 1.5, 5); // C
    lattice.insert(0, 2, 1.6, 6); // AB
    lattice.insert(1, 2, 1.7, 7); // BC
    lattice.insert(0, 3, 1.8, 8); // ABC

    let thetas: Vec<f64> = vec![0.0, 0.01, 0.5, 0.7, 1.0];

    for theta in thetas {
        let mut probs: HashMap<String, f64> = HashMap::new();
        probs.insert("A B C".to_string(), (theta * (1.0 + 1.2 + 1.5)).exp());
        probs.insert("AB C".to_string(), (theta * (1.6 + 1.5)).exp());
        probs.insert("A BC".to_string(), (theta * (1.0 + 1.7)).exp());
        probs.insert("ABC".to_string(), (theta * (1.8)).exp());

        // Computes expected probabilities.
        let mut z = 0.0;

        for (_, p) in probs.iter() {
            z += p;
        }
        for (_, p) in probs.iter_mut() {
            *p /= z;
        }

        let n_trials = 1_000;
        let mut freq: HashMap<String, u32> = HashMap::new();
        for _ in 0..n_trials {
            let string = lattice.sample_token(theta).join(" ");
            *freq.entry(string).or_insert(0) += 1;
        }

        assert_eq!(freq.len(), probs.len());
        for (s, p) in probs.iter() {
            assert_approx_eq!(1.0 * (freq[s] as f64) / (n_trials as f64), p, 0.03)
        }
    }
}
