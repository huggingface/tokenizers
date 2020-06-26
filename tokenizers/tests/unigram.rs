#[cfg(not(debug_assertions))]
use assert_approx_eq::assert_approx_eq;
#[cfg(not(debug_assertions))]
use std::collections::HashMap;
#[cfg(not(debug_assertions))]
use std::fs::read_to_string;
use std::path::Path;
use tokenizers::models::unigram::model::Unigram;
#[cfg(not(debug_assertions))]
use tokenizers::models::unigram::{lattice::Lattice, trainer::UnigramTrainerBuilder};
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::tokenizer::{Model, NormalizedString, PreTokenizer};

#[test]
fn test_unigram_from_file() {
    let model = Unigram::load(Path::new("data/unigram.json")).unwrap();
    let pretok = Whitespace;
    let string = "吾輩《わがはい》は猫である。名前はまだ無い。";
    let input = pretok
        .pre_tokenize(&mut NormalizedString::from(string))
        .unwrap();
    assert_eq!(
        model
            .tokenize(input)
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

    // Check it works with spm_export_vocab model.
    let model = Unigram::load_spm(Path::new("data/unigram.model")).unwrap();
    let input = pretok
        .pre_tokenize(&mut NormalizedString::from(string))
        .unwrap();
    assert_eq!(
        model
            .tokenize(input)
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

        let n_trials = 100_000;
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

#[cfg(not(debug_assertions))]
#[test]
fn test_train_from_file() {
    let trainer = UnigramTrainerBuilder::default()
        .with_progress(false)
        .build();
    let mut word_counts: HashMap<String, u32> = HashMap::new();
    let file = read_to_string("data/unigram_wagahaiwa_nekodearu.txt").unwrap();
    let mut ignored = 0;
    for line in file.split('\n') {
        if line.len() > 2048 {
            ignored += 1;
            continue;
        }
        word_counts.insert(line.to_string(), 1);
    }
    println!("Kept {:?} sentences", word_counts.len());
    println!("Ignored {:?} sentences", ignored);

    // println!("Start train {:?}", word_counts);
    let (model, _) = trainer._train(word_counts).unwrap();
    // println!("Stop train {:?}", model.get_vocab());
    // println!("Vocab {}", model.get_vocab().len());

    let pretok = Whitespace;
    let input = pretok
        .pre_tokenize(&mut NormalizedString::from(
            "吾輩《わがはい》は猫である。名前はまだ無い。",
        ))
        .unwrap();
    // XXX: In original algo the tokens "吾", "輩" are actually
    // fused into "吾輩", it's *probably* because of is_valid_sentencepiece
    // shenanigans. Same for "名前"
    assert_eq!(
        model
            .tokenize(input)
            .unwrap()
            .iter()
            .map(|tok| tok.value.clone())
            .collect::<Vec<_>>(),
        vec![
            "吾",
            "輩",
            "《",
            "わが",
            "はい",
            "》",
            "は",
            "猫",
            "である",
            "。",
            "名",
            "前",
            "はまだ",
            "無い",
            "。"
        ]
    );
}
