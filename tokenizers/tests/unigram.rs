#[cfg(not(debug_assertions))]
use assert_approx_eq::assert_approx_eq;
#[cfg(not(debug_assertions))]
use std::collections::HashMap;
#[cfg(not(debug_assertions))]
use std::fs::read_to_string;
use std::path::Path;
#[cfg(not(debug_assertions))]
use std::process::Command;
use tokenizers::models::unigram::Unigram;
#[cfg(not(debug_assertions))]
use tokenizers::models::unigram::{lattice::Lattice, trainer::UnigramTrainerBuilder};
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::tokenizer::{Model, NormalizedString, PreTokenizer};
#[cfg(not(debug_assertions))]
use unicode_normalization::UnicodeNormalization;

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
        .show_progress(false)
        .split_by_whitespace(true)
        .build()
        .unwrap();
    let mut word_counts: Vec<(String, u32)> = vec![];
    let file = read_to_string("data/unigram_wagahaiwa_nekodearu.txt").unwrap();
    let mut ignored = 0;
    for line in file.split("\n") {
        if line.len() > 4192 || line.len() == 0 {
            ignored += 1;
            continue;
        }
        word_counts.push((line.to_string(), 1));
    }
    println!("Kept {:?} sentences", word_counts.len());
    println!("Ignored {:?} sentences", ignored);

    // println!("Start train {:?}", word_counts);
    let (model, _) = trainer._train(word_counts).unwrap();
    // println!("Stop train {:?}", model.get_vocab());
    // println!("Vocab {}", model.get_vocab().len());
    //
    model
        .save(
            std::path::Path::new("data"),
            Some("unigram_wagahaiwa_nekodearu"),
        )
        .unwrap();

    let pretok = Whitespace;
    let input = pretok
        .pre_tokenize(&mut NormalizedString::from(
            "吾輩《わがはい》は猫である。名前はまだ無い。",
        ))
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
#[ignore]
fn test_spm_compat_train() {
    let n_sentences = 100_000;
    let train_file = "data/wikitext-103-raw/wiki.train.raw";
    let test_file = "data/wikitext-103-raw/wiki.test.raw";
    let output = Command::new("spm_train")
        .args(&[
            "--input",
            train_file,
            "--model_type",
            "unigram",
            "--model_prefix",
            "data/wikitext-103-raw/spm_wiki_103",
            "--input_sentence_size",
            &n_sentences.to_string(),
            "--num_threads",
            "1",
            "--shuffle_input_sentence",
            "0",
            "--character_coverage",
            "1",
        ])
        .output()
        .expect("Command failed is `spm_train` installed ?");
    if !output.status.success() {
        let err_msg = std::str::from_utf8(&output.stderr).unwrap();
        assert!(output.status.success(), "Command failed {}", err_msg)
    }
    println!("train: {}", std::str::from_utf8(&output.stderr).unwrap());

    let output = Command::new("spm_encode")
        .args(&[
            "--model",
            "data/wikitext-103-raw/spm_wiki_103.model",
            "--input",
            test_file,
        ])
        .output()
        .expect("Command failed is `spm_train` installed ?");

    // println!("{}", std::str::from_utf8(output.stdout));

    let trainer = UnigramTrainerBuilder::default()
        // .show_progress(false)
        .split_by_whitespace(true)
        // .space_char('▁')
        .build()
        .unwrap();
    let mut word_counts: Vec<(String, u32)> = vec![];
    let file = read_to_string(train_file).unwrap();
    let mut total = 0;
    for line in file.lines() {
        if total == n_sentences {
            break;
        }

        total += 1;
        match normalize(line) {
            Ok(formatted_line) => {
                word_counts.push((formatted_line, 1));
            }
            _ => (),
        }
    }
    println!("Kept {:?} sentences", word_counts.len());

    // println!("Start train {:?}", word_counts);
    let (model, _) = trainer._train(word_counts).unwrap();
    // println!("Stop train {:?}", model.get_vocab());
    // println!("Vocab {}", model.get_vocab().len());
    //
    model
        .save(
            std::path::Path::new("data/wikitext-103-raw"),
            Some("wiki_train_raw"),
        )
        .unwrap();

    let model = Unigram::load(std::path::Path::new(
        "data/wikitext-103-raw/wiki_train_raw-unigram.json",
    ))
    .unwrap();

    let file = read_to_string(test_file).unwrap();
    let encoded = std::str::from_utf8(&output.stdout).unwrap();

    let mut correct = 0;
    let mut total = 0;
    let mut n_tokenizer_tokens = 0;
    let mut n_spm_tokens = 0;
    for (tokenizer_line, spm_line) in file.lines().zip(encoded.lines()) {
        // println!("Tokenizer line {:?}", tokenizer_line);
        // println!("Spm line {:?}", spm_line);
        let tokenizer_tokens = model.encode(tokenizer_line, true);
        let mut spm_tokens: Vec<String> = spm_line
            .split(' ')
            .map(|s| s.to_string().replace('▁', " "))
            .collect();
        // XXX : For some reason spm_encode mangles trailing spaces which exist in wiki103.
        if spm_tokens.len() == 1 {
            spm_tokens.pop();
        }
        spm_tokens.push(" ".to_string());

        n_tokenizer_tokens += tokenizer_tokens.len();
        n_spm_tokens += spm_tokens.len();
        if tokenizer_tokens == spm_tokens {
            correct += 1;
        }
        total += 1;

        // assert_eq!(tokenizer_tokens, spm_tokens, "Failed on line {}", i + 1,);
    }
    let acc = (correct as f64) / (total as f64) * 100.0;
    println!("Total tokenizer tokens {}", n_tokenizer_tokens);
    println!("Total spm tokens {}", n_spm_tokens);
    println!("Total accuracy {}/{} ({:.2}%)", correct, total, acc);
    assert!(n_tokenizer_tokens < n_spm_tokens);
}

#[cfg(not(debug_assertions))]
fn normalize(s: &str) -> Result<String, ()> {
    let prefixed = format!(
        " {}",
        s.chars()
            .filter(|c| !c.is_control())
            .collect::<String>()
            .nfkc()
            .collect::<String>()
    );
    let mut vecs = vec![""];
    vecs.extend(prefixed.split_whitespace().collect::<Vec<_>>());

    let normalized: String = vecs.join(" ").to_string();
    let result = normalized.replace(' ', "▁");
    if result.len() > 4192 || result.is_empty() {
        return Err(());
    }
    Ok(result)
}

#[cfg(not(debug_assertions))]
#[test]
fn test_spm_compat_encode() {
    let n_sentences = 5_000;
    let train_file = "data/wikitext-103-raw/wiki.train.raw";
    let test_file = "data/wikitext-103-raw/wiki.test.raw";
    let output_file = "data/wikitext-103-raw/spm_wiki_103-exported.model";
    let output = Command::new("spm_train")
        .args(&[
            "--input",
            train_file,
            "--model_type",
            "unigram",
            "--model_prefix",
            "data/wikitext-103-raw/spm_wiki_103",
            "--input_sentence_size",
            &n_sentences.to_string(),
            "--num_threads",
            "1",
            "--shuffle_input_sentence",
            "0",
            "--character_coverage",
            "1",
        ])
        .output()
        .expect("Command failed is `spm_train` installed ?");
    if !output.status.success() {
        let err_msg = std::str::from_utf8(&output.stderr).unwrap();
        assert!(output.status.success(), "Command failed {}", err_msg)
    }
    println!("train: {}", std::str::from_utf8(&output.stderr).unwrap());

    let output = Command::new("spm_encode")
        .args(&[
            "--model",
            "data/wikitext-103-raw/spm_wiki_103.model",
            "--input",
            test_file,
        ])
        .output()
        .expect("Command failed is `spm_encode` installed ?");

    println!("{}", std::str::from_utf8(&output.stderr).unwrap());

    let _output = Command::new("spm_export_vocab")
        .args(&[
            "--model",
            "data/wikitext-103-raw/spm_wiki_103.model",
            "--output",
            output_file,
        ])
        .output()
        .expect("Command failed is `spm_export_vocab` installed ?");

    let model = Unigram::load_spm(std::path::Path::new(output_file)).unwrap();

    let file = read_to_string(test_file)
        .unwrap()
        .nfkc()
        .collect::<String>();

    let encoded = std::str::from_utf8(&output.stdout).unwrap();

    let mut correct = 0;
    let mut total = 0;
    let mut n_tokenizer_tokens = 0;
    let mut n_spm_tokens = 0;
    for (_i, (tokenizer_line, spm_line)) in file.lines().zip(encoded.lines()).enumerate() {
        // XXX: SPM filters lines by removing duplicate spaces.
        let mut filtered_line = String::new();
        let mut last_c = ' ';
        filtered_line.push(last_c);
        for c in tokenizer_line.chars() {
            if c == ' ' && last_c == ' ' {
                continue;
            }
            filtered_line.push(c);
            last_c = c;
        }
        // println!("Tokenizer line {:?}", filtered_line);
        let tokenizer_tokens = model.encode(&filtered_line, true);
        let mut spm_tokens: Vec<String> = spm_line
            .split(' ')
            .map(|s| s.to_string().replace('▁', " "))
            .collect();
        // XXX : For some reason spm_encode mangles trailing spaces which exist in wiki103.
        if spm_tokens == vec![""] {
            spm_tokens.pop();
        }
        spm_tokens.push(" ".to_string());

        n_tokenizer_tokens += tokenizer_tokens.len();
        n_spm_tokens += spm_tokens.len();
        if tokenizer_tokens == spm_tokens {
            correct += 1;
        } else {
            let mut iter = tokenizer_tokens.iter().zip(&spm_tokens).peekable();
            loop {
                if let Some((tok, spm)) = iter.next() {
                    let mut is_ok = false;
                    if tok != spm {
                        if let Some((next_tok, next_spm)) = iter.peek() {
                            if spm == *next_tok && tok == *next_spm {
                                is_ok = true;
                            }
                        }
                    }
                    if is_ok {
                        iter.next();
                        correct += 1;
                    } else {
                        assert_eq!(tok, spm, "Failed on line {}", _i + 1);
                    }
                } else {
                    break;
                }
            }
        }
        total += 1;
    }
    let acc = (correct as f64) / (total as f64) * 100.0;
    println!("Total tokenizer tokens {}", n_tokenizer_tokens);
    println!("Total spm tokens {}", n_spm_tokens);
    println!("Total accuracy {}/{} ({:.2}%)", correct, total, acc);
}

#[cfg(test)]
#[cfg(not(debug_assertions))]
mod tests {
    use super::*;

    #[test]
    fn test_normalize() {
        assert!(normalize("").is_err());
        assert!(normalize("       ").is_err());
        assert!(normalize("  ").is_err());

        // Sentence with heading/tailing/redundant spaces.
        assert_eq!("▁ABC", normalize("ABC").unwrap());
        assert_eq!("▁ABC", normalize(" ABC ").unwrap());
        assert_eq!("▁A▁B▁C", normalize(" A  B  C ").unwrap());
        assert_eq!("▁ABC", normalize("   ABC   ").unwrap());
        assert_eq!("▁ABC", normalize("   ＡＢＣ   ").unwrap());
        assert_eq!("▁ABC", normalize("　　ABC").unwrap());
        assert_eq!("▁ABC", normalize("　　ABC　　").unwrap());

        // NFKC char to char normalization.
        assert_eq!("▁123", normalize("①②③").unwrap());

        // NFKC char to multi-char normalization.
        assert_eq!("▁株式会社", normalize("㍿").unwrap());

        // Half width katakana, character composition happens.
        assert_eq!("▁グーグル", normalize(" ｸﾞｰｸﾞﾙ ").unwrap());

        assert_eq!(
            "▁I▁saw▁a▁girl",
            normalize(" I  saw a　 　girl　　").unwrap()
        );

        // Remove control chars.
        assert!(normalize(&format!("{}", std::char::from_u32(0x7F).unwrap())).is_err());
        assert!(normalize(&format!("{}", std::char::from_u32(0x8F).unwrap())).is_err());
        assert!(normalize(&format!("{}", std::char::from_u32(0x9F).unwrap())).is_err());
        assert!(normalize(&format!("{}", std::char::from_u32(0x0B).unwrap())).is_err());
        for c in 0x10..=0x1F {
            assert!(normalize(&format!("{}", std::char::from_u32(c).unwrap())).is_err());
        }
    }
}
