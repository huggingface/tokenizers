//! Public training/model APIs, thread policy, progress, and error boundaries.
use super::*;

#[test]
fn public_feed_train_and_model_reload_preserve_affixes() {
    use crate::Trainer;
    use tk_encode::{
        models::bpe::{BpeConfig, PipelineBPE},
        pipeline::{
            EncodeOptions, PipelineModel, PipelinePostProcessor, PipelinePreTokenizer,
            PipelineTokenizer,
        },
        vocab::bucket_added_vocabulary::AddedVocabulary,
    };
    let special = vec![AddedToken::from("[UNK]", true)];
    let mut trainer = BpeTrainer::builder()
        .vocab_size(20)
        .min_frequency(1)
        .continuing_subword_prefix("##".into())
        .end_of_word_suffix("</w>".into())
        .special_tokens(special.clone())
        .show_progress(false)
        .build();
    trainer
        .feed(["ab测 测", "ab测"].into_iter(), |text| {
            Ok(text.split_whitespace().map(str::to_owned).collect())
        })
        .unwrap();
    assert_eq!(trainer.get_word_count(), 2);
    let mut model = PipelineBPE::from_config(BpeConfig {
        vocab: [("[old]".into(), 0)].into_iter().collect(),
        ..Default::default()
    })
    .unwrap();
    assert_eq!(trainer.train(&mut model).unwrap(), special);
    let config = model.to_config().unwrap();
    assert_eq!(config.continuing_subword_prefix.as_deref(), Some("##"));
    assert_eq!(config.end_of_word_suffix.as_deref(), Some("</w>"));
    let expected = [
        ("测", vec![config.vocab["测</w>"]]),
        ("a测", vec![config.vocab["a"], config.vocab["##测</w>"]]),
        ("ab测", vec![config.vocab["ab测</w>"]]),
    ];
    let tokenizer = PipelineTokenizer::from_parts(
        AddedVocabulary::new(),
        vec![],
        PipelinePreTokenizer::None,
        PipelineModel::BPE(model),
        PipelinePostProcessor::default(),
        None,
        Default::default(),
        None,
        None,
    );
    let json = tk_serialize::to_json(&tokenizer).unwrap();
    let written: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(written["model"]["continuing_subword_prefix"], "##");
    assert_eq!(written["model"]["end_of_word_suffix"], "</w>");
    let reloaded = tk_serialize::from_json(&json).unwrap();
    for model in [&tokenizer, &reloaded] {
        for (input, expected) in &expected {
            let encoded = model
                .encode(*input, &EncodeOptions::no_specials())
                .wait()
                .unwrap();
            assert_eq!(
                encoded[0]
                    .ids()
                    .iter()
                    .map(|token| token.id())
                    .collect::<Vec<_>>(),
                *expected,
                "{input}"
            );
        }
    }
    check_feed_model_order_boundaries();
}

#[test]
fn wide_frequencies_and_signed_ledger_boundaries() {
    let trainer = BpeTrainer::builder()
        .vocab_size(8)
        .min_frequency(1)
        .show_progress(false)
        .build();
    for weight in [u64::from(u32::MAX) + 17, u64::MAX] {
        let mut trace = Vec::new();
        train(
            &trainer,
            WordCountsView::from_map(&counts(&[("ab", weight)])),
            2,
            Some(&mut |pair, count, id| trace.push((pair, count, id))),
        )
        .unwrap();
        assert_eq!(trace, [((0, 1), weight, 2)]);
        let (vocab, merges, _) = trainer.do_train(&counts(&[("ab", weight)])).unwrap();
        assert_eq!(vocab["ab"], 2);
        assert_eq!(merges, [("a".into(), "b".into())]);
    }
    assert!(trainer.do_train(&counts(&[("aba", u64::MAX)])).is_ok()); // separate keys, not a global u64 mass cap
    assert!(trainer.do_train(&counts(&[("abab", u64::MAX)])).is_err()); // one key's frequency overflows
    let trainer = BpeTrainer::builder()
        .vocab_size(8)
        .min_frequency(1)
        .show_progress(false)
        .end_of_word_suffix("a".into())
        .build();
    assert!(
        trainer
            .do_train(&counts(&[("ab", i64::MAX as u64)]))
            .is_ok()
    );
    assert!(
        trainer
            .do_train(&counts(&[("ab", i64::MAX as u64 + 1)]))
            .is_err()
    );
    assert!(
        trainer
            .do_train(&counts(&[("abc", i64::MAX as u64)]))
            .is_err()
    ); // edge mass exceeds the signed policy domain
}

#[test]
fn public_thread_policy_in_isolated_processes() {
    use crate::trainers::bpe::word_counts::WordCounts;
    use std::sync::atomic::Ordering;
    const CHILD: &str = "BPE_THREAD_POLICY_TEST_CHILD";
    if let Ok(setting) = std::env::var(CHILD) {
        let fields: Vec<_> = setting.split(':').collect();
        let [parallel, workers, ambient] = fields.as_slice() else {
            panic!("expected parallel:training-workers:ambient-workers");
        };
        let workers = workers.parse().unwrap();
        let ambient = ambient.parse().unwrap();
        let parallel = *parallel == "true";
        tk_encode::parallelism::set_num_threads(workers);
        tk_encode::parallelism::set_parallelism(parallel);
        execution::EXPECTED_WORKERS.store(if parallel { workers } else { 1 }, Ordering::Relaxed);
        rayon::ThreadPoolBuilder::new()
            .num_threads(ambient)
            .build()
            .unwrap()
            .install(|| {
                check_feed_nonfused_none_boundaries(workers, parallel);
                check_feed_flush_boundaries(ambient, parallel);
                public_feed_train_and_model_reload_preserve_affixes();
                feed_preserves_flat_counts_and_trainer_equality();
                feed_error_runs_callbacks_without_replacing_previous_counts();

                let mut trainer = BpeTrainer::builder()
                    .vocab_size(10)
                    .min_frequency(1)
                    .show_progress(false)
                    .build();
                let words = counts(&[("aaaaa", 3), ("abcabc", 2), ("测测测", 1)]);
                execution::OBSERVED_TASKS.store(0, Ordering::Relaxed);
                let parts = trainer.do_train(&words).unwrap();
                assert!(execution::OBSERVED_TASKS.load(Ordering::Relaxed) > 0);
                assert_eq!(
                    parts,
                    trainer.do_train_observed(&words, |_, _, _| {}).unwrap()
                );
                trainer.words = WordCounts::from_map(words);
                execution::OBSERVED_TASKS.store(0, Ordering::Relaxed);
                assert_eq!(trainer.train_vocab().unwrap(), parts);
                assert!(execution::OBSERVED_TASKS.load(Ordering::Relaxed) > 0);
            });
        return;
    }
    for setting in [
        "false:4:4",
        "true:1:1",
        "true:2:2",
        "true:4:4",
        "true:4:2",
        "true:4:16",
        "false:4:2",
    ] {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "trainers::bpe::engine::tests::public_contract::public_thread_policy_in_isolated_processes",
                "--nocapture",
            ])
            .env(CHILD, setting)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{setting}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
}

#[test]
fn zero_merge_paths_preserve_count_overflow_and_signed_policy_checks() {
    let mut trainer = BpeTrainer::builder()
        .vocab_size(2)
        .min_frequency(1)
        .show_progress(false)
        .build();
    for words in [counts(&[("ab", 7)]), counts(&[("aba", u64::MAX)])] {
        let (vocab, merges, _) = trainer.do_train(&words).unwrap();
        assert_eq!(vocab.len(), 2);
        assert!(merges.is_empty());
    }
    assert!(trainer.do_train(&counts(&[("abab", u64::MAX)])).is_err());
    trainer.end_of_word_suffix = Some("a".into());
    assert!(
        trainer
            .do_train(&counts(&[("ab", i64::MAX as u64)]))
            .is_ok()
    );
    assert!(
        trainer
            .do_train(&counts(&[("ab", i64::MAX as u64 + 1)]))
            .is_err()
    );
    assert!(
        trainer
            .do_train(&counts(&[("abc", i64::MAX as u64)]))
            .is_err()
    );
}

#[test]
fn public_json_progress_preserves_upstream_fields() {
    const CHILD: &str = "BPE_JSON_PROGRESS_TEST_CHILD";
    if let Ok(setting) = std::env::var(CHILD) {
        use tk_encode::utils::progress::ProgressFormat;
        tk_encode::parallelism::set_num_threads(1);
        let trainer = BpeTrainer::builder()
            .show_progress(setting == "json:bar")
            .progress_format(if setting.starts_with("json") {
                ProgressFormat::JsonLines
            } else {
                ProgressFormat::Silent
            })
            .vocab_size(match setting.as_str() {
                "json:zero" => 3,
                "json:empty" => 0,
                _ => 10,
            })
            .min_frequency(1)
            .build();
        let words = if setting == "json:empty" {
            AHashMap::new()
        } else {
            counts(&[("aaaaa", 3), ("abcabc", 2)])
        };
        let (_, merges, _) = trainer.do_train(&words).unwrap();
        println!("BPE_MERGE_COUNT={}", merges.len());
        return;
    }
    for setting in [
        "json:bar",
        "json:no-bar",
        "json:zero",
        "json:empty",
        "silent",
    ] {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "trainers::bpe::engine::tests::public_contract::public_json_progress_preserves_upstream_fields",
                "--nocapture",
            ])
            .env(CHILD, setting)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let stderr = String::from_utf8(result.stderr).unwrap();
        if setting == "silent" {
            assert!(stderr.is_empty());
            continue;
        }
        let records: Vec<serde_json::Value> = stderr
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert!(!records.is_empty());
        let mut stages = std::collections::HashSet::new();
        for record in &records {
            if stages.insert(record["stage"].as_str().unwrap()) {
                assert_eq!(record["current"], 0, "stage should start at zero: {record}");
            }
            assert_eq!(record.as_object().unwrap().len(), 3);
            assert!(record["stage"].is_string());
            assert!(record["current"].is_u64());
            assert!(record["total"].is_u64());
        }
        let stdout = String::from_utf8(result.stdout).unwrap();
        let merges: u64 = stdout
            .lines()
            .find_map(|line| line.strip_prefix("BPE_MERGE_COUNT="))
            .unwrap()
            .parse()
            .unwrap();
        let finished = records
            .iter()
            .rev()
            .find(|record| record["stage"] == "Compute merges")
            .unwrap();
        assert_eq!(finished["current"], merges);
        assert_eq!(finished["total"], merges);
    }
}

#[test]
fn feed_preserves_flat_counts_and_trainer_equality() {
    use crate::Trainer;
    let mut trainer = BpeTrainer::builder().show_progress(false).build();
    trainer
        .feed(
            (0..257)
                .map(|i| if i % 2 == 0 { "even" } else { "odd" })
                .chain(std::iter::once("singleton")),
            |word| Ok(vec![word.into(), "shared".into(), "shared".into()]),
        )
        .unwrap();
    let json = serde_json::to_value(&trainer).unwrap();
    assert_eq!(
        json["words"],
        serde_json::json!({"even": 129, "odd": 128, "singleton": 1, "shared": 516})
    );
    let restored: BpeTrainer = serde_json::from_value(json).unwrap();
    assert_eq!(trainer, restored);
    assert_eq!(trainer, trainer.clone());
    let words = counts(&[
        ("even", 129),
        ("odd", 128),
        ("singleton", 1),
        ("shared", 516),
    ]);
    assert_eq!(
        trainer.train_vocab().unwrap(),
        trainer.do_train(&words).unwrap()
    );
    assert_eq!(
        trainer.train_vocab().unwrap(),
        restored.train_vocab().unwrap()
    );
    // Deserialization retains the original map's full count domain.
    let mut json = serde_json::to_value(&trainer).unwrap();
    json["words"]["shared"] = serde_json::json!(515);
    let altered: BpeTrainer = serde_json::from_value(json.clone()).unwrap();
    assert_ne!(trainer, altered);
    assert_ne!(restored, altered);
    json["words"] = serde_json::json!({"zero": 0, "wide": u64::MAX});
    let restored: BpeTrainer = serde_json::from_value(json.clone()).unwrap();
    assert_eq!(serde_json::to_value(restored).unwrap(), json);
    let edge_words = vec![
        "".into(),
        "中文🙂".into(),
        "e\u{301}".into(),
        "long".repeat(2048),
    ];
    trainer
        .feed(["empty", "words"].into_iter(), |input| {
            Ok(if input == "empty" {
                Vec::new()
            } else {
                edge_words.clone()
            })
        })
        .unwrap();
    assert_eq!(
        trainer.words,
        crate::trainers::bpe::word_counts::WordCounts::from_map(
            edge_words
                .into_iter()
                .map(|word| (word.into(), 1))
                .collect()
        )
    );
    trainer
        .feed(std::iter::empty::<&str>(), |_| unreachable!())
        .unwrap();
    assert_eq!(trainer.get_word_count(), 0);
}

#[test]
fn feed_error_runs_callbacks_without_replacing_previous_counts() {
    use crate::Trainer;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let mut trainer = BpeTrainer::builder().show_progress(false).build();
    trainer
        .feed(["previous"].into_iter(), |word| Ok(vec![word.into()]))
        .unwrap();
    let previous = trainer.clone();
    let calls = AtomicUsize::new(0);
    let error = trainer
        .feed((0..257).map(|i| i.to_string()), |word| {
            calls.fetch_add(1, Ordering::Relaxed);
            if word == "7" {
                Err("process failed".into())
            } else {
                Ok(vec![word.into()])
            }
        })
        .unwrap_err();
    assert_eq!(error.to_string(), "process failed");
    assert_eq!(calls.load(Ordering::Relaxed), 257);
    assert_eq!(trainer, previous);
    assert_eq!(
        serde_json::to_value(&trainer).unwrap()["words"],
        serde_json::json!({"previous": 1})
    );
}

fn nonfused_inputs_resuming_after_none(before_none: usize) -> impl Iterator<Item = String> + Send {
    let mut step = 0;
    std::iter::from_fn(move || {
        let current = step;
        step += 1;
        if current < before_none {
            Some(format!("word{current}"))
        } else if current == before_none {
            None
        } else if current == before_none + 1 {
            Some("tail".into())
        } else {
            None
        }
    })
}

fn check_feed_nonfused_none_boundaries(workers: usize, parallel: bool) {
    use crate::Trainer;
    use crate::trainers::bpe::word_counts::WordCounts;
    use std::sync::atomic::{AtomicUsize, Ordering};

    assert_eq!(tk_encode::parallelism::num_threads(), workers);
    assert_eq!(tk_encode::parallelism::get_parallelism(), parallel);
    for before_none in [31, 32, 33] {
        let callbacks = AtomicUsize::new(0);
        let mut trainer = BpeTrainer::builder().show_progress(false).build();
        trainer
            .feed(nonfused_inputs_resuming_after_none(before_none), |word| {
                callbacks.fetch_add(1, Ordering::Relaxed);
                Ok(vec![word.to_owned()])
            })
            .unwrap();

        assert_eq!(callbacks.load(Ordering::Relaxed), before_none);
        assert_eq!(trainer.get_word_count(), before_none);
        let expected = (0..before_none)
            .map(|index| (format!("word{index}").into(), 1))
            .collect();
        assert_eq!(trainer.words, WordCounts::from_map(expected));
    }
}

struct FeedWorkerGate {
    workers: std::sync::Mutex<std::collections::HashSet<usize>>,
    ready: std::sync::Condvar,
}

impl FeedWorkerGate {
    fn new() -> Self {
        Self {
            workers: std::sync::Mutex::new(std::collections::HashSet::new()),
            ready: std::sync::Condvar::new(),
        }
    }

    fn wait_for_second_worker(&self) {
        use std::time::{Duration, Instant};
        let worker = rayon::current_thread_index().expect("feed callback should run in Rayon");
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut workers = self.workers.lock().unwrap();
        if !workers.insert(worker) {
            return;
        }
        self.ready.notify_all();
        while workers.len() < 2 {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(
                !remaining.is_zero(),
                "only one Rayon worker ran feed callbacks"
            );
            let (next, timeout) = self.ready.wait_timeout(workers, remaining).unwrap();
            workers = next;
            assert!(
                !timeout.timed_out() || workers.len() >= 2,
                "only one Rayon worker ran feed callbacks"
            );
        }
    }

    fn participants(&self) -> usize {
        self.workers.lock().unwrap().len()
    }
}

fn check_feed_flush_boundaries(pool_workers: usize, parallel: bool) {
    use crate::Trainer;
    use crate::trainers::bpe::word_counts::WordCounts;
    use std::sync::atomic::{AtomicUsize, Ordering};
    const INPUTS_PER_BATCH: usize = 32;

    assert_eq!(tk_encode::parallelism::get_parallelism(), parallel);
    assert_eq!(rayon::current_num_threads(), pool_workers);
    let coordinated = parallel && pool_workers > 1;
    for unique in [2047, 2048, 2049] {
        let output: Vec<_> = (0..unique).map(|i| format!("word{i}")).collect();
        let batch_count = pool_workers.max(1) * 2;
        let input_count = batch_count * INPUTS_PER_BATCH;
        let gate = coordinated.then(FeedWorkerGate::new);
        let callbacks = AtomicUsize::new(0);
        let mut expected: AHashMap<CompactString, u64> = output
            .iter()
            .map(|word| (word.as_str().into(), batch_count as u64))
            .collect();
        expected.insert("shared".into(), (input_count - batch_count) as u64);
        let mut trainer = BpeTrainer::builder().show_progress(false).build();
        trainer
            .feed((0..input_count).map(|index| index.to_string()), |input| {
                callbacks.fetch_add(1, Ordering::Relaxed);
                if let Some(gate) = &gate {
                    gate.wait_for_second_worker();
                }
                let index = input.parse::<usize>().unwrap();
                if index % INPUTS_PER_BATCH == 0 {
                    Ok(output.clone())
                } else {
                    Ok(vec!["shared".into()])
                }
            })
            .unwrap();
        assert_eq!(callbacks.load(Ordering::Relaxed), input_count);
        assert_eq!(trainer.get_word_count(), unique + 1);
        assert_eq!(
            trainer.words.view().iter().map(|(_, n)| n).sum::<u64>(),
            (unique * batch_count + input_count - batch_count) as u64
        );
        assert_eq!(trainer.words, WordCounts::from_map(expected));
        if let Some(gate) = &gate {
            assert!(gate.participants() >= 2);
        }
        assert_eq!(matches!(trainer.words, WordCounts::Entries(_)), coordinated);
    }
}

fn check_feed_model_order_boundaries() {
    use crate::Trainer;
    use crate::trainers::bpe::word_counts::WordCounts;
    let mut trainer = BpeTrainer::builder()
        .vocab_size(8)
        .min_frequency(1)
        .show_progress(false)
        .continuing_subword_prefix("##".into())
        .end_of_word_suffix("</w>".into())
        .build();
    // One word fixes decorated-ID allocation independently of count traversal.
    trainer
        .feed(["ab", "ab"].into_iter(), |word| Ok(vec![word.into()]))
        .unwrap();
    let before = trainer.clone();
    let expected = (
        [("a", 0), ("b", 1), ("##b</w>", 2), ("ab</w>", 3)]
            .into_iter()
            .map(|(token, id)| (token.into(), id))
            .collect(),
        vec![("a".into(), "##b</w>".into())],
        vec![],
    );
    assert_eq!(trainer.train_vocab().unwrap(), expected);
    assert_eq!(trainer.train_vocab().unwrap(), expected);
    assert_eq!(trainer.do_train(&counts(&[("ab", 2)])).unwrap(), expected);
    assert_eq!(trainer, before);

    let trainer = BpeTrainer::builder()
        .show_progress(false)
        .limit_alphabet(1)
        .vocab_size(1)
        .build();
    // Equal-frequency alphabet cutoffs may retain either character upstream.
    // Traversal order is unspecified; check the exact allowed models for each view.
    for words in [
        WordCounts::from_map(counts(&[("a", 1), ("b", 1)])),
        WordCounts::from_entries(vec![("a".into(), 1), ("b".into(), 1)]),
        WordCounts::from_entries(vec![("b".into(), 1), ("a".into(), 1)]),
    ] {
        let (vocab, merges, special) = trainer.train_counts(words.view()).unwrap();
        assert!(
            vocab == [("a".into(), 0)].into_iter().collect()
                || vocab == [("b".into(), 0)].into_iter().collect()
        );
        assert!(merges.is_empty());
        assert!(special.is_empty());
    }
}
