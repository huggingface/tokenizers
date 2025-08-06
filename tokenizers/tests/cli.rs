use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::Path;

const BIN_NAME: &str = "tokenize";

#[test]
fn test_cli_tokenize_success() {
    // Prepare a minimal model file (assume one exists for test)
    let model_path = "./data/tokenizer.json";
    let text = "Hello world!";
    let mut cmd = Command::cargo_bin(BIN_NAME).unwrap();
    cmd.arg("tokenize")
        .arg("--model")
        .arg(model_path)
        .arg("--text")
        .arg(text);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Token IDs:"));
}

#[test]
fn test_cli_tokenize_missing_model() {
    let mut cmd = Command::cargo_bin(BIN_NAME).unwrap();
    cmd.arg("tokenize")
        .arg("--model")
        .arg("/nonexistent/model.json")
        .arg("--text")
        .arg("test");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("Failed to load tokenizer model"));
}

#[test]
fn test_cli_tokenize_invalid_text() {
    // Should still succeed, but may return empty or error if model is bad
    let model_path = "./data/tokenizer.json";
    let mut cmd = Command::cargo_bin(BIN_NAME).unwrap();
    cmd.arg("tokenize")
        .arg("--model")
        .arg(model_path)
        .arg("--text")
        .arg("");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Token IDs:"));
}

#[test]
fn test_cli_train_success() {
    // Prepare a small training file
    let train_file = "./data/small.txt";
    let output_model = "./data/test-model.json";
    if Path::new(output_model).exists() {
        fs::remove_file(output_model).unwrap();
    }
    let mut cmd = Command::cargo_bin(BIN_NAME).unwrap();
    cmd.arg("train")
        .arg("--files")
        .arg(train_file)
        .arg("--vocab-size")
        .arg("100")
        .arg("--output")
        .arg(output_model);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Model trained and saved to"));
    assert!(Path::new(output_model).exists());
    fs::remove_file(output_model).unwrap();
}

#[test]
fn test_cli_train_missing_file() {
    let mut cmd = Command::cargo_bin(BIN_NAME).unwrap();
    cmd.arg("train")
        .arg("--files")
        .arg("/nonexistent/data.txt")
        .arg("--output")
        .arg("/tmp/should-not-exist.json");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("Failed to train tokenizer"));
}

#[test]
fn test_cli_train_invalid_output() {
    // Output to a directory should fail
    let train_file = "./data/small.txt";
    let mut cmd = Command::cargo_bin(BIN_NAME).unwrap();
    cmd.arg("train")
        .arg("--files")
        .arg(train_file)
        .arg("--output")
        .arg("./data/");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("Failed to save trained model"));
}
