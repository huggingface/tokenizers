use tokenizers::Tokenizer;

fn main() {
    // START load
    let tokenizer = Tokenizer::from_file("data/roberta.json").unwrap();
    // END load

    let example = "This is an example";
    let ids = vec![713, 16, 41, 1246];
    let tokens = vec!["This", "Ġis", "Ġan", "Ġexample"];

    let encodings = tokenizer.encode(example, false).unwrap();

    assert_eq!(encodings.get_ids(), ids);
    assert_eq!(encodings.get_tokens(), tokens);

    let decoded = tokenizer.decode(ids, false).unwrap();
    assert_eq!(decoded, example);
}
