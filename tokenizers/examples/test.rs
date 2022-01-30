use tokenizers::Tokenizer;
fn main() {
    let tokenizer =
        Tokenizer::from_pretrained("bigscience/oscar_13_languages_alpha_weight", None).unwrap();
    println!(
        "{:?}",
        tokenizer
            .encode("換金方法としてはコイン商への売却の他、", false)
            .unwrap()
            .get_tokens()
    );
}
