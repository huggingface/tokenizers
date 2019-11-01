/// This is the CLI binary for the Tokenizers project
use tokenizers::WhitespaceTokenizer;

fn main() {
    let s = "Hey man!";
    println!(
        "Tokenizing {:?} gives {:?}",
        s,
        WhitespaceTokenizer::tokenize(&s)
    );
}
