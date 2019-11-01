/// This is the CLI binary for the Tokenizers project
use tokenizers;

fn main() {
    println!("Hello, world!");
    let s = "Hey man!";
    println!("Tokenizing {:?} gives {:?}", s, tokenizers::tokenize(&s));
}
