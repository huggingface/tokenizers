mod common;

use common::*;
use tokenizers::tokenizer::{AddedToken, EncodeInput};

#[test]
fn handle_added_tokens() {
    let mut tokenizer = get_byte_level(true, false);
    tokenizer.add_special_tokens(&[AddedToken::from("<mask>".into()).lstrip(true)]);

    let input = String::from("I saw a <mask> 😺");
    let output = tokenizer.encode(EncodeInput::Single(input), false).unwrap();

    assert_eq!(
        output.get_tokens(),
        &["ĠI", "Ġsaw", "Ġa", "<mask>", "ĠðŁĺ", "º"]
    );
}
