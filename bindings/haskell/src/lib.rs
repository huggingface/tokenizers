use std::ffi::CStr;
use std::os::raw::c_char;

use tokenizers::tokenizer::{Result, Tokenizer, EncodeInput};
use tokenizers::models::bpe::BPE;

#[no_mangle]
pub extern fn tokenize(x: *const c_char) {
  unsafe {
    let cstring = CStr::from_ptr(x);
    if let Ok(input) = cstring.to_str() {
    let bpe_builder = BPE::from_file("roberta-base-vocab.json", "roberta-base-merges.txt");
    let bpe = bpe_builder
        .dropout(0.1)
        .build().unwrap();
    let mut tokenizer = Tokenizer::new(bpe);
    let encoding = tokenizer.encode(input, false).unwrap();
    println!("{:?}", encoding.get_tokens());
    } else {
      panic!("Unable to read string.");
    }
  }
}