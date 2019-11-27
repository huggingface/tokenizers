extern crate jni;
extern crate tokenizers;

use std::error::Error;
use jni::{JNIEnv};
use jni::objects::{JObject, JValue, JString, JList};
use jni::sys::{jint, jobject};

use tokenizers::tokenizer::{PreTokenizer, Model, Decoder};

/// Takes a long ptr argument and reinterpret it as (&mut T) instance
#[inline]
pub unsafe fn reinterpret_cast<T>(ptr: i64) -> &'static mut T{
    &mut *(ptr as *mut T)
}


/// Generic pretokenize method from Pretokenizer trait. Takes a string and return the pretokenized list of string
pub fn pretokenize(_env: &JNIEnv, pretokenizer: &dyn PreTokenizer, text: &JString) -> Result<Vec<String>, String>{
    // Convert Java String to Rust String
    return match _env.get_string(*text) {
        Ok(s) => match s.to_str() {
            Ok(s) => Ok(pretokenizer.pre_tokenize(s)),
            Err(e) => Err(e.description().to_string())
        },
        Err(e) => Err("Unable to get string from JNIEnv".to_string())
    };
}

/// Generic decode method from Decoder trait. Takes a list of tokens and return the initial string.
pub fn decode(_env: &JNIEnv, decoder: &dyn Decoder, words: &JObject) -> Result<String, String>{
    match JList::from_env(&_env, *words){
        Ok(words) => match words.size(){
            Ok(size_t) => {
                let mut tokens: Vec<String> = Vec::with_capacity(size_t as usize);

                // Iterate over words in the ArrayList
                for i in 0..size_t{
                    match words.get(i as jint){
                        Ok(word) => match word {
                            Some(word) => match _env.get_string(JString::from(word)){
                                Ok(jstr) => match jstr.to_str() {
                                    Ok(jstr) => tokens.insert(i as usize, jstr.to_string()),
                                    Err(e) => return Err("Unable to decode UTF8 string from Java".to_string())
                                },
                                Err(e) => return Err("".to_string())
                            },
                            _ => return Err("Unable to convert JNI String to std::string".to_string())
                        }
                        Err(e) => return Err(format!("Unable to access ArrayList element {}", i).to_string())
                    }
                }
                return Ok(decoder.decode(tokens))
            }, Err(e) => return Err("Unable to get the size of the ArrayList".to_string())
        }, Err(e) => return Err("Unable to cast to ArrayList".to_string())
    }
}
