mod glue;
mod helpers;

extern crate jni;
extern crate tokenizers;

use std::error::Error;

use jni::{JNIEnv};
use jni::objects::{JClass, JObject, JValue, JString, JList};
use jni::sys::{jint, jlong, jobject, jstring};

use tokenizers::models::bpe::{Error as BpeError, BPE};
use tokenizers::tokenizer::{Token, PreTokenizer, Model, Decoder};
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;

use helpers::string_vector_to_arraylist;
use glue::{decode, pretokenize, reinterpret_cast};

// Constants
const NATIVE_ALLOCATION_FAILED_EXCEPTION: &str = "co/huggingface/tokenizers/exceptions/NativeAllocationFailedException";
const STRING_DECODING_EXCEPTION: &str = "co/huggingface/tokenizers/exceptions/StringDecodingException";
const TOKEN_CLASS: &str = "Lco/huggingface/tokenizers/Token;";

// Pretokenizer

//// Whitespace
#[no_mangle]
pub extern "system" fn Java_co_huggingface_tokenizers_pretokenizers_WhitespacePretokenizer_allocate(_env: JNIEnv, _class: JClass, _obj: JObject) -> jlong {
    return Box::into_raw(Box::new(Whitespace)) as jlong;
}

#[no_mangle]
pub unsafe extern "system" fn Java_co_huggingface_tokenizers_pretokenizers_WhitespacePretokenizer_finalize(_env: JNIEnv, _obj: JObject) {
    match _env.get_field(_obj, "handle", "J") {
        Ok(ptr) => {
            _env.set_field(_obj, "handle", "J", JValue::Long(-1));
            let _boxed = Box::from_raw(ptr.j().unwrap() as *mut Whitespace);
        },
        Err(_) => _env.throw_new(NATIVE_ALLOCATION_FAILED_EXCEPTION, "Unable to retrieve Whitespace ptr").unwrap()
    };
}

#[no_mangle]
pub unsafe extern "system" fn Java_co_huggingface_tokenizers_pretokenizers_WhitespacePretokenizer_pretokenize(_env: JNIEnv, _obj: JObject, s: JString) -> jobject {
    // Retrieve Whitespace instance ptr and reinterpret_cast<Whitespace>
    let whitespace = match _env.get_field(_obj, "handle", "J"){
        Ok(ptr) => match ptr.j(){
            Ok(ptr) => Some(&mut *(ptr as *mut Whitespace)),
            Err(_) => {
                _env.throw_new(NATIVE_ALLOCATION_FAILED_EXCEPTION, "Failed to reinterpret Whitespace ptr");
                None
            }
        },
        Err(_) => {
            _env.throw_new(NATIVE_ALLOCATION_FAILED_EXCEPTION, "Failed to retrieve Whitespace ptr");
            None
        }
    };

    match pretokenize(&_env, whitespace.unwrap(), &s){
        Ok(tokens) => match string_vector_to_arraylist(&_env, &tokens){
            Ok(jarray_tokens) => return jarray_tokens,
            _ => {
                _env.throw_new(NATIVE_ALLOCATION_FAILED_EXCEPTION, "");
                return JObject::null().into_inner();
            }
        },
        Err(e) => {
            _env.throw_new(NATIVE_ALLOCATION_FAILED_EXCEPTION, e);
            return JObject::null().into_inner();
        }
    }
}

//// Byte Level
#[no_mangle]
pub extern "system" fn Java_co_huggingface_tokenizers_pretokenizers_ByteLevelPretokenizer_allocate(_env: JNIEnv, _class: JClass, _obj: JObject) -> jlong {
    return Box::into_raw(Box::new(ByteLevel)) as jlong;
}

#[no_mangle]
pub unsafe extern "system" fn Java_co_huggingface_tokenizers_pretokenizers_ByteLevelPretokenizer_finalize(_env: JNIEnv, _obj: JObject) {
    // Finalize is very special call, let's handle it with extra care to be sure memory (if any) is desallocated properly
    match _env.get_field(_obj, "handle", "J") {
        Ok(ptr) => {
            _env.set_field(_obj, "handle", "J", JValue::Long(-1));
            let pretokenizer = reinterpret_cast::<ByteLevel>(ptr.j().unwrap());
            let _boxed = Box::from_raw(pretokenizer);
        },
        Err(_) => _env.throw_new(NATIVE_ALLOCATION_FAILED_EXCEPTION, "Unable to retrieve ByteLevel ptr").unwrap()
    };
}

#[no_mangle]
pub unsafe extern "system" fn Java_co_huggingface_tokenizers_pretokenizers_ByteLevelPretokenizer_pretokenize(_env: JNIEnv, _obj: JObject, s: JString) -> jobject {
    // Retrieve Whitespace instance ptr and reinterpret_cast<Whitespace>
    let pretokenizer = reinterpret_cast::<ByteLevel>(_env.get_field(_obj, "handle", "J").unwrap().j().unwrap());

    match pretokenize(&_env, pretokenizer, &s){
        Ok(tokens) => match string_vector_to_arraylist(&_env, &tokens){
            Ok(jarray_tokens) => return jarray_tokens,
            _ => {
                _env.throw_new(NATIVE_ALLOCATION_FAILED_EXCEPTION, "").unwrap();
                return JObject::null().into_inner();
            }
        },
        Err(e) => {
            _env.throw_new(NATIVE_ALLOCATION_FAILED_EXCEPTION, e).unwrap();
            return JObject::null().into_inner();
        }
    }
}

#[no_mangle]
pub unsafe extern "system" fn Java_co_huggingface_tokenizers_pretokenizers_ByteLevelPretokenizer_decode(_env: JNIEnv, _obj: JObject, words: JObject) -> jstring {
    // Retrieve ByteLevel instance ptr and reinterpret_cast<ByteLevel>
    let pretokenizer = reinterpret_cast::<ByteLevel>(_env.get_field(_obj, "handle", "J").unwrap().j().unwrap());

    match decode(&_env, pretokenizer, &words){
        Ok(s) => match _env.new_string(s.as_str()){
            Ok(jstr) => return jstr.into_inner(),
            _ => {
                _env.throw_new(STRING_DECODING_EXCEPTION, "");
                return JObject::null().into_inner();
            }
        },
        Err(e) => {
            _env.throw_new(NATIVE_ALLOCATION_FAILED_EXCEPTION, e);
            return JObject::null().into_inner();
        }
    }
}

// BPE
#[no_mangle]
pub extern "system" fn Java_co_huggingface_tokenizers_models_BytePairEncoder_fromFiles(_env: JNIEnv, _class: JClass, vocabs: JString, merges: JString) -> jobject {
    let vocabs: String = _env.get_string(vocabs)
        .expect("Couldn't get vocab file path")
        .into();

    let merges: String = _env.get_string(merges)
        .expect("Couldn't get merges file path")
        .into();

    let bpe: Result<Box<BPE>, String> = match tokenizers::models::bpe::BPE::from_files(&vocabs, &merges){
        Ok(bpe) => Ok(Box::new(bpe)),
        Err(e) => match e {
            BpeError::BadVocabulary => Err("Bad vocab.json format".to_string()),
            BpeError::Io(io) => Err(io.to_string()),
            BpeError::JsonError(_) => Err("Error while parsing vocab json file".to_string()),
            BpeError::MergeTokenOutOfVocabulary(token) => Err(format!("Merge token out of vocabulary: {}", token).to_string())
        }
    };

    if bpe.is_err() {
        _env.throw_new(NATIVE_ALLOCATION_FAILED_EXCEPTION, "Unable to allocate BytePairEncoder");
        return JObject::null().into_inner()
    }

    let handle = Box::into_raw(bpe.unwrap()) as jlong;
    match _env.new_object("Lco/huggingface/tokenizers/models/BytePairEncoder;", "()V", &[]){
        Ok(j_bpe) => {
            _env.set_field(j_bpe, "handle", "J", JValue::Long(handle)).unwrap();
            return j_bpe.into_inner();
        }
        Err(_) => {
            _env.throw_new(NATIVE_ALLOCATION_FAILED_EXCEPTION, "Unable to set BytePairEncoder.handle").unwrap();
            return JObject::null().into_inner();
        }
    }
}

#[no_mangle]
pub unsafe extern "system" fn Java_co_huggingface_tokenizers_models_BytePairEncoder_tokenize(_env: JNIEnv, _obj: JObject, words: JObject) -> jobject {
    // Retrieve BytePairEncoder object
    let bpe = reinterpret_cast::<BPE>(_env.get_field(_obj, "handle", "J").unwrap().j().unwrap());

    let vec_words = match JList::from_env(&_env, words){
        Ok(j_words) => {
            // Allocate vector on Rust side
            let size_t = match j_words.size(){
                Ok(size) => size as usize,
                Err(_) => 0 as usize
            };

            let mut v_words: Vec<String> = Vec::with_capacity(size_t);
            for i in 0..size_t {
                let s = JString::from(j_words.get(i as i32).unwrap().unwrap());
                let j_word = _env.get_string(s).unwrap();
                v_words.insert(i, j_word.to_str().unwrap().to_string());
            }
            Some(v_words)
        },
        Err(e) => {
            _env.throw_new(STRING_DECODING_EXCEPTION, "Unable to decode token");
            None
        }
    };

    let tokens = bpe.tokenize(vec_words.unwrap());

    match _env.new_object("java/util/ArrayList", "(I)V", &[JValue::Int(2 as jint)]) {
        Ok(jarray_) => match JList::from_env(&_env, jarray_) {
            Ok(jarray_) => {
                // Push words into the ArrayList
                tokens.iter().for_each(|token| {
                    match _env.new_string(token.value.as_str()){

                        // TODO : impl JavaMapper for Token (reduce boilerplate)
                        Ok(jtoken_value) => {
                            let j_token_args = &[
                                JValue::from(token.id as jlong),
                                JValue::from(JObject::from(jtoken_value)),
                                JValue::from(token.offsets.0 as jint),
                                JValue::from(token.offsets.1 as jint)
                            ];

                            match _env.new_object(TOKEN_CLASS, "(JLjava/lang/String;II)V", j_token_args) {
                                Ok(j_token) => jarray_.add(j_token).unwrap(),
                                Err(_) => _env.throw_new(NATIVE_ALLOCATION_FAILED_EXCEPTION, "Failed to allocate Token").unwrap()
                            }
                        },
                        Err(_) => _env.throw_new(STRING_DECODING_EXCEPTION,"Failed to convert Rust string to Java").unwrap()
                    }
                });

                return jarray_.into_inner()
            },
            Err(_) => {
                _env.throw_new(NATIVE_ALLOCATION_FAILED_EXCEPTION, "Unable to reinterpret jarray as List").unwrap_or_default();
                return JObject::null().into_inner()
            }
        },
        Err(_) => {
            _env.throw_new(NATIVE_ALLOCATION_FAILED_EXCEPTION, "Failed to allocate ArrayList<Token>").unwrap_or_default();
            return JObject::null().into_inner()
        }
    }

}