extern crate jni;
extern crate tokenizers;

use jni::{JNIEnv};
use jni::objects::{JClass, JObject, JValue, JString, JList};
use jni::sys::{jint, jlong, jobject};

use std::error::Error;
use tokenizers::models::bpe::{Error as BpeError, BPE};
use tokenizers::tokenizer::{Token, PreTokenizer, Model};
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;


const NATIVE_ALLOCATION_FAILED_EXCEPTION: &str = "co/huggingface/tokenizers/exceptions/NativeAllocationFailedException";
const STRING_DECODING_EXCEPTION: &str = "co/huggingface/tokenizers/exceptions/StringDecodingException";


fn jprint(_env: &JNIEnv, msg: &str){
    let system = _env.find_class("java/lang/System").unwrap();
    let print_stream = _env.find_class("java/io/PrintStream").unwrap();

    match _env.new_string(msg){
        Ok(j_string) => {
            let out = _env
                .get_static_field(system, "out", "Ljava/io/PrintStream;")
                .unwrap();
            _env.call_method(
                out.l().unwrap(),
                "println",
                "(Ljava/lang/String;)V",
                &[JValue::from(JObject::from(j_string))],
            ).unwrap();
        },
        _ => {}
    }
}


fn string_vector_to_arraylist(_env: &JNIEnv, vector: &Vec<String>) -> Result<jobject, String>{
    match _env.new_object("java/util/ArrayList", "(I)V", &[JValue::Int(vector.len() as jint)]){
        Ok(jarray_) => match JList::from_env(&_env, jarray_){
            Ok(jarray_) => {
                // Push words into the ArrayList
                for item in vector {
                    match _env.new_string(&item) {
                        Ok(s) => jarray_.add(JObject::from(s)),
                        Err(e) => return Err("Unable to create new string from JNIEnv".to_string())
                    };
                }
                return Ok(jarray_.into_inner());
            },
            Err(e) => return Err("Object to List convertion failed".to_string())
        },
        Err(e) => return Err("Unable to allocate java.util.ArrayList".to_string())
    };
}

fn pretokenize(_env: &JNIEnv, pretokenizer: &dyn PreTokenizer, text: &JString) -> Result<Vec<String>, String>{
    // Convert Java String to Rust String
    return match _env.get_string(*text) {
        Ok(s) => match s.to_str() {
            Ok(s) => Ok(pretokenizer.pre_tokenize(s)),
            Err(e) => Err(e.description().to_string())
        },
        Err(e) => Err("Unable to get string from JNIEnv".to_string())
    };
}


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
    match _env.get_field(_obj, "handle", "J") {
        Ok(ptr) => {
            _env.set_field(_obj, "handle", "J", JValue::Long(-1));
            let _boxed = Box::from_raw(ptr.j().unwrap() as *mut ByteLevel);
        },
        Err(_) => _env.throw_new(NATIVE_ALLOCATION_FAILED_EXCEPTION, "Unable to retrieve ByteLevel ptr").unwrap()
    };
}

#[no_mangle]
pub unsafe extern "system" fn Java_co_huggingface_tokenizers_pretokenizers_ByteLevelPretokenizer_pretokenize(_env: JNIEnv, _obj: JObject, s: JString) -> jobject {
    // Retrieve Whitespace instance ptr and reinterpret_cast<Whitespace>
    let whitespace = match _env.get_field(_obj, "handle", "J"){
        Ok(ptr) => match ptr.j(){
            Ok(ptr) => Some(&mut *(ptr as *mut ByteLevel)),
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
            _env.set_field(j_bpe, "handle", "J", JValue::Long(handle));
            return j_bpe.into_inner();
        }
        Err(_) => {
            _env.throw_new(NATIVE_ALLOCATION_FAILED_EXCEPTION, "Unable to set BytePairEncoder.handle");
            return JObject::null().into_inner();
        }
    }
}

#[no_mangle]
pub unsafe extern "system" fn Java_co_huggingface_tokenizers_models_BytePairEncoder_tokenize(_env: JNIEnv, _obj: JObject, words: JObject) -> jobject {
    let bpe = match _env.get_field(_obj, "handle", "J") {
        Ok(ptr) => match ptr.j() {
            Ok(ptr) => Some(&mut *(ptr as *mut BPE)),
            Err(_) => {
                _env.throw_new(NATIVE_ALLOCATION_FAILED_EXCEPTION, "Failed to reinterpret BPE ptr");
                None
            }
        },
        Err(_) => {
            _env.throw_new(NATIVE_ALLOCATION_FAILED_EXCEPTION, "Failed to retrieve BPE ptr");
            None
        }
    };

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

    let tokens = bpe.unwrap().tokenize(vec_words.unwrap());

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

                            match _env.new_object("Lco/huggingface/tokenizers/Token;", "(JLjava/lang/String;II)V", j_token_args) {
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
                _env.throw_new(NATIVE_ALLOCATION_FAILED_EXCEPTION, "Unable to reinterpret jarray as List");
                return JObject::null().into_inner()
            }
        },
        Err(_) => {
        _env.throw_new(NATIVE_ALLOCATION_FAILED_EXCEPTION, "Failed to allocate ArrayList<Token>");
        return JObject::null().into_inner()
        }
    }

}