extern crate jni;
extern crate tokenizers;

use jni::objects::{JClass, JObject, JValue, JString, JList};
use jni::sys::{jint, jlong, jobject, jstring};
use jni::{JNIEnv};

use tokenizers::models::bpe::{Error as BpeError, BPE};
use tokenizers::tokenizer::{Token, PreTokenizer, Model};
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use jni::strings::JNIStr;
use std::borrow::Borrow;

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


fn string_vector_to_arraylist(_env: &JNIEnv, vector: &Vec<String>) -> Option<jobject>{
    match _env.new_object("java/util/ArrayList", "(I)V", &[JValue::Int(vector.len() as jint)]){
        Ok(jarray_) => match JList::from_env(&_env, jarray_){
            Ok(jarray_) => {
                // Push words into the ArrayList
                for item in vector {
                    match _env.new_string(&item) {
                        Ok(s) => jarray_.add(JObject::from(s)),
                        _ => {
                            _env.throw_new(STRING_DECODING_EXCEPTION, "Failed to convert Rust String to Java");
                            return Some(JObject::null().into_inner());
                        }
                    };
                }
                return Some(jarray_.into_inner());
            },
            Err(_) => {
                _env.throw_new(NATIVE_ALLOCATION_FAILED_EXCEPTION, "Failed to allocate ArrayList<String>");
                return Some(JObject::null().into_inner());
            }
        },
        Err(_) => {
            _env.throw_new(NATIVE_ALLOCATION_FAILED_EXCEPTION, "Failed to allocate ArrayList<String>");
            return Some(JObject::null().into_inner());
        }
    };
}


// Pretokenizer
#[no_mangle]
pub extern "system" fn Java_co_huggingface_tokenizers_pretokenizers_WhitespacePretokenizer_nativeHandle(_env: JNIEnv, _class: JClass, _obj: JObject) -> jlong {
    return Box::into_raw(Box::new(Whitespace)) as jlong;
}

#[no_mangle]
pub unsafe extern "system" fn Java_co_huggingface_tokenizers_pretokenizers_WhitespacePretokenizer_finalize(_env: JNIEnv, _obj: JObject){
    match _env.get_field(_obj, "ref", "J"){
        Ok(ptr) => {
            _env.set_field(_obj, "ref", "J", JValue::Long(-1));
            let boxed = Box::from_raw(ptr.j().unwrap() as *mut Whitespace);
            return
        },
        Err(_) => {
            _env.throw_new(NATIVE_ALLOCATION_FAILED_EXCEPTION, "Unable to retrieve Whitespace ptr");
            return
        }
    };

}

#[no_mangle]
pub unsafe extern "system" fn Java_co_huggingface_tokenizers_pretokenizers_WhitespacePretokenizer_pretokenize(_env: JNIEnv, _obj: JObject, s: JString) -> jobject {
    // Retrieve Whitespace instance ptr and reinterpret_cast<Whitespace>
    let whitespace = match _env.get_field(_obj, "ref", "J"){
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

    // Convert Java String to Rust String
    let tokens = match _env.get_string(s) {
        Ok(s) => match s.to_str() {
            Ok(s) => Some(whitespace.unwrap().pre_tokenize(s)),
            Err(e) => {
                _env.throw_new(STRING_DECODING_EXCEPTION, e.to_string());
                None
            }
        },
        Err(_) => {
            _env.throw_new(STRING_DECODING_EXCEPTION, "Failed to convert Java string parameter");
            None
        }
    };

    // Allocate ArrayList<String>
    let tokens = tokens.unwrap();
    return string_vector_to_arraylist(&_env, &tokens).unwrap();
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

    let handle = Box::into_raw(bpe.into()) as jlong;
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
            Ok(ptr) => {
                jprint(&_env, format!("{}", ptr).as_str());
                Some(&mut *(ptr as *mut BPE))
            },
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

    jprint(&_env, "tokenizer(): before");

    if (_env).exception_check().unwrap() {
        jprint(&_env, "catch exception");
    }

    let bpe_ = bpe.unwrap();
    let tokens = bpe_.tokenize(vec_words.unwrap().to_owned());

    jprint(&_env, "tokenize(): done");

    match _env.new_object("java/util/ArrayList", "(I)V", &[JValue::Int(2 as jint)]) {
        Ok(jarray_) => match JList::from_env(&_env, jarray_) {
            Ok(jarray_) => {
//                // Push words into the ArrayList
//                for token in tokens {
                    let j_token_args = &[
                        JValue::from(1 as jlong),
//                        JValue::from(JObject::from(_env.new_string(token.value).unwrap())),
                        JValue::from(JObject::from(_env.new_string("Hello").unwrap())),
                        JValue::from(0 as jint),
                        JValue::from(1 as jint)
                    ];

                    match _env.new_object("Lco/huggingface/tokenizers/Token;", "(JLjava/lang/String;II)V", j_token_args) {
                        Ok(j_token) => jarray_.add(j_token),
                        Err(_) => _env.throw_new(STRING_DECODING_EXCEPTION, "Failed to convert Rust Token to Java")
                    };
//                }
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