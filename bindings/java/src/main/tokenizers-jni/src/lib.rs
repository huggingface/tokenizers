extern crate jni;
extern crate tokenizers;

use jni::objects::{JClass, JObject, JValue, JString};
use jni::sys::{jint, jlong, jobject, jstring};
use jni::{JNIEnv};

//use tokenizers::models::bpe::{Error as BpeError, BPE};
use tokenizers::tokenizer::{Token, PreTokenizer};
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use jni::strings::JNIStr;

const NATIVE_ALLOCATION_FAILED_EXCEPTION: &str = "co/huggingface/tokenizers/exceptions/NativeAllocationFailedException";
const STRING_DECODING_EXCEPTION: &str = "co/huggingface/tokenizers/exceptions/StringDecodingException";


//
//    let system = _env.find_class("java/lang/System").unwrap();
//    let print_stream = _env.find_class("java/io/PrintStream").unwrap();
//
//    let out = _env
//        .get_static_field(system, "out", "Ljava/io/PrintStream;")
//        .unwrap();
//        _env.call_method(
//            out.l().unwrap(),
//            "println",
//            "(Ljava/lang/String;)V",
//            &[JValue::from(JObject::from(j_string.unwrap()))],
//        ).unwrap();


// Token
#[no_mangle]
pub unsafe extern "system" fn Java_co_huggingface_tokenizers_Token_internalTokenId(_env: JNIEnv, _obj: JObject, handle: jlong) -> jint{
    let token =&mut *(handle as *mut Token);
    return token.id as jint;
}

#[no_mangle]
pub unsafe extern "system" fn Java_co_huggingface_tokenizers_Token_internalValue(_env: JNIEnv, _obj: JObject, handle: jlong) -> jstring{
    let token = &mut *(handle as *mut Token);
    return _env.new_string(token.value.to_string()).unwrap().into_inner() as jstring;
}

#[no_mangle]
pub unsafe extern "system" fn Java_co_huggingface_tokenizers_Token_internalOffsetStart(_env: JNIEnv, _obj: JObject, handle: jlong) -> jint{
    let token = &mut *(handle as *mut Token);
    return token.offsets.0 as jint;
}

#[no_mangle]
pub unsafe extern "system" fn Java_co_huggingface_tokenizers_Token_internalOffsetEnd(_env: JNIEnv, _obj: JObject, handle: jlong) -> jint {
    let token = &mut *(handle as *mut Token);
    return token.offsets.1 as jint;
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
    match _env.new_object("java/util/ArrayList", "(I)V", &[JValue::Int(tokens.len() as jint)]){
        Ok(jarray_tokens) => {
            // Push words into the ArrayList
            for token in tokens {
                match _env.new_string(&token) {
                    Ok(s) => {
                        let jstring_s = JValue::Object(JObject::from(s));
                        _env.call_method(jarray_tokens, "add", "(Ljava/lang/Object;)Z", &[jstring_s]);
                    }
                    _ => {
                        _env.throw_new(STRING_DECODING_EXCEPTION, "Failed to convert Rust String to Java");
                    }
                };
            }
            return jarray_tokens.into_inner();
        },
        Err(_) => {
            _env.throw_new(NATIVE_ALLOCATION_FAILED_EXCEPTION, "Failed to allocate ArrayList<String>");
            return JObject::null().into_inner();
        }
    };
}

    // BPE
//#[no_mangle]
//pub extern "system" fn Java_co_huggingface_tokenizers_models_BpeTokenizer_getHandle(_env: JNIEnv, _class: JClass, vocabs: JString, merges: JString) -> jlong {
//    let vocabs: String = _env.get_string(vocabs)
//        .expect("Couldn't get vocab file path")
//        .into();
//
//    let merges: String = _env.get_string(merges)
//        .expect("Couldn't get merges file path")
//        .into();
//
//    let bpe: Result<Box<BPE>, String> = match tokenizers::models::bpe::BPE::from_files(&vocabs, &merges){
//        Ok(bpe) => Ok(Box::new(bpe)),
//        Err(e) => match e {
//            BpeError::BadVocabulary => Err("Bad vocab.json format".to_string()),
//            BpeError::Io(io) => Err(io.to_string()),
//            BpeError::JsonError(_) => Err("Error while parsing vocab json file".to_string()),
//            BpeError::MergeTokenOutOfVocabulary(token) => Err(format!("Merge token out of vocabulary: {}", token).to_string())
//        }
//    };
//
//    if bpe.is_err() {
//        let exception_class: JClass = _env.find_class("java/lang/Exception").unwrap();
//        _env.throw_new(exception_class, bpe.err().unwrap());
//        return -1 as jlong
//    }else {
//        return Box::into_raw(bpe.into()) as jlong;
//    }
//}
