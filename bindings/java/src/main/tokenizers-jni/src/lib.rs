extern crate jni;
extern crate tokenizers;

use jni::objects::{JClass, JObject, JString};
use jni::sys::{jlong, jint, jstring};
use jni::{JNIEnv};

use tokenizers::models::bpe::{Error as BpeError, BPE};
use tokenizers::tokenizer::Token;


#[no_mangle]
pub unsafe extern "system" fn Java_co_huggingface_tokenizers_Token_internalTokenId(_env: JNIEnv, _obj: JObject, handle: jlong) -> jint{
    let token =&mut *(handle as *mut Token);
    return token.id as jint;
}

#[no_mangle]
pub unsafe extern "system" fn Java_co_huggingface_tokenizers_Token_internalValue(_env: JNIEnv, _obj: JObject, handle: jlong) -> jstring{
    let token = &mut *(handle as *mut Token);
    return _env.new_string(token.value.to_string()).unwrap().into_inner();
}

#[no_mangle]
pub unsafe extern "system" fn Java_co_huggingface_tokenizers_Token_internalOffsetStart(_env: JNIEnv, _obj: JObject, handle: jlong) -> jint{
    let token = &mut *(handle as *mut Token);
    return token.offsets.0 as jint;
}

#[no_mangle]
pub unsafe extern "system" fn Java_co_huggingface_tokenizers_Token_internalOffsetEnd(_env: JNIEnv, _obj: JObject, handle: jlong) -> jint{
    let token = &mut *(handle as *mut Token);
    return token.offsets.1 as jint;
}

#[no_mangle]
pub unsafe extern "system" fn Java_co_huggingface_tokenizers_Token_getHandle(_env: JNIEnv, _obj: JObject) -> jlong{
    let token = Box::new(Token{ id: 10, value: "Morgan".into(), offsets: (0, 1)});
    return Box::into_raw(token) as jlong;
}


// BPE
#[no_mangle]
pub extern "system" fn Java_co_huggingface_tokenizers_models_BpeTokenizer_getHandle(_env: JNIEnv, _class: JClass, vocabs: JString, merges: JString) -> jlong {
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
        let exception_class: JClass = _env.find_class("java/lang/Exception").unwrap();
        _env.throw_new(exception_class, bpe.err().unwrap());
        return -1 as jlong
    }else {
        return Box::into_raw(bpe.into()) as jlong;
    }
}
