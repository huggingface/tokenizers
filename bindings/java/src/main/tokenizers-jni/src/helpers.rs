use std::error::Error;
use jni::{JNIEnv};
use jni::objects::{JObject, JValue, JString, JList};
use jni::sys::{jint, jobject};

pub fn jprint(_env: &JNIEnv, msg: &str){
    let system   = _env.find_class("java/lang/System").unwrap();
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


pub fn string_vector_to_arraylist(_env: &JNIEnv, vector: &Vec<String>) -> Result<jobject, String>{
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
            Err(e) => return Err("Object to List conversion failed".to_string())
        },
        Err(e) => return Err("Unable to allocate java.util.ArrayList".to_string())
    };
}