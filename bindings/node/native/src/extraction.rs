use neon::prelude::*;
use serde::de::DeserializeOwned;

/// Common Error that can be converted to a neon::result::Throw and put
/// the js engine in a throwing state. Makes it way easier to manage errors
pub struct Error(pub String);
impl<T> From<T> for Error
where
    T: std::fmt::Display,
{
    fn from(e: T) -> Self {
        Self(format!("{}", e))
    }
}
impl From<Error> for neon::result::Throw {
    fn from(err: Error) -> Self {
        let msg = err.0;
        unsafe {
            neon_runtime::error::throw_error_from_utf8(msg.as_ptr(), msg.len() as i32);
            neon::result::Throw
        }
    }
}

pub type LibResult<T> = std::result::Result<T, Error>;

/// This trait is to be implemented for any type that we want to extract from
/// a JsValue.
pub trait FromJsValue: Sized {
    fn from_value<'c, C: Context<'c>>(from: Handle<'c, JsValue>, cx: &mut C) -> LibResult<Self>;
}
/// Any type that implements DeserializeOwned from serde can easily be converted
impl<T> FromJsValue for T
where
    T: DeserializeOwned,
{
    fn from_value<'c, C: Context<'c>>(from: Handle<'c, JsValue>, cx: &mut C) -> LibResult<Self> {
        let val: T = neon_serde::from_value(cx, from)?;
        Ok(val)
    }
}

/// This trait provides some extraction helpers, and we implement it for CallContext
/// so that we can easily extract any type that implements FromJsValue from the arguments.
pub trait Extract {
    fn extract<T: FromJsValue>(&mut self, pos: i32) -> LibResult<T>;
    fn extract_opt<T: FromJsValue>(&mut self, pos: i32) -> LibResult<Option<T>>;
    fn extract_vec<T: FromJsValue>(&mut self, pos: i32) -> LibResult<Vec<T>>;
    fn extract_vec_opt<T: FromJsValue>(&mut self, pos: i32) -> LibResult<Option<Vec<T>>>;
}
impl<'c, T: neon::object::This> Extract for CallContext<'c, T> {
    fn extract<E: FromJsValue>(&mut self, pos: i32) -> LibResult<E> {
        let val = self
            .argument_opt(pos)
            .ok_or_else(|| Error(format!("Argument {} is missing", pos)))?;
        let ext = E::from_value(val, self)?;
        Ok(ext)
    }

    fn extract_opt<E: FromJsValue>(&mut self, pos: i32) -> LibResult<Option<E>> {
        let val = self.argument_opt(pos);
        match val {
            None => Ok(None),
            Some(v) => {
                // For any optional value, we accept both `undefined` and `null`
                if v.downcast::<JsNull>().is_ok() || v.downcast::<JsUndefined>().is_ok() {
                    Ok(None)
                } else if v.downcast::<JsFunction>().is_ok() {
                    // Could be parsed as an empty object, so we don't accept JsFunction here
                    Err(Error("Cannot extract from JsFunction".into()))
                } else {
                    Ok(Some(E::from_value(v, self)?))
                }
            }
        }
    }

    fn extract_vec<E: FromJsValue>(&mut self, pos: i32) -> LibResult<Vec<E>> {
        let vec = self
            .argument_opt(pos)
            .ok_or_else(|| Error(format!("Argument {} is missing", pos)))?
            .downcast::<JsArray>()?
            .to_vec(self)?;

        vec.into_iter().map(|v| E::from_value(v, self)).collect()
    }

    fn extract_vec_opt<E: FromJsValue>(&mut self, pos: i32) -> LibResult<Option<Vec<E>>> {
        self.argument_opt(pos)
            .map(|v| {
                let vec = v.downcast::<JsArray>()?.to_vec(self)?;
                Ok(vec
                    .into_iter()
                    .map(|v| E::from_value(v, self))
                    .collect::<LibResult<Vec<_>>>()?)
            })
            .map_or(Ok(None), |v| v.map(Some))
    }
}
