use crate::utils::{RustOwned, free_ptr, new_ptr};
use std::ffi::{CString, c_char};
use std::panic::{self, AssertUnwindSafe};

pub struct Error {
    message: CString,
}

impl RustOwned for Error {}

impl Error {
    pub(crate) fn into_ptr(msg: impl std::fmt::Display) -> *mut Error {
        let message = CString::new(msg.to_string())
            .unwrap_or_else(|_| CString::new("error message contained a NUL byte").unwrap());
        new_ptr(Error { message })
    }
}

/// Returns a pointer to `err`'s message, or NULL if `err` is NULL. The message points into `err`
/// and is valid until `tk_error_free`.
///
/// # Safety
/// `err` must be NULL, or an error returned by a `tk_*` call that is not freed while this call
/// runs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_error_message(err: Option<&Error>) -> *const c_char {
    match err {
        Some(err) => err.message.as_ptr(),
        None => std::ptr::null(),
    }
}

/// Frees `err` and writes NULL to it, so calling this again on the same pointer is a no-op
/// instead of a double free.
///
/// # Safety
/// `err` must be NULL, or point to a writable `TkError *` holding NULL or an error returned by a
/// `tk_*` call that is not yet freed and that no other thread is using.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_error_free(err: *mut *mut Error) {
    // SAFETY: caller's obligation, documented above.
    unsafe { free_ptr(err) }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".to_string())
}

/// Catches unwind panic and converts it to an error
pub(crate) fn catch_panic<E: std::fmt::Display>(
    body: impl FnOnce() -> Result<(), E>,
) -> *mut Error {
    match panic::catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(())) => std::ptr::null_mut(),
        Ok(Err(err)) => Error::into_ptr(err),
        Err(payload) => Error::into_ptr(panic_message(payload)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message_of(err: *mut Error) -> String {
        let err = unsafe { err.as_ref() }.expect("a non-NULL error");
        err.message.to_str().unwrap().to_string()
    }

    #[test]
    fn ok_body_returns_null() {
        let err = catch_panic(|| -> Result<(), &str> { Ok(()) });
        assert!(err.is_null());
    }

    #[test]
    fn err_body_carries_its_display_message() {
        let err = catch_panic(|| -> Result<(), &str> { Err("bad input") });
        assert_eq!(message_of(err), "bad input");
    }

    #[test]
    fn a_str_panic_payload_is_preserved() {
        // `panic!("literal")` with no format args panics with a `&'static str` payload.
        let err = catch_panic(|| -> Result<(), &str> { panic!("kaboom") });
        assert_eq!(message_of(err), "kaboom");
    }

    #[test]
    fn a_string_panic_payload_is_preserved() {
        // Any format args route the panic payload through `format!`, producing a `String`
        // instead of a `&str`.
        let err = catch_panic(|| -> Result<(), &str> { panic!("{}", "kaboom") });
        assert_eq!(message_of(err), "kaboom");
    }

    #[test]
    fn an_unrecognized_panic_payload_falls_back_to_a_generic_message() {
        let err = catch_panic(|| -> Result<(), &str> {
            panic::panic_any(42_i32);
        });
        assert_eq!(message_of(err), "unknown panic");
    }
}
