use std::ffi::c_char;

use crate::utils::{RustOwned, free_ptr};

/// Text that Rust owns and C reads. Stored with a trailing NUL so that the pointer C gets works
/// as a C string; [`String::len`] leaves the NUL out.
pub struct String(std::string::String);
impl RustOwned for String {}

impl String {
    pub(crate) fn new(mut text: std::string::String) -> Self {
        text.push('\0');
        String(text)
    }

    fn as_ptr(&self) -> *const c_char {
        self.0.as_ptr().cast()
    }

    fn len(&self) -> usize {
        self.0.len() - 1
    }
}

/// Frees `string` and writes NULL to it, so calling this again on the same pointer is a no-op
/// instead of a double free.
///
/// # Safety
/// `string` must be NULL, or point to a writable `TkString *` holding NULL or a string that is
/// not yet freed and that no other thread is using.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_string_free(string: *mut *mut String) {
    // SAFETY: caller's obligation, documented above.
    unsafe { free_ptr(string) }
}

/// Returns `string`'s text as a NUL-terminated C string, or NULL if `string` is NULL. The text is
/// UTF-8 (see `tk_tokenizer_decode` for what happens to bytes a model emits that aren't). It
/// points into `string` and is valid until `tk_string_free`.
///
/// # Safety
/// `string` must be NULL, or a string that is not freed while this call runs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_string_cstr(string: Option<&String>) -> *const c_char {
    match string {
        Some(string) => string.as_ptr(),
        None => std::ptr::null(),
    }
}

/// Returns the number of bytes in `string`'s text, not counting the NUL that follows them, or 0
/// if `string` is NULL. Differs from `strlen(tk_string_cstr(string))` only when the text itself
/// contains a NUL.
///
/// # Safety
/// `string` must be NULL, or a string that is not freed while this call runs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tk_string_len(string: Option<&String>) -> usize {
    string.map_or(0, String::len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    #[test]
    fn cstr_is_the_text_followed_by_a_nul() {
        let string = String::new("héllo".to_string());
        let cstr = unsafe { CStr::from_ptr(string.as_ptr()) };
        assert_eq!(cstr.to_str().unwrap(), "héllo");
        assert_eq!(string.len(), "héllo".len());
    }

    #[test]
    fn empty_text_is_just_the_nul() {
        let string = String::new(std::string::String::new());
        assert_eq!(string.len(), 0);
        assert_eq!(string.0.as_bytes(), b"\0");
    }

    #[test]
    fn len_counts_past_an_interior_nul() {
        let string = String::new("a\0b".to_string());
        assert_eq!(string.len(), 3);
        let cstr = unsafe { CStr::from_ptr(string.as_ptr()) };
        assert_eq!(cstr.to_bytes(), b"a");
    }

    #[test]
    fn accessors_on_null_return_null_and_zero() {
        assert!(unsafe { tk_string_cstr(None) }.is_null());
        assert_eq!(unsafe { tk_string_len(None) }, 0);
    }
}
