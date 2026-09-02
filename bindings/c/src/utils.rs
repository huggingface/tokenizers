use std::{
    ffi::{CStr, c_char},
    str::Utf8Error,
};

use crate::error::{Error, catch_panic};

/// Marks a type whose instances Rust heap-allocates and frees, and hands to C as an opaque
/// pointer.
///
/// C may use one pointer from several threads at once, and every `tk_*` fn borrows it as a
/// `&T`, so the type has to be `Send + Sync` for that to be sound.
pub(crate) trait RustOwned: Send + Sync {}

/// Moves `value` to the heap and returns the pointer C will hold, pairing with [`free_ptr`].
pub(crate) fn new_ptr<T: RustOwned>(value: T) -> *mut T {
    Box::into_raw(Box::new(value))
}

/// Frees the pointer in `slot` and writes NULL to it, so calling this again on the same slot is
/// a no-op instead of a double free.
///
/// # Safety
/// `slot` must be NULL, or point to a writable `*mut T` holding NULL or a pointer from
/// [`new_ptr`] that is not yet freed and that nothing else is using.
pub(crate) unsafe fn free_ptr<T: RustOwned>(slot: *mut *mut T) {
    // SAFETY: caller's obligation, documented above.
    let Some(slot) = (unsafe { slot.as_mut() }) else {
        return;
    };
    if slot.is_null() {
        return;
    }
    // SAFETY: just checked non-NULL; liveness is the caller's obligation, documented above.
    drop(unsafe { Box::from_raw(*slot) });
    *slot = std::ptr::null_mut();
}

/// A wrapper for FFI functions that output a pointer to a new Rust-owned value.
///
/// Takes care of:
/// - reporting a [`Error`] instead of writing anything, if `out` is NULL
/// - initializing the out pointer to NULL
/// - catching panic unwinds so they don't reach C code
/// - moving the output of `inner` to the heap and writing its pointer to `out`, when it succeeds
///
/// # Safety
/// `out` must be NULL, or a writable pointer to a `*mut T`, initialized or not.
pub(crate) unsafe fn wrap_in_ptr<T: RustOwned, E: std::fmt::Display>(
    out: *mut *mut T,
    inner: impl FnOnce() -> Result<T, E>,
) -> *mut Error {
    if out.is_null() {
        return Error::into_ptr("out pointer must not be NULL");
    }
    // SAFETY: caller's obligation, documented above.
    unsafe { out.write(std::ptr::null_mut()) };
    catch_panic(move || -> Result<(), E> {
        let value = inner()?;
        // SAFETY: caller's obligation, documented above.
        unsafe { out.write(new_ptr(value)) };
        Ok(())
    })
}

/// A borrowed view into a Rust-owned slice.
/// Bundles a pointer with the slice length.
/// Valid only as long as whatever it points to is still alive.
#[repr(C)]
pub(crate) struct Slice<T> {
    ptr: *const T,
    len: usize,
}

impl<T> Slice<T> {
    pub(crate) fn null() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
        }
    }

    /// Borrows this slice's data as a `&[T]`, without copying. A NULL `ptr` is treated as an
    /// empty slice regardless of `len`, matching [`Slice::null`].
    ///
    /// # Safety
    /// `ptr` must be NULL, or point to `len` readable `T`s that are not modified for as long as
    /// the returned slice is used.
    pub(crate) unsafe fn as_slice<'a>(&self) -> &'a [T] {
        if self.ptr.is_null() {
            return &[];
        }
        // SAFETY: caller's obligation, documented above.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T> From<&[T]> for Slice<T> {
    fn from(value: &[T]) -> Self {
        Self {
            ptr: value.as_ptr(),
            len: value.len(),
        }
    }
}

/// A wrapper for FFI functions that output a [`Slice`].
///
/// Takes care of:
/// - reporting a [`Error`] instead of writing anything, if `out` is NULL
/// - initializing the out pointer to an empty slice
/// - catching panic unwinds so they don't reach C code
/// - writing `inner`'s slice to the out pointer, when it succeeds
///
/// A missing value (e.g. an encoding with no type ids) is a `body` that returns `Ok(&[])`: a
/// [`Slice`] can't tell "absent" from "empty" apart, and nothing needs it to.
///
/// # Safety
/// `out` must be NULL, or a writable pointer to a [`Slice<T>`], initialized or not.
pub(crate) unsafe fn wrap_in_slice<'a, T: 'a, E: std::fmt::Display>(
    out: *mut Slice<T>,
    inner: impl FnOnce() -> Result<&'a [T], E>,
) -> *mut Error {
    if out.is_null() {
        return Error::into_ptr("out pointer must not be NULL");
    }
    // SAFETY: caller's obligation, documented above.
    unsafe { out.write(Slice::null()) };
    catch_panic(move || -> Result<(), E> {
        let slice = inner()?;
        // SAFETY: caller's obligation, documented above.
        unsafe { out.write(Slice::from(slice)) };
        Ok(())
    })
}

/// Borrows a NUL-terminated C string as a UTF-8 `&str`, without copying.
///
/// # Safety
/// `c_str` must be non-NULL and point to a NUL-terminated string that is neither freed nor
/// modified for as long as the returned `&str` is used.
pub(crate) unsafe fn convert_c_str<'a>(c_str: *const c_char) -> Result<&'a str, Utf8Error> {
    // SAFETY: caller's obligation, documented above.
    let c_str = unsafe { CStr::from_ptr(c_str) };
    c_str.to_str()
}

/// Borrows a `len`-byte buffer as a UTF-8 `&str`, without copying. Unlike [`convert_c_str`],
/// `buf` doesn't need a NUL terminator and may contain embedded NUL bytes.
///
/// # Safety
/// `buf` must be non-NULL and point to `len` readable bytes that are neither freed nor modified
/// for as long as the returned `&str` is used.
pub(crate) unsafe fn convert_c_buf<'a>(
    buf: *const c_char,
    len: usize,
) -> Result<&'a str, Utf8Error> {
    // SAFETY: caller's obligation, documented above.
    let bytes = unsafe { std::slice::from_raw_parts(buf as *const u8, len) };
    std::str::from_utf8(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    struct Dummy;
    impl RustOwned for Dummy {}

    #[test]
    fn slice_as_slice_borrows_the_underlying_data() {
        let data = [1u32, 2, 3];
        let slice = Slice::from(&data[..]);
        assert_eq!(unsafe { slice.as_slice() }, &data);
    }

    #[test]
    fn slice_as_slice_null_ptr_is_empty_even_with_nonzero_len() {
        let slice = Slice::<u32> {
            ptr: std::ptr::null(),
            len: 5,
        };
        assert_eq!(unsafe { slice.as_slice() }, &[] as &[u32]);
    }

    #[test]
    fn convert_c_str_accepts_valid_utf8() {
        let bytes = b"hello\0";
        let result = unsafe { convert_c_str(bytes.as_ptr() as *const c_char) };
        assert_eq!(result.unwrap(), "hello");
    }

    #[test]
    fn convert_c_str_rejects_invalid_utf8() {
        let bytes = b"\xff\xfe\0";
        let result = unsafe { convert_c_str(bytes.as_ptr() as *const c_char) };
        assert!(result.is_err());
    }

    #[test]
    fn convert_c_buf_accepts_embedded_nul_bytes() {
        // Unlike convert_c_str, convert_c_buf takes an explicit length and tolerates NULs
        // embedded in the middle of the buffer.
        let bytes = b"a\0b";
        let result = unsafe { convert_c_buf(bytes.as_ptr() as *const c_char, bytes.len()) };
        assert_eq!(result.unwrap(), "a\0b");
    }

    #[test]
    fn convert_c_buf_rejects_invalid_utf8() {
        let bytes: [u8; 2] = [0xff, 0xfe];
        let result = unsafe { convert_c_buf(bytes.as_ptr() as *const c_char, bytes.len()) };
        assert!(result.is_err());
    }

    #[test]
    fn free_ptr_outer_null_is_a_no_op() {
        // Doesn't crash -- that's the whole test.
        unsafe { free_ptr::<Dummy>(std::ptr::null_mut()) };
    }

    #[test]
    fn free_ptr_frees_and_nulls_the_slot() {
        let mut slot = new_ptr(Dummy);
        assert!(!slot.is_null());
        unsafe { free_ptr(&mut slot) };
        assert!(slot.is_null());
    }

    #[test]
    fn free_ptr_is_a_no_op_the_second_time_on_the_same_slot() {
        let mut slot = new_ptr(Dummy);
        unsafe { free_ptr(&mut slot) };
        unsafe { free_ptr(&mut slot) }; // must not double-free
        assert!(slot.is_null());
    }

    #[test]
    fn wrap_in_ptr_out_null_reports_an_error_without_calling_inner() {
        let called = Cell::new(false);
        let result = unsafe {
            wrap_in_ptr::<Dummy, &str>(std::ptr::null_mut(), || {
                called.set(true);
                Ok(Dummy)
            })
        };
        assert!(!result.is_null());
        assert!(!called.get(), "inner must not run when out is NULL");
    }

    #[test]
    fn wrap_in_ptr_resets_out_to_null_even_when_inner_fails() {
        let mut out: *mut Dummy = std::ptr::dangling_mut(); // poison: some non-null value
        let result = unsafe { wrap_in_ptr(&mut out, || -> Result<Dummy, &str> { Err("nope") }) };
        assert!(!result.is_null());
        assert!(out.is_null());
    }

    #[test]
    fn wrap_in_slice_out_null_reports_an_error_without_calling_inner() {
        let called = Cell::new(false);
        let result = unsafe {
            wrap_in_slice::<u32, &str>(std::ptr::null_mut(), || {
                called.set(true);
                Ok(&[][..])
            })
        };
        assert!(!result.is_null());
        assert!(!called.get(), "inner must not run when out is NULL");
    }

    #[test]
    fn wrap_in_slice_resets_out_to_empty_even_when_inner_fails() {
        let mut out = Slice::<u32> {
            ptr: std::ptr::dangling(),
            len: 99,
        }; // poison
        let result = unsafe { wrap_in_slice(&mut out, || -> Result<&[u32], &str> { Err("nope") }) };
        assert!(!result.is_null());
        assert!(out.ptr.is_null());
        assert_eq!(out.len, 0);
    }

    #[test]
    fn wrap_in_slice_writes_the_slice_on_success() {
        let data = [1u32, 2, 3];
        let mut out = Slice::<u32>::null();
        let result =
            unsafe { wrap_in_slice(&mut out, || -> Result<&[u32], &str> { Ok(&data[..]) }) };
        assert!(result.is_null());
        assert_eq!(
            unsafe { std::slice::from_raw_parts(out.ptr, out.len) },
            &data
        );
    }
}
