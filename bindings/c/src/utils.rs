use std::{
    ffi::{CStr, c_char},
    str::Utf8Error,
};

use crate::error::{Error, catch_panic};

/// Marks a type whose instances Rust heap-allocates and frees.
/// Must be wrapped in a [`Handle`] to be passed to C code.
pub(crate) trait RustOwned {}

/// An opaque pointer to a Rust-allocated type.
#[repr(transparent)]
pub(crate) struct Handle<T>(*mut T);

impl<T> Copy for Handle<T> {}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Handle<T> {
    /// A handle representing "no value", ie a null pointer
    pub(crate) fn null() -> Self {
        Handle(std::ptr::null_mut())
    }

    pub(crate) fn is_null(&self) -> bool {
        self.0.is_null()
    }

    /// Borrows the value the handle points at, for as long as the caller asserts it stays valid.
    ///
    /// # Safety
    /// The handle must be non-NULL and point to a `T` that stays live and unmutated for `'a`.
    pub(crate) unsafe fn as_ref<'a>(self) -> &'a T {
        // SAFETY: caller's obligation, documented above.
        unsafe { &*self.0 }
    }
}

/// Leaks `value` onto the heap and returns it as a raw pointer, pairing with [`free_handle`].
pub(crate) fn new_handle<T: RustOwned>(value: T) -> Handle<T> {
    Handle(Box::into_raw(Box::new(value)))
}

/// Frees `handle` and writes NULL to it, so calling this again on the same pointer is a no-op
/// instead of a double free.
///
/// # Safety
/// `handle` must point to a [`Handle<T>`] that is either NULL or a live
/// (not-yet-freed) instance of `T`.
pub(crate) unsafe fn free_handle<T: RustOwned>(handle: *mut Handle<T>) {
    if handle.is_null() {
        return;
    }
    // SAFETY: caller's obligation, documented above.
    let handle = unsafe { &mut *handle };
    if handle.is_null() {
        return;
    }

    // SAFETY: just checked non-NULL; liveness is the caller's obligation, documented above.
    drop(unsafe { Box::from_raw(handle.0) });
    *handle = Handle::null();
}

/// A wrapper for FFI functions that output a [`Handle`].
///
/// Takes care of:
/// - reporting a [`Error`] instead of writing anything, if `out` is NULL
/// - initializing the out pointer to NULL
/// - catching panic unwinds so they don't reach C code
/// - converting the output of `inner` to a [`Handle`] when successful
///
/// # Safety
/// `out` must be NULL, or point to valid, writable memory for a [`Handle<T>`]. It can be
/// uninitialized.
pub(crate) unsafe fn wrap_in_handle<T: RustOwned, E: std::fmt::Display>(
    out: *mut Handle<T>,
    inner: impl FnOnce() -> Result<T, E>,
) -> Handle<Error> {
    if out.is_null() {
        return Error::into_handle("out pointer must not be NULL");
    }
    // SAFETY: caller's obligation, documented above.
    unsafe { std::ptr::write(out, Handle::null()) };
    catch_panic(move || -> Result<(), E> {
        let value = inner()?;
        // SAFETY: caller's obligation, documented above.
        unsafe { std::ptr::write(out, new_handle(value)) };
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
    /// `ptr` must be NULL, or valid for reads of `len` elements of `T` for as long as the
    /// returned slice is used. The pointed-to memory must not be mutated during that time.
    pub(crate) unsafe fn as_slice<'a>(&self) -> &'a [T] {
        if self.ptr.is_null() {
            return &[];
        }
        // SAFETY: caller's obligation, documented above.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

/// Writes a view of `value` to `out`, bundling pointer and slice length in one [`Slice`].
///
/// # Safety
/// `out` must point to valid, writable memory for a [`Slice<T>`]. It can be uninitialized.
pub(crate) unsafe fn write_slice<T>(out: *mut Slice<T>, value: &[T]) {
    let slice = Slice {
        ptr: value.as_ptr(),
        len: value.len(),
    };
    // SAFETY: caller's obligation, documented above.
    unsafe {
        std::ptr::write(out, slice);
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
/// `out` must be NULL, or point to valid, writable memory for a [`Slice<T>`]. It can be
/// uninitialized.
pub(crate) unsafe fn wrap_in_slice<'a, T: 'a, E: std::fmt::Display>(
    out: *mut Slice<T>,
    inner: impl FnOnce() -> Result<&'a [T], E>,
) -> Handle<Error> {
    if out.is_null() {
        return Error::into_handle("out pointer must not be NULL");
    }
    // SAFETY: caller's obligation, documented above.
    unsafe { std::ptr::write(out, Slice::null()) };
    catch_panic(move || -> Result<(), E> {
        let slice = inner()?;
        // SAFETY: caller's obligation, documented above.
        unsafe { write_slice(out, slice) };
        Ok(())
    })
}

/// Borrows a NUL-terminated C string as a UTF-8 `&str`, without copying.
///
/// # Safety
/// `c_str` must be non-NULL, point to a single NUL-terminated byte string, and be valid for
/// reads up to and including that NUL byte for as long as the returned `&str` is alive. The
/// pointed-to memory must not be mutated during that time.
pub(crate) unsafe fn convert_c_str<'a>(c_str: *const c_char) -> Result<&'a str, Utf8Error> {
    // SAFETY: caller's obligation, documented above.
    let c_str = unsafe { CStr::from_ptr(c_str) };
    c_str.to_str()
}

/// Borrows a `len`-byte buffer as a UTF-8 `&str`, without copying. Unlike [`convert_c_str`],
/// `buf` doesn't need a NUL terminator and may contain embedded NUL bytes.
///
/// # Safety
/// `buf` must be non-NULL and valid for reads of `len` bytes for as long as the returned `&str`
/// is alive. The pointed-to memory must not be mutated during that time.
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
        let slice = Slice {
            ptr: data.as_ptr(),
            len: data.len(),
        };
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
    fn free_handle_outer_null_is_a_no_op() {
        // Doesn't crash -- that's the whole test.
        unsafe { free_handle::<Dummy>(std::ptr::null_mut()) };
    }

    #[test]
    fn free_handle_frees_and_nulls_the_slot() {
        let mut handle = new_handle(Dummy);
        assert!(!handle.is_null());
        unsafe { free_handle(&mut handle) };
        assert!(handle.is_null());
    }

    #[test]
    fn free_handle_is_a_no_op_the_second_time_on_the_same_slot() {
        let mut handle = new_handle(Dummy);
        unsafe { free_handle(&mut handle) };
        unsafe { free_handle(&mut handle) }; // must not double-free
        assert!(handle.is_null());
    }

    #[test]
    fn wrap_in_handle_out_null_reports_an_error_without_calling_inner() {
        let called = Cell::new(false);
        let result = unsafe {
            wrap_in_handle::<Dummy, &str>(std::ptr::null_mut(), || {
                called.set(true);
                Ok(Dummy)
            })
        };
        assert!(!result.is_null());
        assert!(!called.get(), "inner must not run when out is NULL");
    }

    #[test]
    fn wrap_in_handle_resets_out_to_null_even_when_inner_fails() {
        let mut out = Handle::<Dummy>(std::ptr::dangling_mut()); // poison: some non-null value
        let result = unsafe { wrap_in_handle(&mut out, || -> Result<Dummy, &str> { Err("nope") }) };
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
