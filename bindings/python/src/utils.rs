/// A Container type
///
/// Provides an interface to allow transfer of ownership between Python and Rust.
/// It either contains a Box with full ownership of the content, or a pointer to the content.
///
/// The main goal here is to allow Python calling into Rust to initialize some objects. Later
/// these objects may need to be used by Rust who will expect to take ownership. Since Python
/// does not allow any sort of ownership transfer, it will keep a reference to this object
/// until it gets cleaned up by the GC. In this case, we actually give the ownership to Rust,
/// and just keep a pointer in the Python object.
pub enum Container<T: ?Sized> {
    Owned(Box<T>),
    Pointer(*mut T),
}

impl<T> Container<T>
where
    T: ?Sized,
{
    pub fn from_ref(reference: &Box<T>) -> Self {
        let content: *const T = &**reference;
        Container::Pointer(content as *mut _)
    }

    /// Consumes ourself and return the Boxed element if we have the ownership, None otherwise.
    pub fn take(self) -> Option<Box<T>> {
        match self {
            Container::Owned(obj) => Some(obj),
            Container::Pointer(_) => None,
        }
    }

    /// Return the owned T, keeping a Pointer to it if we currently own it. None otherwise
    pub fn to_pointer(&mut self) -> Option<Box<T>> {
        if let Container::Owned(_) = self {
            unsafe {
                let old_container = std::ptr::read(self);
                let ptr = Box::into_raw(old_container.take().unwrap());
                let new_container = Container::Pointer(ptr);
                std::ptr::write(self, new_container);

                Some(Box::from_raw(ptr))
            }
        } else {
            None
        }
    }

    pub fn execute<F, U>(&self, closure: F) -> U
    where
        F: FnOnce(&Box<T>) -> U,
    {
        match self {
            Container::Owned(val) => closure(val),
            Container::Pointer(ptr) => unsafe {
                let val = Box::from_raw(*ptr);
                let res = closure(&val);
                // We call this to make sure we don't drop the Box
                Box::into_raw(val);
                res
            },
        }
    }
}
