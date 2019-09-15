use std::cell::UnsafeCell;

/// `UnsafeSync` is a struct that allows to pass any data trough threads.
/// Of course this data is not thread safe and doesn't prevent data race, so it's up to you use it in a safe way.
#[derive(Debug)]
pub struct UnsafeSync<T>(UnsafeCell<T>);

impl<T> UnsafeSync<T> {
    /// Wrap the data to make it sharable trough threads.
    pub fn new(data: T) -> Self {
        UnsafeSync(UnsafeCell::new(data))
    }

    /// Returns a reference of your data
    pub fn get(&self) -> &T{
        unsafe {
            &*self.0.get()
        }
    }

    /// Returns a mutable reference of your data, make sure to use this safelly.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn get_mut(&self) -> &mut T{
        &mut *self.0.get()
    }

    /// Unwrap your data.
    pub fn into_inner(self) -> T {
        self.0.into_inner()
    }
}

unsafe impl<T> std::marker::Sync for UnsafeSync<T>{}