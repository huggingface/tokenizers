use std::ops::{Add, Index, IndexMut, Sub};

// Ord so we can sort them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct BucketIdx(pub u32);

impl Add<usize> for BucketIdx {
    type Output = Self;

    fn add(self, rhs: usize) -> Self::Output {
        Self(self.0 + rhs as u32)
    }
}

impl Sub<usize> for BucketIdx {
    type Output = Self;

    fn sub(self, rhs: usize) -> Self::Output {
        Self(self.0 - rhs as u32)
    }
}

impl BucketIdx {
    pub const NONE: BucketIdx = BucketIdx(u32::MAX);
    pub fn range(num_buckets: usize) -> impl Iterator<Item = Self> + Clone {
        (0..num_buckets as u32).map(Self)
    }
    pub fn is_some(&self) -> bool {
        self.0 != u32::MAX
    }
    pub fn is_none(&self) -> bool {
        self.0 == u32::MAX
    }
}

impl<T> Index<BucketIdx> for [T] {
    type Output = T;

    fn index(&self, index: BucketIdx) -> &Self::Output {
        unsafe { self.get_unchecked(index.0 as usize) }
    }
}

impl<T> IndexMut<BucketIdx> for [T] {
    fn index_mut(&mut self, index: BucketIdx) -> &mut Self::Output {
        unsafe { self.get_unchecked_mut(index.0 as usize) }
    }
}

impl<T> Index<BucketIdx> for Vec<T> {
    type Output = T;

    fn index(&self, index: BucketIdx) -> &Self::Output {
        unsafe { self.get_unchecked(index.0 as usize) }
    }
}

impl<T> IndexMut<BucketIdx> for Vec<T> {
    fn index_mut(&mut self, index: BucketIdx) -> &mut Self::Output {
        unsafe { self.get_unchecked_mut(index.0 as usize) }
    }
}
