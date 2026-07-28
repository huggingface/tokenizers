use ahash::AHashMap;

const MAX_LENGTH: usize = 1024;

pub struct WordCache {
    lookup: AHashMap<Box<[u8]>, Box<[u32]>>,
    capacity: usize,
}

impl WordCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            lookup: AHashMap::with_capacity(capacity),
        }
    }

    pub fn get(&self, key: &[u8]) -> Option<&[u32]> {
        if key.len() > MAX_LENGTH {
            return None;
        }
        self.lookup.get(key).map(|boxed| &boxed[..])
    }

    pub fn insert(&mut self, key: &[u8], ids: impl ExactSizeIterator<Item = u32>) {
        if key.len() > MAX_LENGTH {
            return;
        }
        if self.lookup.len() >= self.capacity {
            self.evict();
        }
        self.lookup.insert(Box::from(key), ids.collect());
    }

    fn evict(&mut self) {
        self.lookup.extract_if(|_, _| true).next();
    }
}
