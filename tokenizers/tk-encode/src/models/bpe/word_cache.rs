use ahash::AHashMap;


/// naive implem of word -> IDs cache
pub struct WordCache {
    capacity: usize,
    lookup: AHashMap<String, Box<[u32]>>,
}

impl WordCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            lookup: AHashMap::with_capacity(capacity),
        }
    }

    pub fn get(&self, key: &str) -> Option<&[u32]> {
        self.lookup.get(key).map(|bx| &bx[..])
    }

    pub fn insert(&mut self, k: String, v: Vec<u32>) {
        if self.lookup.len() >= self.capacity {
            // Pop an arbitrary entry
            self.lookup.extract_if(|_, _| true).next();
        }
        self.lookup.insert(k, v.into_boxed_slice());
    }
}
