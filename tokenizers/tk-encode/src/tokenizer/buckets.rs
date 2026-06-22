#[derive(Clone, Hash)]
pub struct Bucket {
    pub prefix: [u8; 4],
    pub prefix_len: u8,
    pub start: u32,
    pub end: u32,
}

#[derive(Clone)]
pub enum Buckets {
    Inline { buf: [Bucket; 4], len: u8 },
    Heap(Vec<Bucket>),
}

impl std::ops::Deref for Buckets {
    type Target = [Bucket];

    fn deref(&self) -> &Self::Target {
        match self {
            Buckets::Inline { buf, len } => &buf[..*len as usize],
            Buckets::Heap(vec) => vec.as_slice(),
        }
    }
}
