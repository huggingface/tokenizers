use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    sync::Mutex,
};

#[cfg(feature = "clap")]
use clap::builder::PossibleValue;

use log::{info, trace};

use super::*;

/// The sharding method to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, MemSize)]
#[mem_size(flat)]
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[cfg_attr(feature = "epserde", epserde(deep_copy))]
pub enum Sharding {
    /// Process all hashes as a single Vec in memory.
    #[default]
    None,
    /// Repeatedly hash all elements, and each time only process a chunk of 2^31 of them.
    Memory,
    /// Hash everything once, write shards of up to 2^31 hashes to disk.
    Disk,
    /// Hybrid that repeatedly fills the given amount (in bytes) of disk space with hashes.
    Hybrid(usize),
}

#[cfg(feature = "clap")]
impl clap::ValueEnum for Sharding {
    fn value_variants<'a>() -> &'a [Self] {
        // 128 GiB for Hybrid & usize::MAX for 32-bit systems.
        #[cfg(target_pointer_width = "64")]
        const HYBRID_BYTES: usize = 1 << 37;
        #[cfg(not(target_pointer_width = "64"))]
        const HYBRID_BYTES: usize = usize::MAX;
        &[
            Sharding::None,
            Sharding::Memory,
            Sharding::Disk,
            Sharding::Hybrid(HYBRID_BYTES),
        ]
    }
    fn to_possible_value<'a>(&self) -> Option<PossibleValue> {
        Some(match self {
            Sharding::None => PossibleValue::new("none"),
            Sharding::Memory => PossibleValue::new("memory"),
            Sharding::Disk => PossibleValue::new("disk"),
            Sharding::Hybrid(_) => PossibleValue::new("hybrid"),
        })
    }
}

impl<
        Key: KeyT + ?Sized,
        BF: BucketFn,
        F: Packed,
        Hx: KeyHasher<Key>,
        const SINGLE_PART: bool,
        const REMAP: bool,
    > PtrHash<Key, BF, F, Hx, Vec<u8>, SINGLE_PART, REMAP>
{
    /// Return an iterator over the Vec of hashes of each shard.
    pub(crate) fn shards<'a>(
        &'a self,
        keys: impl ParallelIterator<Item = impl Borrow<Key>> + Clone + 'a,
    ) -> Box<dyn Iterator<Item = Vec<Hx::H>> + 'a> {
        match self.params.sharding {
            Sharding::None => self.no_sharding(keys.clone()),
            Sharding::Memory => self.shard_keys_in_memory(keys.clone()),
            Sharding::Disk => self.shard_keys_hybrid(usize::MAX, keys.clone()),
            Sharding::Hybrid(mem) => self.shard_keys_hybrid(mem, keys.clone()),
        }
    }

    /// Collect all hashes to a Vec directly and return it.
    fn no_sharding<'a>(
        &'a self,
        keys: impl ParallelIterator<Item = impl Borrow<Key>> + Clone + 'a,
    ) -> Box<dyn Iterator<Item = Vec<Hx::H>> + 'a> {
        trace!("No sharding: collecting all {} hashes in memory.", self.n);
        let start = std::time::Instant::now();
        let hashes = keys.map(|key| self.hash_key(key.borrow())).collect();
        log_duration("collect hash", start);
        Box::new(std::iter::once(hashes))
    }

    /// Loop over the keys once per shard.
    /// Return an iterator over shards.
    /// For each shard, a filtered copy of the ParallelIterator is returned.
    fn shard_keys_in_memory<'a>(
        &'a self,
        keys: impl ParallelIterator<Item = impl Borrow<Key>> + Clone + 'a,
    ) -> Box<dyn Iterator<Item = Vec<Hx::H>> + 'a> {
        trace!(
            "In-memory sharding: iterate keys once for each of {} shards, each of ~{} keys.",
            self.shards,
            self.n / self.shards
        );
        let it = (0..self.shards).map(move |shard| {
            trace!("Shard {shard:>3}/{:3}\r", self.shards);
            let start = std::time::Instant::now();
            let hashes: Vec<_> = keys
                .clone()
                .map(|key| self.hash_key(key.borrow()))
                .filter(move |h| self.shard(*h) == shard)
                .collect();
            trace!("Shard {shard:>3}/{:3}: {} keys", self.shards, hashes.len());
            log_duration("collect shrd", start);
            hashes
        });
        Box::new(it)
    }

    /// Loop over the keys and write each keys hash to the corresponding shard.
    /// Returns an iterator over shards.
    /// Files are written to /tmp by default, but this can be changed using the
    /// TMPDIR environment variable.
    ///
    /// This is based on `SigStore` in `sux-rs`, but simplified for the specific use case here.
    /// https://github.com/vigna/sux-rs/blob/main/src/utils/sig_store.rs
    fn shard_keys_hybrid<'a>(
        &'a self,
        mem: usize,
        keys: impl ParallelIterator<Item = impl Borrow<Key>> + Clone + 'a,
    ) -> Box<dyn Iterator<Item = Vec<Hx::H>> + 'a> {
        let total_shards = self.shards;
        let keys_per_shard = self.n / total_shards;
        let shards_on_disk = mem / std::mem::size_of::<Hx::H>() / keys_per_shard;
        assert!(
            shards_on_disk > 0,
            "Each shard takes more than the provided memory."
        );
        if mem < usize::MAX {
            info!("Hybrid sharding: writing hashes to disk for {shards_on_disk} shards at a time, for total {} shards each of ~{} keys.", self.shards, self.n / self.shards);
        } else {
            info!(
                "On-disk sharding: writing hashes to disk for all {} shards at a time, each of ~{} keys.",
                self.shards, self.n / self.shards

            );
        }

        let it = (0..self.shards)
            .step_by(shards_on_disk)
            .flat_map(move |first_shard| {
                let temp_dir = tempfile::TempDir::new().unwrap();
                info!("TMP PATH: {:?}", temp_dir.path());

                let shard_range = first_shard..(first_shard + shards_on_disk).min(self.shards);
                info!("Writing keys for shards {shard_range:?}/{}", self.shards);

                let start = std::time::Instant::now();

                // Create a file writer and count for each shard.
                let writers = shard_range
                    .clone()
                    .map(|shard| {
                        Mutex::new((
                            BufWriter::new(
                                File::options()
                                    .read(true)
                                    .write(true)
                                    .create(true)
                                    .open(temp_dir.path().join(format!("{}.tmp", shard)))
                                    .unwrap(),
                            ),
                            0,
                        ))
                    })
                    .collect_vec();

                // Each thread has a local buffer per shard.
                let init = || writers.iter().map(ThreadLocalBuf::new).collect_vec();
                // Iterate over keys.
                keys.clone()
                    .map(|key| self.hash_key(key.borrow()))
                    .for_each_init(init, |bufs, h| {
                        let shard = self.shard(h);
                        if shard_range.contains(&shard) {
                            bufs[shard - shard_range.start].push(h);
                        }
                    });
                let start = log_duration("Writing files", start);

                // Flush writers and convert to files.
                let files = writers
                    .into_iter()
                    .map(|w| {
                        let (mut w, cnt) = w.into_inner().unwrap();
                        w.flush().unwrap();
                        let mut file = w.into_inner().unwrap();
                        file.seek(SeekFrom::Start(0)).unwrap();
                        (file, cnt)
                    })
                    .collect_vec();
                log_duration("Flushing writers", start);

                files
                    .into_iter()
                    .zip(shard_range)
                    .map(move |((f, cnt), _shard)| {
                        let start = std::time::Instant::now();
                        let mut v = vec![Hx::H::default(); cnt];
                        let mut reader = BufReader::new(f);
                        let (pre, data, post) = unsafe { v.align_to_mut::<u8>() };
                        assert!(pre.is_empty());
                        assert!(post.is_empty());
                        Read::read_exact(&mut reader, data).unwrap();
                        log_duration("Read shard", start);
                        v
                    })

                // Files are cleaned up automatically when tmpdir goes out of scope.
            });
        Box::new(it)
    }
}

struct ThreadLocalBuf<'a, H> {
    buf: Vec<H>,
    file: &'a Mutex<(BufWriter<File>, usize)>,
}

impl<'a, H> ThreadLocalBuf<'a, H> {
    fn new(file: &'a Mutex<(BufWriter<File>, usize)>) -> Self {
        Self {
            // buffer 1GB of data at a time.
            buf: Vec::with_capacity(1 << 28),
            file,
        }
    }
    fn push(&mut self, h: H) {
        self.buf.push(h);
        if self.buf.len() == self.buf.capacity() {
            self.flush();
        }
    }
    fn flush(&mut self) {
        let mut file = self.file.lock().unwrap();
        let (pre, bytes, post) = unsafe { self.buf.align_to::<u8>() };
        assert!(pre.is_empty());
        assert!(post.is_empty());
        file.0.write_all(bytes).unwrap();
        file.1 += self.buf.len();
        self.buf.clear();
    }
}

impl<'a, H> Drop for ThreadLocalBuf<'a, H> {
    fn drop(&mut self) {
        self.flush();
    }
}
