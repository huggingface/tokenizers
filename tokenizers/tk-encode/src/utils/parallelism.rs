//!
//! This module defines helpers to allow optional Rayon usage.
//!

use rayon::iter::IterBridge;
use rayon::prelude::*;
use rayon_cond::CondIterator;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::OnceLock;
use std::sync::TryLockError;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

// Re-export rayon current_num_threads
pub use rayon::current_num_threads;

/// TODO: deprecate
pub const ENV_VARIABLE: &str = "TOKENIZERS_PARALLELISM";

/// TODO: deprecate
static USED_PARALLELISM: AtomicBool = AtomicBool::new(false);

/// TODO: deprecate
static PARALLELISM: AtomicU8 = AtomicU8::new(0);

/// 0 means deafult value
static NUM_THREADS: AtomicUsize = AtomicUsize::new(0);
/// Counter to track the current version of the pool
/// After forking or changing the number of threads, we need to invalidate and recreate a new pool
/// Old pools will be dropped when they go out of scope (arc refcount goes to 0)
static POOL_GEN: AtomicUsize = AtomicUsize::new(0);

/// register an invalidation callback to be called after a fork with pthread_atfork
/// this is required because when forking only the parent thread is copied to the child process so
/// you lose access to the previously built thread pool -> rebuild needed
/// cf the POSIX spec: https://pubs.opengroup.org/onlinepubs/9699919799/functions/fork.html
#[cfg(unix)]
fn register_fork_handler() {
    static REGISTERED: AtomicBool = AtomicBool::new(false);
    if REGISTERED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
    {
        unsafe extern "C" fn child_after_fork() {
            POOL_GEN.fetch_add(1, Ordering::SeqCst);
        }
        unsafe {
            let _ = libc::pthread_atfork(None, None, Some(child_after_fork));
        }
    }
}

#[cfg(not(unix))]
fn register_fork_handler() {}

#[derive(Clone)]
struct Slot {
    pool: Arc<rayon::ThreadPool>,
    version: usize,
    pid: u32,
}

static CELL: Mutex<Option<Slot>> = Mutex::new(None);

type MaybeLockGuard = Option<MutexGuard<'static, Option<Slot>>>;

fn lock() -> MaybeLockGuard {
    match CELL.try_lock() {
        Ok(g) => Some(g),
        Err(TryLockError::Poisoned(p)) => Some(p.into_inner()),
        Err(TryLockError::WouldBlock) => None,
    }
}

pub(crate) fn pool() -> Option<Arc<rayon::ThreadPool>> {
    register_fork_handler();

    let generation = POOL_GEN.load(Ordering::Acquire);
    if let Some(guard) = lock()
        && let Some(slot) = guard.as_ref()
        && generation == slot.version
    {
        return Some(slot.pool.clone());
    }

    let num_threads = num_threads();
    // We don't create a thread pool when thread == 1
    let slot = if num_threads == 1 {
        None
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|i| format!("tk-encode-{i}"))
            .build()
            .ok()?;
        let slot = Slot {
            pool: Arc::new(pool),
            version: generation,
            pid: std::process::id(),
        };
        Some(slot)
    };

    let old = lock().and_then(|mut guard| match &slot {
        Some(slot) => guard.replace(slot.clone()),
        None => guard.take(),
    });

    if let Some(old) = old
        && old.pid != std::process::id()
    {
        // mem::forget is here to avoid deadlocking on the pool drop after forking
        std::mem::forget(old.pool);
    }

    slot.map(|slot| slot.pool.clone())
}

pub fn num_threads() -> usize {
    match NUM_THREADS.load(Ordering::Acquire) {
        // 0 == default value
        0 => default_num_threads(),
        n => n,
    }
}

/// The default pool size: **physical** cores, not logical ones.
///
/// `available_parallelism()` counts SMT siblings, and filling them costs throughput
/// here. Encoding medium documents (~8 KiB) peaks at the physical core count and
/// then falls off:
///
/// | machine | at physical | at logical |
/// |---|---|---|
/// | aarch64, 88 physical | 6815 MiB/s @88t | 5462 MiB/s @176t (-20%) |
/// | Granite Rapids, 2 sockets | 1314 MiB/s @128t | 698 MiB/s @512t (-47%) |
///
/// Two workers on one physical core share a front-end and an L1, and this encode
/// path is branchy and L1i-hungry (a PGO build cuts L1i misses ~26%), so it feels
/// that sharing more than most workloads do.
///
/// Capped by `available_parallelism()` so a cgroup quota or an affinity mask still
/// wins — this only ever lowers the thread count, never raises it. Callers who want
/// the old behaviour can still ask for it with [`set_num_threads`].
fn default_num_threads() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        let logical = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        physical_cores().map_or(logical, |p| p.clamp(1, logical))
    })
}

/// Physical cores, or `None` when the platform will not say.
///
/// `None` is a normal answer, not a failure: the caller falls back to the logical
/// count, which is what this code did before.
#[cfg(target_os = "linux")]
fn physical_cores() -> Option<usize> {
    let logical = std::thread::available_parallelism().ok()?.get();

    // The direct answer, on kernels that expose it. Largely x86: it needs
    // CONFIG_HOTPLUG_SMT, which most arm64 configs do not set — hence the fallback.
    let per_core = std::fs::read_to_string("/sys/devices/system/cpu/smt/num_threads_per_core")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
        .or_else(siblings_per_core)?;

    Some(logical / per_core.max(1))
}

/// How many logical CPUs share cpu0's physical core, read from its sibling list.
#[cfg(target_os = "linux")]
fn siblings_per_core() -> Option<usize> {
    let raw = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list")
        .ok()?;
    parse_siblings_list(&raw)
}

/// Count the CPUs in a `thread_siblings_list`.
///
/// The kernel writes either a comma-separated list (`"0,88"`) or a range
/// (`"0-1"`), and mixes them (`"0-1,4-5"`); all three appear in the wild. Kept
/// free of I/O and compiled everywhere so it can be tested off Linux — a silent
/// misparse here would divide the pool size by the wrong number on every box.
#[cfg(any(target_os = "linux", test))]
fn parse_siblings_list(raw: &str) -> Option<usize> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    let mut total = 0usize;
    for part in raw.split(',') {
        let part = part.trim();
        if part.is_empty() {
            return None;
        }
        total += match part.split_once('-') {
            Some((lo, hi)) => {
                let lo = lo.trim().parse::<usize>().ok()?;
                let hi = hi.trim().parse::<usize>().ok()?;
                hi.checked_sub(lo)?.checked_add(1)?
            }
            None => {
                part.parse::<usize>().ok()?;
                1
            }
        };
    }
    (total > 0).then_some(total)
}

#[cfg(target_os = "macos")]
fn physical_cores() -> Option<usize> {
    let mut out: libc::c_int = 0;
    let mut len = std::mem::size_of::<libc::c_int>();
    // SAFETY: `hw.physicalcpu` is a NUL-terminated name; `out`/`len` describe a
    // correctly sized, initialised c_int that sysctl writes at most `len` bytes into.
    let rc = unsafe {
        libc::sysctlbyname(
            c"hw.physicalcpu".as_ptr(),
            (&raw mut out).cast(),
            &raw mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0 && out > 0).then_some(out as usize)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn physical_cores() -> Option<usize> {
    None
}

fn invalidate() {
    POOL_GEN.fetch_add(1, Ordering::SeqCst);
}

/// Passing in 0 will reset to the default value
pub fn set_num_threads(n: usize) {
    NUM_THREADS.store(n, Ordering::Release);
    invalidate();
}

/// Check if the TOKENIZERS_PARALLELISM env variable has been explicitly set
pub fn is_parallelism_configured() -> bool {
    std::env::var(ENV_VARIABLE).is_ok() || get_override_parallelism().is_some()
}

/// Check if at some point we used a parallel iterator
pub fn has_parallelism_been_used() -> bool {
    USED_PARALLELISM.load(Ordering::SeqCst)
}

/// Get internally set parallelism
fn get_override_parallelism() -> Option<bool> {
    match PARALLELISM.load(Ordering::SeqCst) {
        0 => None,
        1 => Some(false),
        2 => Some(true),
        _ => unreachable!(),
    }
}

/// Get the currently set value for `TOKENIZERS_PARALLELISM` env variable
fn get_env_parallelism() -> bool {
    match std::env::var(ENV_VARIABLE) {
        Ok(mut v) => {
            v.make_ascii_lowercase();
            !matches!(v.as_ref(), "" | "off" | "false" | "f" | "no" | "n" | "0")
        }
        Err(_) => true, // If we couldn't get the variable, we use the default
    }
}

pub fn get_parallelism() -> bool {
    if let Some(parallel) = get_override_parallelism() {
        parallel
    } else {
        get_env_parallelism()
    }
}

/// Set the value for `TOKENIZERS_PARALLELISM` for the current process
pub fn set_parallelism(val: bool) {
    PARALLELISM.store(if val { 2 } else { 1 }, Ordering::SeqCst);
}

/// Allows to convert into an iterator that can be executed either parallelly or serially.
///
/// The choice is made according to the currently set `TOKENIZERS_PARALLELISM` environment variable.
/// This variable can have one of the following values
///   - False => "" (empty value), "false", "f", "off", "no", "n", "0"
///   - True => Any other value
///
pub trait MaybeParallelIterator<P, S>
where
    P: ParallelIterator,
    S: Iterator<Item = P::Item>,
{
    /// Convert ourself in a CondIterator, that will be executed either in parallel or serially,
    /// based solely on the `TOKENIZERS_PARALLELISM` environment variable
    fn into_maybe_par_iter(self) -> CondIterator<P, S>;
    /// Convert ourself in a CondIterator, that will be executed either in parallel or serially,
    /// based on both the `TOKENIZERS_PARALLELISM` environment variable and the provided bool.
    /// Both must be true to run with parallelism activated.
    fn into_maybe_par_iter_cond(self, cond: bool) -> CondIterator<P, S>;
}

impl<P, S, I> MaybeParallelIterator<P, S> for I
where
    I: IntoParallelIterator<Iter = P, Item = P::Item> + IntoIterator<IntoIter = S, Item = S::Item>,
    P: ParallelIterator,
    S: Iterator<Item = P::Item>,
{
    fn into_maybe_par_iter(self) -> CondIterator<P, S> {
        let parallelism = get_parallelism();
        if parallelism {
            USED_PARALLELISM.store(true, Ordering::SeqCst);
        }
        CondIterator::new(self, parallelism)
    }

    fn into_maybe_par_iter_cond(self, cond: bool) -> CondIterator<P, S> {
        if cond {
            self.into_maybe_par_iter()
        } else {
            CondIterator::from_serial(self)
        }
    }
}

/// Shared reference version of MaybeParallelIterator, works the same but returns an iterator
/// over references, does not consume self
pub trait MaybeParallelRefIterator<'data, P, S>
where
    P: ParallelIterator,
    S: Iterator<Item = P::Item>,
    P::Item: 'data,
{
    fn maybe_par_iter(&'data self) -> CondIterator<P, S>;
    fn maybe_par_iter_cond(&'data self, cond: bool) -> CondIterator<P, S>;
}

impl<'data, P, S, I: 'data + ?Sized> MaybeParallelRefIterator<'data, P, S> for I
where
    &'data I: MaybeParallelIterator<P, S>,
    P: ParallelIterator,
    S: Iterator<Item = P::Item>,
    P::Item: 'data,
{
    fn maybe_par_iter(&'data self) -> CondIterator<P, S> {
        self.into_maybe_par_iter()
    }

    fn maybe_par_iter_cond(&'data self, cond: bool) -> CondIterator<P, S> {
        self.into_maybe_par_iter_cond(cond)
    }
}

/// Exclusive reference version of MaybeParallelIterator, works the same but returns an iterator
/// over mutable references, does not consume self
pub trait MaybeParallelRefMutIterator<'data, P, S>
where
    P: ParallelIterator,
    S: Iterator<Item = P::Item>,
    P::Item: 'data,
{
    fn maybe_par_iter_mut(&'data mut self) -> CondIterator<P, S>;
    fn maybe_par_iter_mut_cond(&'data mut self, cond: bool) -> CondIterator<P, S>;
}

impl<'data, P, S, I: 'data + ?Sized> MaybeParallelRefMutIterator<'data, P, S> for I
where
    &'data mut I: MaybeParallelIterator<P, S>,
    P: ParallelIterator,
    S: Iterator<Item = P::Item>,
    P::Item: 'data,
{
    fn maybe_par_iter_mut(&'data mut self) -> CondIterator<P, S> {
        self.into_maybe_par_iter()
    }

    fn maybe_par_iter_mut_cond(&'data mut self, cond: bool) -> CondIterator<P, S> {
        self.into_maybe_par_iter_cond(cond)
    }
}

/// Converts any serial iterator into a CondIterator, that can either run parallelly or serially.
pub trait MaybeParallelBridge<T, S>
where
    S: Iterator<Item = T> + Send,
    T: Send,
{
    fn maybe_par_bridge(self) -> CondIterator<IterBridge<S>, S>;
    fn maybe_par_bridge_cond(self, cond: bool) -> CondIterator<IterBridge<S>, S>;
}

impl<T, S> MaybeParallelBridge<T, S> for S
where
    S: Iterator<Item = T> + Send,
    T: Send,
{
    fn maybe_par_bridge(self) -> CondIterator<IterBridge<S>, S> {
        let iter = CondIterator::from_serial(self);

        if get_parallelism() {
            USED_PARALLELISM.store(true, Ordering::SeqCst);
            CondIterator::from_parallel(iter.into_parallel().right().unwrap())
        } else {
            iter
        }
    }

    fn maybe_par_bridge_cond(self, cond: bool) -> CondIterator<IterBridge<S>, S> {
        if cond {
            self.maybe_par_bridge()
        } else {
            CondIterator::from_serial(self)
        }
    }
}

/// Allows to convert into `chunks` that can be executed either parallelly or serially.
pub trait MaybeParallelSlice<'data, T>
where
    T: Sync,
{
    /// Create a CondIterator, that will be executed either in parallel or serially,
    /// based solely on the `TOKENIZERS_PARALLELISM` environment variable
    fn maybe_par_chunks(
        &'_ self,
        chunk_size: usize,
    ) -> CondIterator<rayon::slice::Chunks<'_, T>, std::slice::Chunks<'_, T>>;
    /// Create a CondIterator, that will be executed either in parallel or serially,
    /// based on both the `TOKENIZERS_PARALLELISM` environment variable and the provided bool.
    /// Both must be true to run with parallelism activated.
    fn maybe_par_chunks_cond(
        &'_ self,
        cond: bool,
        chunk_size: usize,
    ) -> CondIterator<rayon::slice::Chunks<'_, T>, std::slice::Chunks<'_, T>>;
}

impl<T> MaybeParallelSlice<'_, T> for [T]
where
    T: Sync,
{
    fn maybe_par_chunks(
        &'_ self,
        chunk_size: usize,
    ) -> CondIterator<rayon::slice::Chunks<'_, T>, std::slice::Chunks<'_, T>> {
        let parallelism = get_parallelism();
        if parallelism {
            CondIterator::from_parallel(self.par_chunks(chunk_size))
        } else {
            CondIterator::from_serial(self.chunks(chunk_size))
        }
    }
    fn maybe_par_chunks_cond(
        &'_ self,
        cond: bool,
        chunk_size: usize,
    ) -> CondIterator<rayon::slice::Chunks<'_, T>, std::slice::Chunks<'_, T>> {
        if cond {
            self.maybe_par_chunks(chunk_size)
        } else {
            CondIterator::from_serial(self.chunks(chunk_size))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maybe_parallel_iterator() {
        let mut v = vec![1u32, 2, 3, 4, 5, 6];

        assert_eq!(v.maybe_par_iter().sum::<u32>(), 21);
        assert_eq!(
            v.maybe_par_iter_mut()
                .map(|v| {
                    *v *= 2;
                    *v
                })
                .sum::<u32>(),
            42
        );
        assert_eq!(v.maybe_par_iter().sum::<u32>(), 42);
        assert_eq!(v.into_maybe_par_iter().sum::<u32>(), 42);
    }

    #[test]
    fn test_maybe_parallel_slice() {
        let v = [1, 2, 3, 4, 5];

        let chunks: Vec<_> = v.maybe_par_chunks(2).collect();
        assert_eq!(chunks, vec![&[1, 2][..], &[3, 4], &[5]]);
    }

    /// The default must stay inside what we are allowed to use, and must never be
    /// zero — a zero would ask rayon for an unbounded pool.
    #[test]
    fn default_pool_size_never_exceeds_available_parallelism() {
        let logical = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let got = default_num_threads();
        assert!(got >= 1, "pool size must be at least 1, got {got}");
        assert!(
            got <= logical,
            "default {got} exceeds available parallelism {logical}; a cgroup quota \
             or affinity mask must still win"
        );
    }

    /// Whatever the platform reports, it has to be a plausible core count. This is
    /// the guard against a parse going wrong (an empty sibling list, a `0`) and
    /// silently collapsing the pool to one thread on every machine.
    #[test]
    fn physical_core_count_is_plausible() {
        let logical = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        if let Some(p) = physical_cores() {
            assert!(p >= 1, "physical core count must be positive, got {p}");
            assert!(
                p <= logical,
                "physical cores {p} > logical {logical}, which cannot be right"
            );
        }
    }

    /// Both sysfs spellings, and the malformed cases, which must report "no idea"
    /// rather than a wrong divisor.
    #[test]
    fn parses_thread_siblings_lists() {
        // SMT-2, the two shapes the kernel uses.
        assert_eq!(parse_siblings_list("0,88\n"), Some(2));
        assert_eq!(parse_siblings_list("0-1\n"), Some(2));
        // No SMT.
        assert_eq!(parse_siblings_list("0"), Some(1));
        // SMT-4 (POWER-style), and a mixed list.
        assert_eq!(parse_siblings_list("0-3"), Some(4));
        assert_eq!(parse_siblings_list("0-1,4-5"), Some(4));
        // Garbage must not silently become a divisor.
        assert_eq!(parse_siblings_list(""), None);
        assert_eq!(parse_siblings_list("   "), None);
        assert_eq!(parse_siblings_list("abc"), None);
        assert_eq!(parse_siblings_list("0,"), None);
        assert_eq!(parse_siblings_list("3-1"), None);
    }
}
