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
use std::sync::atomic::AtomicPtr;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

// Re-export rayon current_num_threads
pub use rayon::current_num_threads;

/// TODO: deprecate
pub(crate) const ENV_VARIABLE: &str = "TOKENIZERS_PARALLELISM";

/// 0 means deafult value
static NUM_THREADS: AtomicUsize = AtomicUsize::new(0);
/// Bumped by [`set_num_threads`], so the next [`pool`] call builds a pool with the new count.
static POOL_GEN: AtomicUsize = AtomicUsize::new(0);

struct CachedPool {
    pool: Arc<rayon::ThreadPool>,
    generation: usize,
}

type PoolCache = Mutex<Option<CachedPool>>;

/// The pool cache until the first `fork()`.
static FIRST_POOL_CACHE: PoolCache = Mutex::new(None);

/// The pool cache [`pool`] reads: [`FIRST_POOL_CACHE`], or a fresh one in a forked child.
///
/// `fork()` copies the whole memory but only the calling thread: the child's copy of the cache may
/// be locked by a thread that doesn't exist there, and its pool has no worker threads (dropping it
/// can deadlock). So the child points this at a fresh cache, and leaks the old one.
static POOL_CACHE: AtomicPtr<PoolCache> = AtomicPtr::new((&raw const FIRST_POOL_CACHE).cast_mut());

fn pool_cache() -> &'static PoolCache {
    // SAFETY: points to `FIRST_POOL_CACHE` or to a leaked `Box`, neither ever freed
    unsafe { &*POOL_CACHE.load(Ordering::Acquire) }
}

/// Registers the fork handler that gives a forked child a fresh pool cache.
///
/// `Once`: no caller proceeds before the handler is registered.
#[cfg(unix)]
fn register_fork_handler() {
    use std::sync::Once;
    static REGISTERED: Once = Once::new();
    // allocating is fine here: glibc and libSystem unlock malloc before running child handlers
    extern "C" fn switch_to_fresh_cache() {
        POOL_CACHE.store(Box::leak(Box::new(Mutex::new(None))), Ordering::Release);
    }
    REGISTERED.call_once(|| {
        // SAFETY: the handler only allocates and stores an atomic
        unsafe {
            let _ = libc::pthread_atfork(None, None, Some(switch_to_fresh_cache));
        }
    });
}

#[cfg(not(unix))]
fn register_fork_handler() {}

/// How many times [`try_lock_briefly`] retries a held lock before giving up.
const LOCK_ATTEMPTS: u32 = 64;

/// Takes the [`pool_cache`] lock, or gives up after [`LOCK_ATTEMPTS`] tries: the holder may be
/// building a pool, and encoding serially beats waiting for it.
fn try_lock_briefly() -> Option<MutexGuard<'static, Option<CachedPool>>> {
    for attempt in 0..LOCK_ATTEMPTS {
        match pool_cache().try_lock() {
            Ok(g) => return Some(g),
            Err(TryLockError::Poisoned(p)) => return Some(p.into_inner()),
            Err(TryLockError::WouldBlock) if attempt < LOCK_ATTEMPTS / 2 => std::hint::spin_loop(),
            Err(TryLockError::WouldBlock) => std::thread::yield_now(),
        }
    }
    None
}

pub(crate) fn pool() -> Option<Arc<rayon::ThreadPool>> {
    register_fork_handler();

    let generation = POOL_GEN.load(Ordering::Acquire);
    // callers that find the lock held encode serially, so only one of them builds a pool
    let mut cache = try_lock_briefly()?;
    if let Some(cached) = cache.as_ref()
        && cached.generation == generation
    {
        return Some(cached.pool.clone());
    }
    let num_threads = num_threads();
    // no pool for a single thread
    if num_threads == 1 {
        *cache = None;
        return None;
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .thread_name(|i| format!("tk-encode-{i}"))
        .build()
        .ok()?;
    let pool = Arc::new(pool);
    *cache = Some(CachedPool {
        pool: pool.clone(),
        generation,
    });
    Some(pool)
}

pub(crate) fn num_threads() -> usize {
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

/// Get the currently set value for `TOKENIZERS_PARALLELISM` env variable
pub(crate) fn get_parallelism() -> bool {
    match std::env::var(ENV_VARIABLE) {
        Ok(mut v) => {
            v.make_ascii_lowercase();
            !matches!(v.as_ref(), "" | "off" | "false" | "f" | "no" | "n" | "0")
        }
        Err(_) => true, // If we couldn't get the variable, we use the default
    }
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
}

impl<P, S, I> MaybeParallelIterator<P, S> for I
where
    I: IntoParallelIterator<Iter = P, Item = P::Item> + IntoIterator<IntoIter = S, Item = S::Item>,
    P: ParallelIterator,
    S: Iterator<Item = P::Item>,
{
    fn into_maybe_par_iter(self) -> CondIterator<P, S> {
        CondIterator::new(self, get_parallelism())
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
}

/// Converts any serial iterator into a CondIterator, that can either run parallelly or serially.
pub trait MaybeParallelBridge<T, S>
where
    S: Iterator<Item = T> + Send,
    T: Send,
{
    fn maybe_par_bridge(self) -> CondIterator<IterBridge<S>, S>;
}

impl<T, S> MaybeParallelBridge<T, S> for S
where
    S: Iterator<Item = T> + Send,
    T: Send,
{
    fn maybe_par_bridge(self) -> CondIterator<IterBridge<S>, S> {
        let iter = CondIterator::from_serial(self);

        if get_parallelism() {
            CondIterator::from_parallel(iter.into_parallel().right().unwrap())
        } else {
            iter
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::PoisonError;

    /// Tests touching the process-wide pool run one at a time: they hold its lock.
    static POOL_TESTS: Mutex<()> = Mutex::new(());

    /// Forks a child that runs `child` and `_exit`s with its code, killed after 10s instead of
    /// hanging; returns whether it exited 0.
    #[cfg(unix)]
    fn in_forked_child(child: impl FnOnce() -> i32) -> bool {
        // SAFETY: the child only runs `child` and `_exit`s
        let pid = unsafe { libc::fork() };
        if pid == 0 {
            unsafe {
                libc::alarm(10);
                libc::_exit(child());
            }
        }
        let mut status = 0;
        assert_eq!(unsafe { libc::waitpid(pid, &mut status, 0) }, pid);
        libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0
    }

    #[test]
    fn test_maybe_parallel_iterator() {
        let v = vec![1u32, 2, 3, 4, 5, 6];

        assert_eq!(v.maybe_par_iter().sum::<u32>(), 21);
        assert_eq!(v.into_maybe_par_iter().sum::<u32>(), 21);
    }

    /// Contended callers must reuse the cached pool, not build their own.
    #[test]
    fn concurrent_callers_share_one_pool() {
        let _serial = POOL_TESTS.lock().unwrap_or_else(PoisonError::into_inner);
        if num_threads() == 1 {
            return; // no pool on a one-core machine
        }
        // start stale, so the callers race the rebuild too
        invalidate();
        let callers: Vec<_> = (0..32)
            .map(|_| {
                std::thread::spawn(|| {
                    (0..2_000)
                        .filter_map(|_| pool())
                        .map(|pool| Arc::as_ptr(&pool) as usize)
                        .collect::<std::collections::HashSet<_>>()
                })
            })
            .collect();
        let seen: std::collections::HashSet<_> = callers
            .into_iter()
            .flat_map(|caller| caller.join().unwrap())
            .collect();
        assert_eq!(seen.len(), 1, "contended callers built pools of their own");
    }

    /// A child forked while another thread holds the cache lock (as a build does) still gets a pool.
    #[cfg(unix)]
    #[test]
    fn child_forked_mid_lock_gets_a_pool() {
        let _serial = POOL_TESTS.lock().unwrap_or_else(PoisonError::into_inner);
        if num_threads() == 1 {
            return; // no pool on a one-core machine
        }
        register_fork_handler();
        let (held_tx, held_rx) = std::sync::mpsc::channel();
        let holder = std::thread::spawn(move || {
            let _held = try_lock_briefly();
            held_tx.send(()).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(50));
        });
        held_rx.recv().unwrap();
        let ok = in_forked_child(|| i32::from(pool().is_none()));
        holder.join().unwrap();
        assert!(ok, "the child got no pool");
    }

    /// The parent's pool has no workers in a forked child: the child must build and use its own.
    #[cfg(unix)]
    #[test]
    fn child_runs_on_a_pool_of_its_own() {
        let _serial = POOL_TESTS.lock().unwrap_or_else(PoisonError::into_inner);
        if num_threads() == 1 {
            return; // no pool on a one-core machine
        }
        let parent = pool().expect("a pool");
        let parent = Arc::as_ptr(&parent) as usize;
        let ok = in_forked_child(|| {
            let Some(pool) = pool() else { return 1 };
            if Arc::as_ptr(&pool) as usize == parent {
                return 2;
            }
            // hangs (and the alarm kills the child) if the pool has no workers
            let sum: u32 = pool.install(|| (0..1_000u32).into_par_iter().sum());
            i32::from(sum != 499_500)
        });
        assert!(ok, "the child didn't get a working pool of its own");
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
