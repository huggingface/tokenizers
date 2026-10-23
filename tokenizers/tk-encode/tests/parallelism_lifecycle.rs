use std::error::Error;
use std::sync::{Arc, Barrier};

// Exercise the private resource owner without adding a public testing API.
#[allow(dead_code, unused_imports)]
#[path = "../src/utils/parallelism.rs"]
mod parallelism;

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

fn concurrent_pool() -> Result<Arc<rayon::ThreadPool>> {
    const CALLERS: usize = 16;
    let gate = Barrier::new(CALLERS);
    let pools = std::thread::scope(|scope| {
        let callers: Vec<_> = (0..CALLERS)
            .map(|_| {
                scope.spawn(|| {
                    gate.wait();
                    parallelism::pool().ok_or("pool construction failed")
                })
            })
            .collect();
        callers
            .into_iter()
            .map(|caller| caller.join().map_err(|_| "caller panicked")?)
            .collect::<std::result::Result<Vec<_>, _>>()
    })?;
    let first = pools.first().ok_or("no callers completed")?;
    if pools.iter().any(|pool| !Arc::ptr_eq(first, pool)) {
        return Err("concurrent calls constructed distinct shared pools".into());
    }
    Ok(first.clone())
}

fn check_workers(pool: &rayon::ThreadPool, expected: usize) -> Result<()> {
    let (workers, worker) = pool.install(|| {
        (
            parallelism::current_num_threads(),
            rayon::current_thread_index(),
        )
    });
    if workers != expected || worker.is_none() {
        return Err(
            format!("expected a job on {expected} workers, got {workers}, {worker:?}").into(),
        );
    }
    Ok(())
}

#[cfg(unix)]
fn fork_pool(parent: &Arc<rayon::ThreadPool>) -> Result<()> {
    // The child exits without dropping objects inherited from the parent.
    let pid = unsafe { libc::fork() };
    if pid < 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    if pid == 0 {
        unsafe { libc::alarm(10) };
        let result = (|| -> Result<()> {
            let first = concurrent_pool()?;
            let second = concurrent_pool()?;
            if Arc::ptr_eq(parent, &first) || !Arc::ptr_eq(&first, &second) {
                return Err("child must create and then reuse its own pool".into());
            }
            check_workers(&first, 4)
        })();
        let status = match result {
            Ok(()) => 0,
            Err(error) => {
                eprintln!("fork child: {error}");
                1
            }
        };
        unsafe { libc::_exit(status) };
    }
    let mut status = 0;
    loop {
        if unsafe { libc::waitpid(pid, &mut status, 0) } >= 0 {
            break;
        }
        let error = std::io::Error::last_os_error();
        if error.kind() != std::io::ErrorKind::Interrupted {
            return Err(error.into());
        }
    }
    if !libc::WIFEXITED(status) || libc::WEXITSTATUS(status) != 0 {
        return Err(format!("fork child failed with wait status {status}").into());
    }
    let after = parallelism::pool().ok_or("parent lost its pool after fork")?;
    if !Arc::ptr_eq(parent, &after) {
        return Err("fork replaced the parent's pool".into());
    }
    check_workers(&after, 4)
}

fn main() -> Result<()> {
    parallelism::set_num_threads(4);
    let cold = concurrent_pool()?;
    check_workers(&cold, 4)?;
    let warm = concurrent_pool()?;
    if !Arc::ptr_eq(&cold, &warm) {
        return Err("warm callers replaced the cached pool".into());
    }
    println!("cold and warm callers share one pool");

    #[cfg(unix)]
    fork_pool(&cold)?;
    #[cfg(unix)]
    println!("child creates and reuses its pool; parent retains its pool");

    parallelism::set_num_threads(1);
    for _ in 0..4 {
        if parallelism::pool().is_some() {
            return Err("disabled parallelism returned a pool".into());
        }
    }
    check_workers(&cold, 4)?;
    parallelism::set_num_threads(2);
    let enabled = concurrent_pool()?;
    if Arc::ptr_eq(&cold, &enabled) {
        return Err("re-enabling parallelism reused the retired pool".into());
    }
    check_workers(&enabled, 2)?;
    drop(cold);
    drop(warm);
    println!("disable and re-enable preserve borrowed pools and apply the new size");

    parallelism::set_num_threads(0);
    let expected = std::thread::available_parallelism()
        .map(|threads| threads.get())
        .unwrap_or(1);
    match parallelism::pool() {
        Some(pool) if expected > 1 => check_workers(&pool, expected)?,
        None if expected == 1 => {}
        _ => return Err("resetting the thread count did not restore the default".into()),
    }
    println!("resetting to zero restores the platform default");
    Ok(())
}
