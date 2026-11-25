//! Helpers shared by tests in several modules.

/// Forks a child that runs `child` and exits with its return value. Returns whether that was 0.
pub(crate) fn in_forked_child(child: impl FnOnce() -> i32) -> bool {
    forked_child_exit_code(child) == Some(0)
}

/// Forks a child that runs `child` and exits with its return value. Returns that value, or `None`
/// if the child didn't exit normally.
///
/// The child is killed after 10 s, so a child that hangs fails the test instead of hanging it.
pub(crate) fn forked_child_exit_code(child: impl FnOnce() -> i32) -> Option<i32> {
    // SAFETY: the child only runs `child`, then `_exit`s without running the parent's cleanup
    let pid = unsafe { libc::fork() };
    if pid == 0 {
        // SAFETY: plain libc calls in the child
        unsafe {
            libc::alarm(10);
            libc::_exit(child());
        }
    }
    let mut status = 0;
    // SAFETY: waits on the child forked above
    assert_eq!(unsafe { libc::waitpid(pid, &mut status, 0) }, pid);
    libc::WIFEXITED(status).then(|| libc::WEXITSTATUS(status))
}
