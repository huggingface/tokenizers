import pytest


try:
    import pytest_run_parallel  # noqa:F401
    PARALLEL_RUN_AVAILABLE = True
except ImportError:
    PARALLEL_RUN_AVAILABLE = False

def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    # Don't warn for `pytest-run-parallel` markers if they're not available
    if not PARALLEL_RUN_AVAILABLE:
        config.addinivalue_line(
            "markers",
            "thread_unsafe: mark the test function as single-threaded",
        )

def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
