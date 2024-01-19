import pytest

def pytest_addoption(parser):
    parser.addoption(
        "-E",
        action="store",
        metavar="NAME",
        help="only run tests matching the exercise NAME.",
    )

def pytest_configure(config):
    # register an additional marker
    config.addinivalue_line(
        "markers", "ex(name): mark test to run only on named environment"
    )

def pytest_runtest_setup(item):
    envnames = [mark.args[0] for mark in item.iter_markers(name="ex")]
    if envnames:
        if item.config.getoption("-E") not in envnames:
            pytest.skip(f"test requires the exercise to be one of these: {envnames!r}")