import pytest

@pytest.fixture
def fix1():
    print("setup fix1")
    yield
    print("teardown fix1")

@pytest.fixture
def fix2(fix1):
    print("setup fix2")
    yield
    print("teardown fix2")

def test_deps(fix2):
    pass
