from evoflow.config import set_backend, get_backend


def test_set_backend():
    set_backend('numpy')
    assert get_backend() == 'numpy'
