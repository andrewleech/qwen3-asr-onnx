import pytest
import torch

_EXPORT_TESTS = {"test_encoder.py", "test_decoder.py", "test_pipeline.py", "test_compare.py"}


def pytest_collection_modifyitems(items):
    for item in items:
        if item.path.name in _EXPORT_TESTS:
            item.add_marker(pytest.mark.integration)


@pytest.fixture(scope="session", autouse=True)
def bounded_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)
