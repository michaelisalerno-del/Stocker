import pytest

from stocker_execution import recorder


@pytest.fixture(autouse=True)
def quick_recorder_batches(monkeypatch):
    # The recorder gathers rows for a second before each gzip member (disk economy); tests
    # that wait on queue.join() would pay that per flush.
    monkeypatch.setattr(recorder, "BATCH_SECONDS", 0.1)
