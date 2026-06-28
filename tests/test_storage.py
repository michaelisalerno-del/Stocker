from pathlib import Path

import pandas as pd

from stocker_data.storage import dataset_exists, read_parquet, resolve_data_path, write_parquet


def test_resolve_data_path_stays_under_data_dir(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"

    path = resolve_data_path(data_dir, "raw", "prices.parquet")

    assert path == data_dir / "raw" / "prices.parquet"


def test_write_and_read_parquet_round_trip(tmp_path: Path) -> None:
    frame = pd.DataFrame({"close": [100.0, 101.5]})
    path = resolve_data_path(tmp_path, "processed", "sample.parquet")

    write_parquet(frame, path)

    assert dataset_exists(path)
    loaded = read_parquet(path)
    assert loaded["close"].tolist() == [100.0, 101.5]
