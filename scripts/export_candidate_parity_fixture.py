"""Export existing research evidence, never recompute expected scores with production code."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(research_root: Path, output: Path, sessions: list[str]) -> None:
    root = research_root / "session_hard_initial_rv50_recall_v0"
    sources = pd.read_csv(root / "session_source_inventory.csv").set_index("session")
    expected = pd.read_parquet(
        root / "causal_features.parquet",
        columns=[
            "session",
            "symbol",
            "snapshot",
            "candidate_available",
            "cap_bucket",
            "tie",
            "rv",
            "range_pct",
        ],
        filters=[("session", "in", sessions), ("snapshot", "in", [5, 10, 15])],
    )
    downstream = research_root / "session_hard_upstream_rv30_recall_v0" / "causal_features.parquet"
    expected = pd.concat(
        [
            expected,
            pd.read_parquet(
                downstream,
                columns=expected.columns.tolist(),
                filters=[("session", "in", sessions), ("snapshot", "==", 15)],
            ),
        ],
        ignore_index=True,
    )
    frames = []
    provenance = []
    for day in sessions:
        source = Path(sources.loc[day, "path"])
        if digest(source) != sources.loc[day, "sha256"]:
            raise ValueError(f"Research bar source hash mismatch: {day}")
        # The US historical fixtures are stamped explicitly in their source market zone.
        opening = pd.Timestamp(day + " 09:30", tz="America/New_York").tz_convert("UTC")
        frame = (
            ds.dataset(source, format="parquet")
            .to_table(
                columns=["symbol", "bar_start_utc", "open", "high", "low", "close", "volume"],
                filter=(ds.field("bar_start_utc") >= opening)
                & (ds.field("bar_start_utc") < opening + pd.Timedelta(minutes=15)),
            )
            .to_pandas()
        )
        frame["session"] = day
        frames.append(frame)
        provenance.append(
            {
                "session": day,
                "source_file": source.name,
                "source_sha256": digest(source),
                "opening_utc": opening.isoformat(),
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(output / "opening_bars.parquet", index=False)
    expected.to_parquet(output / "expected_scores.parquet", index=False)
    lists = {}
    for stage, filename in [
        ("RANGE250", "initial_pool_watchlists.json"),
        ("RV50", "post_initial_rv50_watchlists.json"),
        ("RV30", "final_rv30_watchlists.json"),
    ]:
        saved = json.loads((root / filename).read_text())
        lists[stage] = {day: saved[day] for day in sessions}
    (output / "expected_watchlists.json").write_text(json.dumps(lists, indent=2) + "\n")
    inputs = [
        "causal_features.parquet",
        "initial_pool_watchlists.json",
        "post_initial_rv50_watchlists.json",
        "final_rv30_watchlists.json",
        "frozen_initial_recipe.json",
        "session_source_inventory.csv",
    ]
    manifest = {
        "purpose": "Research-only test fixtures; never production or PAPER history inputs",
        "research": "session_hard_initial_rv50_recall_v0",
        "evidence": "Already exposed US development/validation population, not untouched evidence",
        "sources": provenance,
        "downstream_rv15_source_sha256": digest(downstream),
        "research_sha256": {name: digest(root / name) for name in inputs},
        "fixtures_sha256": {
            name: digest(output / name)
            for name in [
                "opening_bars.parquet",
                "expected_scores.parquet",
                "expected_watchlists.json",
            ]
        },
        "identity_note": "Tests use deterministic synthetic conIds; research symbol ties unchanged",
        "cap_note": "Saved current static cap metadata is descriptive only; never a selector input",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sessions", nargs="+", default=["2025-06-02", "2025-07-03", "2025-07-17"])
    arguments = parser.parse_args()
    export(arguments.research_root, arguments.output, arguments.sessions)
