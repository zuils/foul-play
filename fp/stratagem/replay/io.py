"""Atomic JSON persistence for Stratagem self-play replays."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from .schema import SelfPlayReplay


def save_replay(filepath: str | Path, replay: SelfPlayReplay) -> None:
    """Atomically write one versioned replay JSON document."""
    destination = Path(filepath)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        try:
            json.dump(replay.to_dict(), temporary_file, sort_keys=True)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    try:
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_replay(filepath: str | Path) -> SelfPlayReplay:
    """Read and validate a versioned self-play replay JSON document."""
    source = Path(filepath)
    if not source.is_file():
        raise FileNotFoundError(f"Replay file does not exist: {source}")
    with source.open(encoding="utf-8") as replay_file:
        value = json.load(replay_file)
    if not isinstance(value, dict):
        raise ValueError("Replay payload must be a dictionary")
    return SelfPlayReplay.from_dict(value)