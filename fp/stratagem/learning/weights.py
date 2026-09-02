"""Versioned, atomic persistence for Stratagem model and trainer checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess
import tempfile
from typing import Dict

import torch

from fp.stratagem.config import CONFIG
from fp.stratagem.learning.model import StratagemModelWrapper


CHECKPOINT_VERSION = 2
FEATURE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CheckpointMetadata:
    """Metadata required to inspect and safely resume a Stratagem checkpoint."""

    checkpoint_version: int
    feature_schema_version: int
    feature_size: int
    action_vocabulary_fingerprint: str
    hidden_sizes: tuple[int, ...]
    training_step: int
    episode_count: int
    format: str
    seed: int | None
    configuration: Dict[str, object]
    git_commit: str | None
    created_at: str


def save_training_checkpoint(
    filepath: str | Path,
    model_wrapper: StratagemModelWrapper,
    optimizer: torch.optim.Optimizer,
    *,
    training_step: int,
    episode_count: int,
    total_reward: float,
) -> CheckpointMetadata:
    """Atomically write a complete training checkpoint and return its metadata."""
    metadata = _metadata(model_wrapper, training_step, episode_count)
    payload = {
        "metadata": asdict(metadata),
        "model_state_dict": model_wrapper.model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "total_reward": total_reward,
    }
    destination = Path(filepath)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_checkpoint_path(destination)
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return metadata


def load_training_checkpoint(
    filepath: str | Path,
    model_wrapper: StratagemModelWrapper,
    optimizer: torch.optim.Optimizer,
) -> tuple[CheckpointMetadata, float]:
    """Validate and restore a checkpoint or raise a precise incompatibility error."""
    checkpoint_path = Path(filepath)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=model_wrapper.device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dictionary")
    required_keys = {
        "metadata",
        "model_state_dict",
        "optimizer_state_dict",
        "total_reward",
    }
    if set(payload) != required_keys:
        raise ValueError("Checkpoint payload has an incompatible schema")
    metadata = _parse_metadata(payload["metadata"])
    _validate_compatibility(metadata, model_wrapper)
    total_reward = payload["total_reward"]
    if not isinstance(total_reward, (int, float)):
        raise ValueError("Checkpoint total_reward must be numeric")
    model_wrapper.model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return metadata, float(total_reward)


def _metadata(
    model_wrapper: StratagemModelWrapper, training_step: int, episode_count: int
) -> CheckpointMetadata:
    if training_step < 0 or episode_count < 0:
        raise ValueError("Checkpoint counters must be non-negative")
    return CheckpointMetadata(
        checkpoint_version=CHECKPOINT_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_size=model_wrapper.feature_extractor.feature_size,
        action_vocabulary_fingerprint=_action_vocabulary_fingerprint(model_wrapper),
        hidden_sizes=model_wrapper.hidden_sizes,
        training_step=training_step,
        episode_count=episode_count,
        format=CONFIG.format,
        seed=CONFIG.seed,
        configuration=asdict(CONFIG),
        git_commit=_git_commit(),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _parse_metadata(value: object) -> CheckpointMetadata:
    if not isinstance(value, dict):
        raise ValueError("Checkpoint metadata must be a dictionary")
    expected_keys = set(CheckpointMetadata.__dataclass_fields__)
    if set(value) != expected_keys:
        raise ValueError("Checkpoint metadata has an incompatible schema")
    try:
        metadata = CheckpointMetadata(**value)
    except TypeError as error:
        raise ValueError("Checkpoint metadata has invalid fields") from error
    if not isinstance(metadata.configuration, dict):
        raise ValueError("Checkpoint configuration must be a dictionary")
    return metadata


def _validate_compatibility(
    metadata: CheckpointMetadata, model_wrapper: StratagemModelWrapper
) -> None:
    if metadata.checkpoint_version != CHECKPOINT_VERSION:
        raise ValueError("Checkpoint version is not supported")
    if metadata.feature_schema_version != FEATURE_SCHEMA_VERSION:
        raise ValueError("Checkpoint feature schema version is not supported")
    if metadata.feature_size != model_wrapper.feature_extractor.feature_size:
        raise ValueError("Checkpoint feature size does not match the current model")
    if metadata.action_vocabulary_fingerprint != _action_vocabulary_fingerprint(model_wrapper):
        raise ValueError("Checkpoint action vocabulary does not match the current model")


def _action_vocabulary_fingerprint(model_wrapper: StratagemModelWrapper) -> str:
    vocabulary = "\n".join(model_wrapper.action_vocabulary)
    return hashlib.sha256(vocabulary.encode("ascii")).hexdigest()


def _git_commit() -> str | None:
    repository_root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        capture_output=True,
        check=False,
        text=True,
    )
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


def _temporary_checkpoint_path(destination: Path) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as temporary_file:
        return Path(temporary_file.name)