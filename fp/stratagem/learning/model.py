"""Learned value and policy model components for Stratagem."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from fp import constants
from fp.data import all_move_json, pokedex
from fp.stratagem.core.observation import Observation
from fp.stratagem.learning.features import StratagemFeatureExtractor, extract_features


class StratagemValueNetwork(nn.Module):
    """Predict a scalar state value and logits for the fixed Stratagem action vocabulary."""

    def __init__(
        self,
        feature_size: int,
        action_size: int,
        hidden_sizes: Sequence[int] = (256, 128),
    ) -> None:
        super().__init__()
        self.feature_size = feature_size
        self.action_size = action_size
        layers: list[nn.Module] = []
        previous_size = feature_size
        for hidden_size in hidden_sizes:
            layers.extend((nn.Linear(previous_size, hidden_size), nn.ReLU(), nn.Dropout(0.1)))
            previous_size = hidden_size
        self.shared_layers = nn.Sequential(*layers)
        self.value_head = nn.Sequential(nn.Linear(previous_size, 64), nn.ReLU(), nn.Linear(64, 1))
        self.policy_head = nn.Sequential(nn.Linear(previous_size, 64), nn.ReLU(), nn.Linear(64, action_size))

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared_features = self.shared_layers(features)
        return self.value_head(shared_features), self.policy_head(shared_features)

    def get_action_preferences(self, features: torch.Tensor) -> torch.Tensor:
        _, policy_logits = self(features)
        return functional.softmax(policy_logits, dim=-1)

    def get_state_value(self, features: torch.Tensor) -> torch.Tensor:
        value, _ = self(features)
        return value


class StratagemModelWrapper:
    """Own the fixed action vocabulary, feature extraction, and model inference."""

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "cpu",
        hidden_sizes: Sequence[int] | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.feature_extractor = StratagemFeatureExtractor()
        self.action_vocabulary = self._build_action_vocabulary()
        self.action_to_idx = {
            action: index for index, action in enumerate(self.action_vocabulary)
        }
        self.action_size = len(self.action_vocabulary)
        if hidden_sizes is None:
            hidden_sizes = self._checkpoint_hidden_sizes(model_path) if model_path else (128,)
        self.hidden_sizes = tuple(hidden_sizes)
        self.model = StratagemValueNetwork(
            feature_size=self.feature_extractor.feature_size,
            action_size=self.action_size,
            hidden_sizes=self.hidden_sizes,
        ).to(self.device)
        if model_path is not None:
            self.load_weights(model_path)
        self.model.eval()

    @staticmethod
    def _build_action_vocabulary() -> Tuple[str, ...]:
        moves = tuple(sorted(all_move_json))
        switches = tuple(
            f"{constants.SWITCH_STRING} {species}" for species in sorted(pokedex)
        )
        return moves + switches

    def action_to_index(self, action: str) -> int:
        try:
            return self.action_to_idx[action]
        except KeyError as error:
            raise ValueError(f"Action is not in the fixed model vocabulary: {action}") from error

    def predict(self, observation: Observation) -> Tuple[float, np.ndarray]:
        features = extract_features(observation)
        features_tensor = torch.as_tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            value_tensor, policy_logits = self.model(features_tensor)
            action_probabilities = functional.softmax(policy_logits, dim=-1).cpu().numpy().flatten()
        return float(value_tensor.item()), action_probabilities

    def get_action_priors(self, observation: Observation) -> np.ndarray:
        _, action_probabilities = self.predict(observation)
        return action_probabilities

    def get_candidate_action_priors(
        self, observation: Observation, candidate_actions: Sequence[str]
    ) -> Dict[str, float]:
        if not candidate_actions:
            raise ValueError("Candidate action priors require at least one legal action")
        if len(set(candidate_actions)) != len(candidate_actions):
            raise ValueError("Candidate actions must be unique")
        indices = [self.action_to_index(action) for action in candidate_actions]
        action_probabilities = self.get_action_priors(observation)
        candidate_probabilities = [float(action_probabilities[index]) for index in indices]
        total_probability = sum(candidate_probabilities)
        if total_probability <= 0.0 or not np.isfinite(total_probability):
            raise ValueError("Model produced invalid candidate action probabilities")
        return {
            action: float(probability / total_probability)
            for action, probability in zip(candidate_actions, candidate_probabilities)
        }

    def get_state_value(self, observation: Observation) -> float:
        value, _ = self.predict(observation)
        return value

    def load_weights(self, model_path: str) -> None:
        payload = torch.load(model_path, map_location=self.device, weights_only=True)
        if isinstance(payload, dict) and "model_state_dict" in payload:
            from fp.stratagem.learning.weights import (
                _parse_metadata,
                _validate_compatibility,
            )

            _validate_compatibility(_parse_metadata(payload.get("metadata")), self)
            state_dict = payload["model_state_dict"]
        else:
            state_dict = payload
        if not isinstance(state_dict, dict):
            raise ValueError("Model weights must be a state dictionary or Stratagem checkpoint")
        self.model.load_state_dict(state_dict)

    def save_weights(self, model_path: str) -> None:
        torch.save(self.model.state_dict(), model_path)

    @staticmethod
    def _checkpoint_hidden_sizes(model_path: str) -> Sequence[int]:
        """Read the architecture recorded by a trainer checkpoint before construction."""
        payload = torch.load(model_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or "metadata" not in payload:
            return (128,)
        metadata = payload["metadata"]
        if not isinstance(metadata, dict):
            raise ValueError("Checkpoint metadata must be a dictionary")
        hidden_sizes = metadata.get("hidden_sizes")
        if (
            not isinstance(hidden_sizes, (list, tuple))
            or not hidden_sizes
            or any(not isinstance(size, int) or size <= 0 for size in hidden_sizes)
        ):
            raise ValueError("Checkpoint hidden_sizes must be a non-empty positive integer list")
        return tuple(hidden_sizes)


_model_wrapper: StratagemModelWrapper | None = None


def get_model_wrapper(device: str = "cpu") -> StratagemModelWrapper:
    global _model_wrapper
    if _model_wrapper is None:
        _model_wrapper = StratagemModelWrapper(device=device)
    return _model_wrapper


def predict_action_values(observation: Observation, device: str = "cpu") -> Tuple[float, np.ndarray]:
    return get_model_wrapper(device).predict(observation)
