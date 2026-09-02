"""Action filtering and mixed strategy selection for Stratagem candidates."""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional

from fp import constants
from fp.battle.helpers import type_effectiveness_modifier
from fp.data import all_move_json

from .observation import Observation


class StrategicActionSelector:
    """Filters publicly invalid actions and samples near-tied engine candidates."""

    def __init__(self, temperature: float = 1.0, random_seed: Optional[int] = None):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature
        self.rng = random.Random(random_seed)

    def legal_actions(self, observation: Observation, scores: Dict[str, float]) -> Dict[str, float]:
        """Keep engine candidates that are legal and not visibly immune attacks."""
        legal = {
            action: score
            for action, score in scores.items()
            if self._is_available(observation, action)
            and not self._is_revealed_immune_move(observation, action)
        }
        if legal:
            return legal
        raise ValueError("No legal scored actions remain after public-information guards")

    def select_action(self, observation: Observation, scores: Dict[str, float]) -> str:
        """Sample a legal action using a stable softmax over engine scores."""
        legal = self.legal_actions(observation, scores)
        highest_score = max(legal.values())
        actions = sorted(legal)
        weights = [
            math.exp((legal[action] - highest_score) / self.temperature)
            for action in actions
        ]
        return self.rng.choices(actions, weights=weights, k=1)[0]

    @staticmethod
    def _is_available(observation: Observation, action: str) -> bool:
        if action in observation.available_moves:
            return True
        if action.startswith("switch "):
            return observation.can_switch
        if action.endswith("-tera"):
            return (
                observation.can_terastallize
                and not observation.is_terastallized
                and action.removesuffix("-tera") in observation.available_moves
            )
        if action.endswith("-mega"):
            return (
                observation.can_mega_evolve
                and not observation.is_mega_evolved
                and action.removesuffix("-mega") in observation.available_moves
            )
        return False

    @staticmethod
    def _is_revealed_immune_move(observation: Observation, action: str) -> bool:
        opponent = observation.opponent_active
        base_action = action.removesuffix("-tera").removesuffix("-mega")
        if (
            base_action not in all_move_json
            or not opponent
            or not opponent.get("revealed")
        ):
            return False
        move = all_move_json[base_action]
        if move[constants.CATEGORY] == constants.MoveCategory.STATUS:
            return False
        opponent_types: List[str] = opponent.get("types", [])
        if opponent.get("is_terastallized") and opponent.get("tera_type"):
            opponent_types = [opponent["tera_type"]]
        return type_effectiveness_modifier(move[constants.TYPE], opponent_types) == 0