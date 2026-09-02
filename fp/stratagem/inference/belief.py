"""
Belief module for Stratagem system.
Maintains a distribution over opponent team hypotheses and updates based on evidence.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional

from fp import constants
from fp.battle.helpers import calculate_stats
from fp.data import pokedex
from fp.stratagem.core import Observation
from fp.stratagem.inference.team_sampler import TeamSampler


class Belief:
    """Maintains a distribution over possible opponent team hypotheses."""

    def __init__(
        self,
        team_sampler: TeamSampler,
        world_count: int = 100,
        random_seed: Optional[int] = None,
    ):
        self.team_sampler = team_sampler
        self.world_count = world_count
        self.rng = random.Random(random_seed) if random_seed is not None else random.Random()
        self.worlds: List[List[dict]] = []
        self.weights: List[float] = []
        self._evidence_applied = False
        self._previous_observation: Optional[Observation] = None

    def _calculate_pokemon_stats(
        self, pkmn_dict: dict, level: int = 100
    ) -> Optional[Dict[str, int]]:
        """Calculate candidate stats when the candidate identifies a known species."""
        species = pkmn_dict.get("species")
        if not species or species not in pokedex:
            return None

        evs = pkmn_dict.get("evs", [0, 0, 0, 0, 0, 0])
        if len(evs) != 6:
            evs = (evs + [0, 0, 0, 0, 0, 0])[:6]

        return calculate_stats(
            base_stats=pokedex[species][constants.BASESTATS],
            level=level,
            evs=evs,
            nature=pkmn_dict.get("nature", "serious"),
        )

    def initialize_from_observation(self, observation: Observation) -> None:
        """Initialize sampled worlds from an observation."""
        self.worlds = self.team_sampler.sample_multiple_teams(observation, self.world_count)
        self.weights = [0.0] * self.world_count
        self._evidence_applied = False
        self._previous_observation = observation

    def update_with_evidence(self, observation: Observation) -> None:
        """Update world weights from public evidence in an observation."""
        for index, world in enumerate(self.worlds):
            self.weights[index] += self._compute_log_likelihood(
                world, observation, self._previous_observation
            )
        self._evidence_applied = True
        self._previous_observation = observation

    def _compute_log_likelihood(
        self,
        world: List[dict],
        observation: Observation,
        previous_observation: Optional[Observation] = None,
    ) -> float:
        """Score a world using only the Observation's public opponent evidence."""
        del previous_observation

        evidence = []
        if observation.opponent_active.get("revealed", False):
            evidence.append(
                (
                    observation.opponent_active,
                    observation.opponent_is_choice_locked,
                    observation.opponent_locked_move,
                )
            )
        evidence.extend(
            (reserve, False, None)
            for reserve in observation.opponent_reserve_revealed
            if reserve.get("revealed", False)
        )

        log_likelihood = 0.0
        for observed_pokemon, is_choice_locked, locked_move in evidence:
            candidate_scores = [
                self._score_candidate(
                    candidate, observed_pokemon, is_choice_locked, locked_move
                )
                for candidate in world
                if self._pkmn_matches_observation(candidate, observed_pokemon)
            ]
            if not candidate_scores:
                return -float("inf")
            log_likelihood += max(candidate_scores)

        return log_likelihood

    def _score_candidate(
        self,
        candidate: dict,
        observed_pokemon: dict,
        is_choice_locked: bool,
        locked_move: Optional[str],
    ) -> float:
        """Return soft evidence for a candidate already compatible with observation."""
        score = 0.0
        observed_max_hp = observed_pokemon.get("max_hp")
        observed_level = observed_pokemon.get("level")
        if observed_max_hp is not None and observed_level is not None:
            stats = self._calculate_pokemon_stats(candidate, level=observed_level)
            if stats is not None and stats.get(constants.HITPOINTS) == observed_max_hp:
                score += 0.5

        if is_choice_locked and locked_move:
            candidate_moves = set(candidate.get("moves", []))
            if locked_move in candidate_moves:
                score += 1.0

        return score

    def _pkmn_matches_observation(self, pkmn: dict, obs: dict) -> bool:
        """Check the required public compatibility constraints for one Pokemon."""
        if pkmn.get("species") != obs.get("name"):
            return False

        observed_ability = obs.get("ability")
        if observed_ability is not None and pkmn.get("ability") != observed_ability:
            return False

        observed_item = obs.get("item")
        if observed_item not in (None, constants.UNKNOWN_ITEM):
            if self._normalize_item(pkmn.get("item")) != self._normalize_item(observed_item):
                return False

        observed_moves = set(obs.get("moves") or [])
        if not observed_moves.issubset(set(pkmn.get("moves", []))):
            return False

        observed_tera_type = obs.get("tera_type")
        if observed_tera_type is not None and pkmn.get("tera_type") != observed_tera_type:
            return False

        return True

    @staticmethod
    def _normalize_item(item: Optional[str]) -> Optional[str]:
        if item is None or item == constants.UNKNOWN_ITEM:
            return item
        return item.lower().replace(" ", "").replace("-", "")

    def sample_world(self) -> List[dict]:
        """Sample one world according to its current likelihood."""
        if not self._evidence_applied:
            return self.rng.choice(self.worlds)
        return self.rng.choices(self.worlds, weights=self.get_world_weights(), k=1)[0]

    def sample_worlds(self, count: int) -> List[List[dict]]:
        """Sample multiple worlds according to their current likelihood."""
        return [self.sample_world() for _ in range(count)]

    def get_world_weights(self) -> List[float]:
        """Return normalized probability weights for worlds."""
        if not self.worlds:
            return []
        if not self._evidence_applied:
            return [1.0 / len(self.worlds)] * len(self.worlds)

        max_weight = max(self.weights)
        if max_weight == -float("inf"):
            return [1.0 / len(self.worlds)] * len(self.worlds)
        exp_weights = [math.exp(weight - max_weight) for weight in self.weights]
        total = sum(exp_weights)
        if total == 0:
            return [1.0 / len(self.worlds)] * len(self.worlds)
        return [weight / total for weight in exp_weights]

    def get_most_likely_world(self) -> List[dict]:
        """Return the world with the largest log likelihood."""
        if not self.worlds:
            return []
        return self.worlds[self.weights.index(max(self.weights))]

    def get_effective_world_count(self) -> float:
        """Return the exponential entropy of the belief distribution."""
        if not self._evidence_applied:
            return float(self.world_count)

        entropy = 0.0
        for probability in self.get_world_weights():
            if probability > 0:
                entropy -= probability * math.log(probability)
        return math.exp(entropy)


UNKNOWN_ITEM = constants.UNKNOWN_ITEM