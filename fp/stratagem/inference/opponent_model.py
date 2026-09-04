"""
Opponent model for Stratagem system.
Predicts opponent behavior (lead, moves, sequences) based on belief state.
"""

from __future__ import annotations

import random
from typing import Dict, Iterator, List, Optional, Tuple
from fp.battle.helpers import normalize_name
from poke_engine import MctsResult, generate_instructions, monte_carlo_tree_search
from fp.stratagem.inference.belief import Belief
from fp.stratagem.core import Observation
from fp.stratagem.engine.adapter import build_state, mcts_action_to_engine_input


class OpponentModel:
    """
    Predicts opponent behavior based on belief state over possible teams.
    Provides lead prediction, move prediction, and sequence prediction.
    """

    def __init__(
        self,
        belief: Belief,
        our_team: Optional[List[Dict]] = None,
        search_time_ms: int = 100,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize opponent model.

        Args:
            belief: Belief state maintaining distribution over opponent teams
            our_team: Full known team specification for engine-backed predictions.
            search_time_ms: MCTS budget for each predicted state.
            random_seed: Optional seed for reproducible predictions
        """
        self.belief = belief
        self.our_team = our_team
        self.search_time_ms = search_time_ms
        self.rng = random.Random(random_seed) if random_seed is not None else random.Random()

    def predict_lead(
        self,
        observation: Observation,
        our_team: Optional[List[Dict]] = None,
        search_time_ms: Optional[int] = None,
    ) -> Dict[str, object]:
        """
        Predict opponent lead Pokemon at battle start.

        Args:
            observation: Current battle observation

        Returns:
            Dictionary with prediction results:
                - species: Predicted lead Pokemon species
                - confidence: Confidence score (0-1)
                - alternatives: List of alternative predictions with scores
        """
        observed_active = observation.opponent_active.get("name")
        if observed_active:
            return {
                "species": observed_active,
                "confidence": 1.0,
                "alternatives": [{"species": observed_active, "probability": 1.0}],
                "probabilities": {observed_active: 1.0},
                "source": "observed",
            }

        engine_config = self._engine_config(our_team, search_time_ms)
        if engine_config and observation.team_preview:
            action_votes = self._engine_action_votes(observation, *engine_config)
            lead_votes = {
                action.removeprefix("switch "): probability
                for action, probability in action_votes.items()
                if action.startswith("switch ")
            }
            predicted_species, confidence, alternatives = self._rank_distribution(
                lead_votes, "species"
            )
            return {
                "species": predicted_species,
                "confidence": confidence,
                "alternatives": alternatives,
                "probabilities": self._normalized_distribution(lead_votes),
                "source": "mcts",
            }

        lead_votes: Dict[str, float] = {}
        for world, weight in self._weighted_worlds():
            if not world:
                continue
            per_pokemon_weight = weight / len(world)
            for candidate in world:
                species = candidate["species"]
                lead_votes[species] = lead_votes.get(species, 0.0) + per_pokemon_weight

        predicted_species, confidence, alternatives = self._rank_distribution(
            lead_votes, "species"
        )

        return {
            "species": predicted_species,
            "confidence": confidence,
            "alternatives": alternatives,
            "probabilities": self._normalized_distribution(lead_votes),
            "source": "belief",
        }

    def predict_move(
        self,
        observation: Observation,
        our_team: Optional[List[Dict]] = None,
        search_time_ms: Optional[int] = None,
    ) -> Dict[str, object]:
        """
        Predict opponent first move.

        Args:
            observation: Current battle observation

        Returns:
            Dictionary with prediction results:
                - move: Predicted move name
                - confidence: Confidence score (0-1)
                - alternatives: List of alternative predictions with scores
        """
        active_species = self._active_species(observation)
        engine_config = self._engine_config(our_team, search_time_ms)
        if engine_config:
            action_votes = self._engine_action_votes(observation, *engine_config)
            predicted_move, confidence, alternatives = self._rank_distribution(
                action_votes, "move"
            )
            return {
                "move": predicted_move,
                "confidence": confidence,
                "alternatives": alternatives,
                "probabilities": self._normalized_distribution(action_votes),
                "source": "mcts",
            }

        move_votes: Dict[str, float] = {}

        for world, weight in self._weighted_worlds():
            candidates = self._pokemon_for_species(world, active_species)
            if not candidates:
                continue
            per_candidate_weight = weight / len(candidates)
            for candidate in candidates:
                moves = candidate.get("moves", [])
                if not moves:
                    continue
                per_move_weight = per_candidate_weight / len(moves)
                for move in moves:
                    move_votes[move] = move_votes.get(move, 0.0) + per_move_weight

        predicted_move, confidence, alternatives = self._rank_distribution(
            move_votes, "move"
        )

        return {
            "move": predicted_move,
            "confidence": confidence,
            "alternatives": alternatives,
            "probabilities": self._normalized_distribution(move_votes),
            "source": "belief",
        }

    def _weighted_worlds(self) -> Iterator[Tuple[List[Dict], float]]:
        weights = self.belief.get_world_weights()
        if not self.belief.worlds or len(weights) != len(self.belief.worlds):
            raise ValueError("Opponent predictions require an initialized belief state")
        yield from zip(self.belief.worlds, weights)

    def _engine_config(
        self, our_team: Optional[List[Dict]], search_time_ms: Optional[int]
    ) -> Optional[Tuple[List[Dict], int]]:
        resolved_our_team = our_team or self.our_team
        if not resolved_our_team:
            return None
        resolved_search_time = search_time_ms or self.search_time_ms
        if resolved_search_time <= 0:
            raise ValueError("search_time_ms must be positive")
        return resolved_our_team, resolved_search_time

    def _engine_action_votes(
        self, observation: Observation, our_team: List[Dict], search_time_ms: int
    ) -> Dict[str, float]:
        action_votes: Dict[str, float] = {}
        for world, weight in self._weighted_worlds():
            result = monte_carlo_tree_search(
                build_state(observation, world, our_team), search_time_ms
            )
            for action, probability in self._policy_distribution(result, side="two").items():
                action_votes[action] = action_votes.get(action, 0.0) + weight * probability
        return action_votes

    @staticmethod
    def _pokemon_for_species(world: List[Dict], species: str) -> List[Dict]:
        return [
            candidate
            for candidate in world
            if normalize_name(candidate["species"]) == species
        ]

    @staticmethod
    def _active_species(observation: Observation) -> str:
        active = observation.opponent_active
        if not active or not active.get("name"):
            raise ValueError("Move prediction requires an observed opponent active Pokemon")
        return normalize_name(active["name"])

    @staticmethod
    def _normalized_distribution(distribution: Dict[str, float]) -> Dict[str, float]:
        total = sum(distribution.values())
        if total <= 0:
            raise ValueError("No candidate actions are available for prediction")
        return {name: weight / total for name, weight in distribution.items()}

    @classmethod
    def _rank_distribution(
        cls, distribution: Dict[str, float], label: str
    ) -> Tuple[str, float, List[Dict[str, object]]]:
        ranked = sorted(
            cls._normalized_distribution(distribution).items(),
            key=lambda entry: (-entry[1], entry[0]),
        )
        prediction, confidence = ranked[0]
        return (
            prediction,
            confidence,
            [{label: name, "probability": probability} for name, probability in ranked[:3]],
        )

    def predict_sequence(
        self,
        observation: Observation,
        horizon: int = 3,
        our_team: Optional[List[Dict]] = None,
        search_time_ms: Optional[int] = None,
    ) -> Dict[str, object]:
        """
        Predict multi-step action sequences.

        Args:
            observation: Current battle observation
            horizon: Number of steps to predict into the future

        Returns:
            Dictionary with prediction results:
                - sequence: List of predicted actions (each action is a dict)
                - confidence: Overall confidence score (0-1)
                - step_by_step: Per-step predictions with confidence
        """
        if horizon < 0:
            raise ValueError("Prediction horizon must be non-negative")
        if horizon == 0:
            return {
                "sequence": [],
                "confidence": 1.0,
                "step_by_step": [],
                "source": "engine-rollout",
            }

        engine_config = self._engine_config(our_team, search_time_ms)
        if not engine_config:
            raise ValueError(
                "Sequence prediction requires the actor's full known team for engine simulation"
            )
        resolved_our_team, resolved_search_time = engine_config

        active_states = [
            (build_state(observation, world, resolved_our_team), weight)
            for world, weight in self._weighted_worlds()
        ]
        sequence = []
        step_by_step = []

        for step in range(1, horizon + 1):
            action_votes: Dict[str, float] = {}
            next_states = []

            for state, world_weight in active_states:
                result = monte_carlo_tree_search(state, resolved_search_time)
                user_policy = self._policy_distribution(result, side="one")
                opponent_policy = self._policy_distribution(result, side="two")
                for action, probability in opponent_policy.items():
                    action_votes[action] = (
                        action_votes.get(action, 0.0) + world_weight * probability
                    )

                user_action = self._most_likely_action(user_policy)
                opponent_action = self._most_likely_action(opponent_policy)
                transitions = generate_instructions(
                    state,
                    mcts_action_to_engine_input(user_action),
                    mcts_action_to_engine_input(opponent_action),
                )
                if not transitions:
                    continue
                transition = self.rng.choices(
                    transitions,
                    weights=[branch.percentage for branch in transitions],
                    k=1,
                )[0]
                state = state.apply_instructions(transition)
                next_states.append((state, world_weight))

            if not action_votes:
                break
            action, confidence, alternatives = self._rank_distribution(
                action_votes, "action"
            )
            prediction = {
                "action": action,
                "confidence": confidence,
                "alternatives": alternatives,
                "probabilities": self._normalized_distribution(action_votes),
                "source": "engine-rollout",
            }
            step_by_step.append({"step": step, "type": "action", "prediction": prediction})
            sequence.append({"step": step, **prediction})
            active_states = next_states
            if not active_states:
                break

        overall_confidence = 1.0
        for step in sequence:
            overall_confidence *= step["confidence"]

        return {
            "sequence": sequence,
            "confidence": overall_confidence,
            "step_by_step": step_by_step,
            "source": "engine-rollout",
        }

    @staticmethod
    def _policy_distribution(result: MctsResult, side: str) -> Dict[str, float]:
        options = result.side_one if side == "one" else result.side_two
        total_visits = sum(option.visits for option in options)
        if total_visits <= 0:
            raise ValueError("MCTS returned no action visits for sequence prediction")
        return {
            option.move_choice: option.visits / total_visits
            for option in options
            if option.visits > 0
        }

    @staticmethod
    def _most_likely_action(policy: Dict[str, float]) -> str:
        if not policy:
            raise ValueError("No legal MCTS action is available for sequence prediction")
        return min(policy, key=lambda action: (-policy[action], action))

    @staticmethod
    def score_prediction(prediction: Dict[str, object], actual_action: str) -> float:
        """Return the probability assigned to a subsequently observed action."""
        probabilities = prediction.get("probabilities")
        if not isinstance(probabilities, dict):
            raise ValueError("Prediction does not include a probability distribution")
        probability = probabilities.get(actual_action, 0.0)
        if not isinstance(probability, (float, int)):
            raise ValueError("Prediction probabilities must be numeric")
        return float(probability)
