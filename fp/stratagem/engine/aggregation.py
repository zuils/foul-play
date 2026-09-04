"""Belief-weighted MCTS aggregation with optional learned action priors."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from fp.stratagem.config import CONFIG
from fp.stratagem.core.actions import StrategicActionSelector
from fp.stratagem.engine.planning import ActionConditionedPlanner, CandidatePlanningResult
from fp.stratagem.engine.search import search_world_with_visits
from fp.stratagem.inference.belief import Belief

if TYPE_CHECKING:
    from fp.stratagem.learning.model import StratagemModelWrapper


class WorldAggregator:
    """Aggregate MCTS scores across belief worlds, optionally using trained priors."""

    def __init__(
        self, belief: Belief, learned_model: StratagemModelWrapper | None = None
    ) -> None:
        self.belief = belief
        self.learned_model = learned_model

    def aggregate_worlds(
        self,
        observation,
        our_team,
        search_time_ms: int = 100,
        prediction_horizon: Optional[int] = None,
    ) -> Dict[str, float]:
        scores, _ = self.aggregate_worlds_with_visits(
            observation, our_team, search_time_ms, prediction_horizon
        )
        return scores

    def aggregate_worlds_with_visits(
        self,
        observation,
        our_team,
        search_time_ms: int = 100,
        prediction_horizon: Optional[int] = None,
        future_rollouts: Optional[int] = None,
        future_action_limit: Optional[int] = None,
        future_candidate_limit: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> Tuple[Dict[str, float], Dict[str, int]]:
        scores, visits, _ = self.aggregate_worlds_with_details(
            observation,
            our_team,
            search_time_ms,
            prediction_horizon,
            future_rollouts,
            future_action_limit,
            future_candidate_limit,
            random_seed,
        )
        return scores, visits

    def aggregate_worlds_with_details(
        self,
        observation,
        our_team,
        search_time_ms: int = 100,
        prediction_horizon: Optional[int] = None,
        future_rollouts: Optional[int] = None,
        future_action_limit: Optional[int] = None,
        future_candidate_limit: Optional[int] = None,
        random_seed: Optional[int] = None,
        use_learned_model: bool = True,
        learned_model_weight: float = 0.1,
    ) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, CandidatePlanningResult]]:
        """Return belief-weighted MCTS candidate scores and diagnostics."""
        if not 0.0 <= learned_model_weight <= 1.0:
            raise ValueError("learned_model_weight must be in [0, 1]")
        worlds_and_weights = self._unique_weighted_worlds()
        if not worlds_and_weights:
            raise ValueError("Cannot aggregate an uninitialized belief state")

        resolved_horizon = CONFIG.prediction_horizon if prediction_horizon is None else prediction_horizon
        if resolved_horizon < 0:
            raise ValueError("prediction_horizon must be non-negative")
        if resolved_horizon > 0:
            planner = ActionConditionedPlanner(
                search_time_ms=search_time_ms,
                prediction_horizon=resolved_horizon,
                random_seed=random_seed if random_seed is not None else CONFIG.seed,
                _future_rollouts=future_rollouts,
                _future_action_limit=future_action_limit,
                _future_candidate_limit=future_candidate_limit,
            )
            final_values, visits, details = planner.evaluate(observation, our_team, worlds_and_weights)
            final_values = self._blend_learned_priors(
                observation, final_values, use_learned_model, learned_model_weight
            )
            if self.learned_model is not None and use_learned_model:
                details = {
                    action: replace(
                        detail,
                        final_value=final_values[action],
                        future_delta=final_values[action] - detail.current_engine_value,
                    )
                    for action, detail in details.items()
                }
            strategic_selector = StrategicActionSelector()
            guarded_scores = strategic_selector.legal_actions(observation, final_values)
            filtered_visits = {action: value for action, value in visits.items() if action in guarded_scores}
            filtered_details = {action: value for action, value in details.items() if action in guarded_scores}
            return guarded_scores, filtered_visits, filtered_details

        return self._aggregate_root_worlds(
            worlds_and_weights,
            observation,
            our_team,
            search_time_ms,
            use_learned_model,
            learned_model_weight,
        )

    def _aggregate_root_worlds(
        self,
        worlds_and_weights: List[Tuple[List[Dict], float]],
        observation,
        our_team,
        search_time_ms: int,
        use_learned_model: bool,
        learned_model_weight: float,
    ) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, CandidatePlanningResult]]:
        weighted_scores: Dict[str, float] = {}
        action_weights: Dict[str, float] = {}
        aggregated_visits: Dict[str, int] = {}
        for world, weight in worlds_and_weights:
            world_scores, world_visits = search_world_with_visits(observation, world, our_team, search_time_ms)
            for action, score in world_scores.items():
                weighted_scores[action] = weighted_scores.get(action, 0.0) + score * weight
                action_weights[action] = action_weights.get(action, 0.0) + weight
                aggregated_visits[action] = aggregated_visits.get(action, 0) + world_visits.get(action, 0)
        scores = {
            action: weighted_scores[action] / action_weights[action]
            for action in weighted_scores
            if action_weights[action] > 0
        }
        scores = self._blend_learned_priors(
            observation, scores, use_learned_model, learned_model_weight
        )
        details = {
            action: CandidatePlanningResult(
                action=action,
                current_engine_value=score,
                conditional_future_value=score,
                future_delta=0.0,
                final_value=score,
                branch_count=0,
                planned=False,
            )
            for action, score in scores.items()
        }
        return scores, aggregated_visits, details

    def _blend_learned_priors(
        self,
        observation,
        scores: Dict[str, float],
        use_learned_model: bool,
        learned_model_weight: float,
    ) -> Dict[str, float]:
        if not scores or self.learned_model is None or not use_learned_model:
            return scores
        actions = list(scores)
        priors = self.learned_model.get_candidate_action_priors(observation, actions)
        if set(priors) != set(actions):
            raise ValueError("Learned model priors must cover exactly the MCTS candidates")
        maximum_score = max(scores.values())
        exp_scores = {action: math.exp(score - maximum_score) for action, score in scores.items()}
        total_score = sum(exp_scores.values())
        mcts_probabilities = {
            action: exp_score / total_score for action, exp_score in exp_scores.items()
        }
        return {
            action: (1.0 - learned_model_weight) * mcts_probabilities[action]
            + learned_model_weight * priors[action]
            for action in actions
        }

    def _unique_weighted_worlds(self) -> List[Tuple[List[Dict], float]]:
        worlds = self.belief.worlds
        weights = self.belief.get_world_weights()
        if len(worlds) != len(weights):
            raise ValueError("Belief world and weight counts must match")
        unique_worlds: Dict[str, Tuple[List[Dict], float]] = {}
        for world, weight in zip(worlds, weights):
            world_key = json.dumps(world, sort_keys=True, separators=(",", ":"))
            if world_key in unique_worlds:
                saved_world, saved_weight = unique_worlds[world_key]
                unique_worlds[world_key] = (saved_world, saved_weight + weight)
            else:
                unique_worlds[world_key] = (world, weight)
        return list(unique_worlds.values())


def aggregate_worlds_simple(
    belief: Belief,
    observation,
    our_team,
    search_time_ms: int = 100,
    prediction_horizon: Optional[int] = None,
    learned_model: StratagemModelWrapper | None = None,
) -> Dict[str, float]:
    return WorldAggregator(belief, learned_model).aggregate_worlds(
        observation, our_team, search_time_ms, prediction_horizon
    )
