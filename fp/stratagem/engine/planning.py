"""Action-conditioned future evaluation using poke-engine state transitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
import time
from typing import Dict, List, Sequence, Tuple

from poke_engine import MctsResult, State, generate_instructions, monte_carlo_tree_search

from fp.stratagem.core import Observation
from fp.stratagem.engine.adapter import build_state, mcts_action_to_engine_input


@dataclass(frozen=True)
class CandidatePlanningResult:
    """Normalized values used to rank one current action."""

    action: str
    current_engine_value: float
    conditional_future_value: float
    future_delta: float
    final_value: float
    branch_count: int
    planned: bool
    future_action_count: int = 0
    rollout_count: int = 0
    world_count: int = 0
    allocated_time_ms: float = 0.0
    stop_reason: str = "not planned"

    def to_dict(self) -> Dict[str, float | int | str | bool]:
        return asdict(self)


@dataclass(frozen=True)
class JointActionBranch:
    """One engine-valid simultaneous action outcome with normalized policy mass."""

    side_one_action: str
    side_two_action: str
    transition: object
    probability: float


class ActionConditionedPlanner:
    """Evaluates serious root actions against distributions of future joint actions."""

    def __init__(
        self,
        search_time_ms: int,
        prediction_horizon: int,
        random_seed: int | None = None,
        # Expert/debug parameters - not for normal use
        _future_rollouts: int | None = None,
        _future_action_limit: int | None = None,
        _future_candidate_limit: int | None = None,
    ):
        if search_time_ms <= 0:
            raise ValueError("search_time_ms must be positive")
        if prediction_horizon < 0:
            raise ValueError("prediction_horizon must be non-negative")

        self.search_time_ms = search_time_ms
        self.prediction_horizon = prediction_horizon
        # Expert/debug parameters - secondary to adaptive budgeting
        self._future_rollouts = _future_rollouts
        self._future_action_limit = _future_action_limit
        self._future_candidate_limit = _future_candidate_limit
        self.rng = random.Random(random_seed)
        self._deadline = 0.0
        self._last_action_count = 0
        self._last_rollout_count = 0

    def evaluate(
        self,
        observation: Observation,
        our_team: List[Dict],
        worlds_and_weights: Sequence[Tuple[List[Dict], float]],
    ) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, CandidatePlanningResult]]:
        """Return future-adjusted scores, root visits, and diagnostics by action."""
        if not worlds_and_weights:
            raise ValueError("Action-conditioned planning requires at least one world")

        # Reserve budget: split between root search and future planning
        # Only split if we have sufficient budget (at least 3ms) and prediction horizon > 0
        if self.prediction_horizon > 0 and self.search_time_ms >= 3:
            # Reserve 60% for root search, 40% for future planning
            root_search_budget_ms = int(self.search_time_ms * 0.6)
            future_planning_budget_ms = self.search_time_ms - root_search_budget_ms
        else:
            # No future planning or insufficient budget, use full budget for root search
            root_search_budget_ms = self.search_time_ms
            future_planning_budget_ms = 0

        # Phase 1: Root MCTS searches with reserved time
        self._deadline = time.perf_counter() + root_search_budget_ms / 1000.0

        # Allocate time to worlds based on weights and uncertainty
        world_allocations = self._allocate_time_to_worlds(worlds_and_weights)
        # Scale allocations to match the root search budget
        if root_search_budget_ms < self.search_time_ms:
            world_allocations = [
                ((world, weight), int(allocated_ms * root_search_budget_ms / self.search_time_ms))
                for (world, weight), allocated_ms in world_allocations
            ]

        root_searches = []
        weighted_scores: Dict[str, float] = {}
        action_weights: Dict[str, float] = {}
        visits: Dict[str, int] = {}

        # Process each world with its allocated time
        for (world, world_weight), allocated_time_ms in world_allocations:
            # Check if we've exhausted our root search budget
            if self._budget_exhausted():
                break

            # Temporarily override search_time_ms for this world's MCTS search
            original_search_time_ms = self.search_time_ms
            self.search_time_ms = allocated_time_ms

            try:
                state = build_state(observation, world, our_team)
                result = monte_carlo_tree_search(state, self.search_time_ms)
                root_values, root_visits = self._action_values(result)
                root_searches.append((state, result, world_weight))
                for action, value in root_values.items():
                    weighted_scores[action] = weighted_scores.get(action, 0.0) + value * world_weight
                    action_weights[action] = action_weights.get(action, 0.0) + world_weight
                    visits[action] = visits.get(action, 0) + root_visits[action]
            finally:
                # Restore original search_time_ms
                self.search_time_ms = original_search_time_ms

        immediate_values = {
            action: weighted_scores[action] / action_weights[action]
            for action in weighted_scores
            if action_weights[action] > 0
        }
        serious_actions = self._serious_actions(visits, immediate_values)
        final_values = dict(immediate_values)
        details: Dict[str, CandidatePlanningResult] = {}

        # Phase 2: Future planning with reserved time
        if self.prediction_horizon > 0 and future_planning_budget_ms > 0:
            # Set new deadline for future planning phase
            self._deadline = time.perf_counter() + future_planning_budget_ms / 1000.0

        for action, immediate_value in immediate_values.items():
            if action not in serious_actions or self.prediction_horizon == 0:
                details[action] = CandidatePlanningResult(
                    action=action,
                    current_engine_value=immediate_value,
                    conditional_future_value=immediate_value,
                    future_delta=0.0,
                    final_value=immediate_value,
                    branch_count=0,
                    planned=False,
                    stop_reason="candidate not competitive or horizon is zero",
                )
                continue

            weighted_future_value = 0.0
            future_weight = 0.0
            branch_count = 0
            started = time.perf_counter()
            action_count = 0
            rollout_count = 0
            worlds_planned = 0

            # Process each world for future planning using cached root MCTS results
            # (avoid re-running expensive MCTS by using root_searches cache)
            for state, result, world_weight in root_searches:
                # Check if we've exhausted our total budget
                if self._budget_exhausted():
                    break

                # Use cached root search result - no need to re-run MCTS
                root_values, _ = self._action_values(result)
                if action not in root_values:
                    continue
                future_value, action_branches = self._evaluate_current_action(
                    state, result, action
                )
                if future_value is None:
                    continue
                weighted_future_value += future_value * world_weight
                future_weight += world_weight
                branch_count += action_branches
                action_count += self._last_action_count
                rollout_count += self._last_rollout_count
                worlds_planned += 1

            if future_weight <= 0:
                details[action] = CandidatePlanningResult(
                    action=action,
                    current_engine_value=immediate_value,
                    conditional_future_value=immediate_value,
                    future_delta=0.0,
                    final_value=immediate_value,
                    branch_count=0,
                    planned=False,
                    stop_reason="no valid future transitions",
                )
                continue
            conditional_future_value = weighted_future_value / future_weight
            future_delta = conditional_future_value - immediate_value
            final_value = immediate_value + future_delta
            final_values[action] = final_value
            details[action] = CandidatePlanningResult(
                action=action,
                current_engine_value=immediate_value,
                conditional_future_value=conditional_future_value,
                future_delta=future_delta,
                final_value=final_value,
                branch_count=branch_count,
                planned=True,
                future_action_count=action_count,
                rollout_count=rollout_count,
                world_count=worlds_planned,
                allocated_time_ms=(time.perf_counter() - started) * 1000,
                stop_reason=(
                    "search budget exhausted"
                    if self._budget_exhausted()
                    else "ranking stable or branch support exhausted"
                ),
            )

        return final_values, visits, details

    def _evaluate_current_action(
        self, root_state: State, root_result: MctsResult, action: str
    ) -> Tuple[float | None, int]:
        opponent_policy = self._policy_distribution(root_result.side_two)
        expected_value = 0.0
        branch_count = 0
        action_count = len(opponent_policy)
        rollout_count = 0
        self._last_action_count = action_count
        self._last_rollout_count = 0

        for branch in self._joint_transition_distribution(
            root_state, {action: 1.0}, opponent_policy
        ):
            if self._budget_exhausted():
                break
            rollouts = self._adaptive_rollout_count(branch.probability)
            if rollouts <= 0:
                break
            next_state = self._clone_state(root_state)
            next_state = next_state.apply_instructions(branch.transition)
            values = []
            for _ in range(rollouts):
                if self._budget_exhausted():
                    break
                values.append(self._future_rollout_value(
                    self._clone_state(next_state), self.prediction_horizon - 1
                ))
            if not values:
                break
            expected_value += branch.probability * (sum(values) / len(values))
            branch_count += len(values)
            rollout_count += len(values)

        if branch_count == 0:
            return None, 0
        self._last_rollout_count = rollout_count
        return expected_value, branch_count

    def _future_rollout_value(self, state: State, remaining_turns: int) -> float:
        if remaining_turns <= 0:
            return self._state_value(state)

        result = monte_carlo_tree_search(state, self._remaining_ms())
        user_policy = self._policy_distribution(result.side_one)
        opponent_policy = self._policy_distribution(result.side_two)
        transition_distribution = self._joint_transition_distribution(
            state, user_policy, opponent_policy
        )
        if not transition_distribution:
            return self._state_value(state)
        transition = self.rng.choices(
            [branch for branch in transition_distribution],
            weights=[branch.probability for branch in transition_distribution],
            k=1,
        )[0]
        state = state.apply_instructions(transition.transition)
        return self._future_rollout_value(state, remaining_turns - 1)

    def _state_value(self, state: State) -> float:
        terminal_value = self._terminal_value(state)
        if terminal_value is not None:
            return terminal_value
        result = monte_carlo_tree_search(state, self._remaining_ms())
        total_visits = sum(option.visits for option in result.side_one)
        if total_visits <= 0:
            raise ValueError("MCTS returned no side-one visits for a non-terminal state")
        return sum(option.total_score for option in result.side_one) / total_visits

    @staticmethod
    def _terminal_value(state: State) -> float | None:
        side_one_alive = any(pokemon.hp > 0 for pokemon in state.side_one.pokemon)
        side_two_alive = any(pokemon.hp > 0 for pokemon in state.side_two.pokemon)
        if side_one_alive and side_two_alive:
            return None
        if side_one_alive:
            return 1.0
        if side_two_alive:
            return 0.0
        return 0.5

    def _policy_distribution(self, options) -> Dict[str, float]:
        usable = [option for option in options if option.visits > 0]
        ranked = sorted(usable, key=lambda option: (-option.visits, option.move_choice))
        total_visits = sum(option.visits for option in ranked)
        if total_visits <= 0:
            raise ValueError("MCTS returned no usable policy actions")

        # Retain enough posterior mass to cover uncertainty, with time and debug bounds
        # determining how much of the tail can be afforded.
        target_mass = 1.0 - (1.0 / (total_visits ** 0.5))
        if len(ranked) == 1:
            retained = ranked
        else:
            retained = []
            mass = 0.0
            capacity = max(1, int((self._remaining_ms() or self.search_time_ms) ** 0.5))
            # Expert/debug parameter - secondary to adaptive budgeting
            if self._future_action_limit is not None:
                capacity = min(capacity, self._future_action_limit)
            for option in ranked:
                retained.append(option)
                mass += option.visits / total_visits
                if mass >= target_mass or len(retained) >= capacity:
                    break
            # Preserve a strategically material tail response, even when it is not likely.
            if len(retained) < len(ranked):
                mean_value = sum(option.total_score for option in ranked) / total_visits
                def impact(option) -> float:
                    return (
                        (option.visits / total_visits)
                        * abs(option.total_score / option.visits - mean_value)
                        + abs(option.total_score / option.visits - mean_value)
                        / (option.visits ** 0.5)
                    )
                tail = max(
                    (option for option in ranked if option not in retained), key=impact
                )
                if len(retained) < capacity:
                    retained.append(tail)
                else:
                    weakest = min(retained, key=impact)
                    if impact(tail) > impact(weakest):
                        retained[retained.index(weakest)] = tail
        return {
            option.move_choice: option.visits / total_visits
            for option in retained
        }

    @staticmethod
    def _action_values(result: MctsResult) -> Tuple[Dict[str, float], Dict[str, int]]:
        values: Dict[str, float] = {}
        visits: Dict[str, int] = {}
        for option in result.side_one:
            if option.visits <= 0:
                continue
            values[option.move_choice] = option.total_score / option.visits
            visits[option.move_choice] = option.visits
        return values, visits

    def _serious_actions(
        self, visits: Dict[str, int], values: Dict[str, float] | None = None
    ) -> set[str]:
        if not visits:
            return set()
        ranked = sorted(visits.items(), key=lambda entry: (-entry[1], entry[0]))
        best_visits = ranked[0][1]
        if values:
            best_action = max(values, key=values.get)
            best_value = values[best_action]
            retained = {
                action
                for action, action_visits in ranked
                if action in values
                and values[action] + 0.5 * action_visits ** -0.5
                >= best_value - 0.5 * best_visits ** -0.5
            }
        else:
            retained = {
                action for action, action_visits in ranked
                if action_visits + best_visits ** 0.5 >= best_visits
            }
        capacity = max(1, int((self._remaining_ms() or self.search_time_ms) ** 0.5))
        # Expert/debug parameter - secondary to adaptive budgeting
        if self._future_candidate_limit is not None:
            capacity = min(capacity, self._future_candidate_limit)
        return set(sorted(retained, key=lambda action: (-visits[action], action))[:capacity])

    def _adaptive_rollout_count(self, probability: float) -> int:
        remaining_ms = self._remaining_ms()
        if remaining_ms <= 0:
            return 0
        # Probability-weighted allocation gives uncertain policy mass more samples while
        # allowing the time budget, rather than a fixed repetition count, to set capacity.
        capacity = max(1, remaining_ms // max(1, self.prediction_horizon + 1))
        count = max(1, int(round(capacity * probability)))
        # Expert/debug parameter - secondary to adaptive budgeting
        if self._future_rollouts is not None:
            count = min(count, self._future_rollouts)
        return count

    def _remaining_ms(self) -> int:
        if not self._deadline:
            return self.search_time_ms
        return max(1, int((self._deadline - time.perf_counter()) * 1000))

    def _allocate_time_to_worlds(self, worlds_and_weights: Sequence[Tuple[List[Dict], float]]) -> List[Tuple[Tuple[List[Dict], float], int]]:
        """Allocate search time to worlds based on posterior weights and uncertainty.

        Returns a list of ((world, weight), allocated_time_ms) tuples.
        """
        if not worlds_and_weights:
            return []

        # Calculate total weight (only positive weights)
        total_weight = sum(max(0.0, weight) for _, weight in worlds_and_weights)
        if total_weight <= 0:
            # If all weights are zero or negative, allocate equally
            equal_time = self.search_time_ms // len(worlds_and_weights)
            return [((world, weight), equal_time) for world, weight in worlds_and_weights]

        # Base allocation proportional to weight
        allocations = []
        allocated_time_sum = 0

        # First pass: allocate proportional to weight, ensuring minimum time per world
        min_time_per_world = max(1, self.search_time_ms // (len(worlds_and_weights) * 10))  # At least 1/10th of equal share

        for world, weight in worlds_and_weights:
            # Only allocate time to worlds with positive weight
            if weight > 0:
                # Proportional allocation
                proportional_time = int((weight / total_weight) * self.search_time_ms)
                # Ensure minimum time
                allocated_time = max(min_time_per_world, proportional_time)
            else:
                # Zero or negative weight gets minimum time only
                allocated_time = min_time_per_world
            allocations.append(((world, weight), allocated_time))
            allocated_time_sum += allocated_time

        # Second pass: if we haven't used all time, distribute the remainder
        if allocated_time_sum < self.search_time_ms:
            remaining_time = self.search_time_ms - allocated_time_sum

            # Distribute remaining time proportional to weight (only for positive weight worlds)
            if total_weight > 0:
                for i, ((world, weight), allocated_time) in enumerate(allocations):
                    if weight > 0:
                        additional_time = int((weight / total_weight) * remaining_time)
                        allocations[i] = ((world, weight), allocated_time + additional_time)
                    # Zero weight worlds don't get additional time in this pass

            # Handle any remaining time due to rounding errors
            # Allocate to the world with highest weight
            if remaining_time > 0:
                # Find world with maximum weight
                max_weight_idx = max(
                    range(len(worlds_and_weights)),
                    key=lambda i: worlds_and_weights[i][1]
                )
                world, weight = worlds_and_weights[max_weight_idx]
                # Find this world in allocations
                for i, ((alloc_world, alloc_weight), _) in enumerate(allocations):
                    if alloc_world is world and alloc_weight == weight:
                        allocations[i] = ((alloc_world, alloc_weight), allocations[i][1] + remaining_time)
                        break

        # Final check: ensure we don't exceed the budget (adjust if necessary due to rounding)
        total_allocated = sum(allocated_time for _, allocated_time in allocations)
        if total_allocated > self.search_time_ms:
            # Scale down proportionally if we exceeded the budget
            scale_factor = self.search_time_ms / total_allocated
            allocations = [
                ((world, weight), max(1, int(allocated_time * scale_factor)))
                for (world, weight), allocated_time in allocations
            ]
        elif total_allocated < self.search_time_ms:
            # Distribute any remaining time to the highest weight world
            remaining_time = self.search_time_ms - total_allocated
            if remaining_time > 0:
                # Find world with maximum weight
                max_weight_idx = max(
                    range(len(worlds_and_weights)),
                    key=lambda i: worlds_and_weights[i][1]
                )
                world, weight = worlds_and_weights[max_weight_idx]
                # Find this world in allocations
                for i, ((alloc_world, alloc_weight), _) in enumerate(allocations):
                    if alloc_world is world and alloc_weight == weight:
                        allocations[i] = ((alloc_world, alloc_weight), allocations[i][1] + remaining_time)
                        break

        return allocations

    def _budget_exhausted(self) -> bool:
        return bool(self._deadline and time.perf_counter() >= self._deadline)

    def _joint_transition_distribution(
        self,
        state: State,
        user_policy: Dict[str, float],
        opponent_policy: Dict[str, float],
    ) -> List[JointActionBranch]:
        weighted_transitions = []
        for user_action, user_probability in user_policy.items():
            for opponent_action, opponent_probability in opponent_policy.items():
                transitions = self._engine_transitions(
                    state, user_action, opponent_action
                )
                for transition, transition_probability in self._transition_probabilities(
                    transitions
                ):
                    weighted_transitions.append(
                        JointActionBranch(
                            side_one_action=user_action,
                            side_two_action=opponent_action,
                            transition=transition,
                            probability=(
                                user_probability
                                * opponent_probability
                                * transition_probability
                            ),
                        )
                    )
        total_probability = sum(branch.probability for branch in weighted_transitions)
        if total_probability <= 0:
            return []
        return [
            JointActionBranch(
                side_one_action=branch.side_one_action,
                side_two_action=branch.side_two_action,
                transition=branch.transition,
                probability=branch.probability / total_probability,
            )
            for branch in weighted_transitions
        ]

    @staticmethod
    def _engine_transitions(
        state: State, user_action: str, opponent_action: str
    ):
        try:
            return generate_instructions(
                state,
                mcts_action_to_engine_input(user_action),
                mcts_action_to_engine_input(opponent_action),
            )
        except ValueError as error:
            if str(error).startswith("Invalid move"):
                return []
            raise

    @staticmethod
    def _transition_probabilities(transitions) -> List[Tuple[object, float]]:
        valid_transitions = [transition for transition in transitions if transition.percentage > 0]
        total_probability = sum(transition.percentage for transition in valid_transitions)
        if total_probability <= 0:
            return []
        return [
            (transition, transition.percentage / total_probability)
            for transition in valid_transitions
        ]

    @staticmethod
    def _clone_state(state: State) -> State:
        return State.from_string(state.to_string())