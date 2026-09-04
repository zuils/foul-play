"""
Search adapter for running poke-engine MCTS per hidden world.
"""
from __future__ import annotations

from typing import Dict, List, Tuple
from fp.stratagem.core import Observation
from fp.stratagem.engine.adapter import build_state
from poke_engine import monte_carlo_tree_search, MctsResult, State as PokeEngineState


def _state_to_string(state: PokeEngineState) -> str:
    """Convert PokeEngineState to its string representation."""
    return state.to_string()


def search_world(
    observation: Observation,
    hidden_team: List[Dict],
    our_team: List[Dict],
    search_time_ms: int = 100,
) -> Dict[str, float]:
    """
    Run MCTS for a single hidden world and return action values.

    Args:
        observation: Public observation state.
        hidden_team: Full opponent team hypothesis (list of 6 dicts).
        our_team: Full own team specification (list of 6 dicts).
        search_time_ms: MCTS search time in milliseconds.

    Returns:
        Dictionary mapping action name to average score (or visit proportion).
    """
    scores, _ = search_world_with_visits(
        observation, hidden_team, our_team, search_time_ms
    )
    return scores


def search_world_with_visits(
    observation: Observation,
    hidden_team: List[Dict],
    our_team: List[Dict],
    search_time_ms: int = 100,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Run MCTS for one world and return per-action scores and real visits."""
    state = build_state(observation, hidden_team, our_team)
    result: MctsResult = monte_carlo_tree_search(state, search_time_ms)
    action_values: Dict[str, float] = {}
    action_visits: Dict[str, int] = {}
    for action in result.side_one:
        move_name = action.move_choice
        action_visits[move_name] = action.visits
        if action.visits > 0:
            avg_score = action.total_score / action.visits
        else:
            avg_score = 0.0
        action_values[move_name] = avg_score
    return action_values, action_visits


def search_multiple_worlds(
    observation: Observation,
    team_hypotheses: List[List[Dict]],
    our_team: List[Dict],
    search_time_ms: int = 100,
) -> List[Dict[str, float]]:
    """
    Run MCTS for multiple world hypotheses.

    Args:
        observation: Public observation state.
        team_hypotheses: List of opponent team hypotheses (each a list of 6 dicts).
        our_team: Full own team specification.
        search_time_ms: MCTS search time per world.

    Returns:
        List of action-value dictionaries, one per world.
    """
    results = []
    for hyp in team_hypotheses:
        vals = search_world(observation, hyp, our_team, search_time_ms)
        results.append(vals)
    return results

