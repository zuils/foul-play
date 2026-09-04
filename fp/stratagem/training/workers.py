"""Isolated parallel local self-play workers with central learner aggregation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import random
from typing import TYPE_CHECKING, Dict, List, Sequence

from .self_play import LocalSelfPlayGame, MctsSelfPlayAgent, SelfPlayResult

if TYPE_CHECKING:
    from fp.stratagem.learning.trainer import StratagemTrainer


@dataclass(frozen=True)
class SelfPlayBatch:
    """Ordered worker results and their independent deterministic seeds."""

    results: tuple[SelfPlayResult, ...]
    seeds: tuple[int, ...]

    def add_to_trainers(
        self,
        side_one_trainer: "StratagemTrainer | None" = None,
        side_two_trainer: "StratagemTrainer | None" = None,
    ) -> None:
        """Serially add completed-worker experiences to central trainer instances."""
        for result in self.results:
            if side_one_trainer is not None:
                result.add_to_trainer(side_one_trainer, side_one=True)
            if side_two_trainer is not None:
                result.add_to_trainer(side_two_trainer, side_one=False)


def run_parallel_self_play(
    side_one_team: List[Dict],
    side_two_team: List[Dict],
    side_one_candidate_worlds: Sequence[List[Dict]],
    side_two_candidate_worlds: Sequence[List[Dict]],
    *,
    games: int,
    parallel_games: int,
    max_turns: int,
    search_time_ms: int,
    seed: int | None = None,
    side_one_learned_model=None,
    side_two_learned_model=None,
) -> SelfPlayBatch:
    """Run independent local games without sharing engine state or agent beliefs.

    Candidate worlds are explicit inputs owned by each agent; the true opposite team
    is intentionally not injected as an agent belief by this coordinator.
    """
    if games <= 0:
        raise ValueError("games must be positive")
    if parallel_games <= 0:
        raise ValueError("parallel_games must be positive")
    if max_turns <= 0 or max_turns > 100:
        raise ValueError("max_turns must be between 1 and 100")
    if search_time_ms <= 0:
        raise ValueError("search_time_ms must be positive")
    if not side_one_candidate_worlds or not side_two_candidate_worlds:
        raise ValueError("Each self-play side requires at least one candidate world")

    seeds = _worker_seeds(games, seed)
    with ThreadPoolExecutor(max_workers=min(parallel_games, games)) as executor:
        futures = [
            executor.submit(
                _run_one_game,
                side_one_team,
                side_two_team,
                side_one_candidate_worlds,
                side_two_candidate_worlds,
                max_turns,
                search_time_ms,
                worker_seed,
                side_one_learned_model,
                side_two_learned_model,
            )
            for worker_seed in seeds
        ]
        results = tuple(future.result() for future in futures)
    return SelfPlayBatch(results=results, seeds=seeds)


def _run_one_game(
    side_one_team: List[Dict],
    side_two_team: List[Dict],
    side_one_candidate_worlds: Sequence[List[Dict]],
    side_two_candidate_worlds: Sequence[List[Dict]],
    max_turns: int,
    search_time_ms: int,
    seed: int,
    side_one_learned_model,
    side_two_learned_model,
) -> SelfPlayResult:
    side_one_agent = MctsSelfPlayAgent(
        side_one_team,
        side_one_candidate_worlds,
        search_time_ms=search_time_ms,
        learned_model=side_one_learned_model,
        random_seed=seed,
    )
    side_two_agent = MctsSelfPlayAgent(
        side_two_team,
        side_two_candidate_worlds,
        search_time_ms=search_time_ms,
        learned_model=side_two_learned_model,
        random_seed=seed + 1,
    )
    return LocalSelfPlayGame.from_teams(
        side_one_team,
        side_two_team,
        side_one_agent,
        side_two_agent,
        max_turns=max_turns,
        random_seed=seed + 2,
    ).run()


def _worker_seeds(games: int, seed: int | None) -> tuple[int, ...]:
    rng = random.Random(seed)
    return tuple(rng.randrange(0, 2**63) for _ in range(games))