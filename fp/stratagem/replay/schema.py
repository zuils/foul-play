"""Versioned JSON replay schema for offline Stratagem self-play analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Dict, List, Sequence

if TYPE_CHECKING:
    from fp.stratagem.training.self_play import SelfPlayResult


REPLAY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ReplayTurn:
    """One serialized public turn plus private decision diagnostics."""

    turn: int
    side_one_public_observation: Dict
    side_two_public_observation: Dict
    side_one_action: str
    side_two_action: str
    side_one_mcts_values: Dict[str, float]
    side_two_mcts_values: Dict[str, float]
    side_one_strategic_progress: float
    side_two_strategic_progress: float


@dataclass(frozen=True)
class SelfPlayReplay:
    """Offline-only replay record with public traces and simulator truth data."""

    schema_version: int
    format: str
    seed: int | None
    winner: str | None
    turn_count: int
    reached_turn_cap: bool
    turns: tuple[ReplayTurn, ...]
    true_teams: Dict[str, List[Dict]]
    hidden_worlds: Dict[str, Sequence[List[Dict]]] | None = None

    @classmethod
    def from_result(
        cls,
        result: "SelfPlayResult",
        *,
        format_name: str,
        seed: int | None,
        true_teams: Dict[str, List[Dict]],
        hidden_worlds: Dict[str, Sequence[List[Dict]]] | None = None,
    ) -> "SelfPlayReplay":
        """Create an offline replay from a completed local game result."""
        if set(true_teams) != {"side_one", "side_two"}:
            raise ValueError("Replay true_teams must include side_one and side_two")
        if result.turns != len(result.turns_data):
            raise ValueError("Self-play result turn count does not match its turn data")
        turns = tuple(
            ReplayTurn(
                turn=index,
                side_one_public_observation=turn.side_one_observation.to_dict(),
                side_two_public_observation=turn.side_two_observation.to_dict(),
                side_one_action=turn.side_one_decision.action,
                side_two_action=turn.side_two_decision.action,
                side_one_mcts_values=dict(turn.side_one_decision.action_values),
                side_two_mcts_values=dict(turn.side_two_decision.action_values),
                side_one_strategic_progress=turn.side_one_strategic_progress,
                side_two_strategic_progress=turn.side_two_strategic_progress,
            )
            for index, turn in enumerate(result.turns_data, start=1)
        )
        return cls(
            schema_version=REPLAY_SCHEMA_VERSION,
            format=format_name,
            seed=seed,
            winner=result.winner,
            turn_count=result.turns,
            reached_turn_cap=result.reached_turn_cap,
            turns=turns,
            true_teams={side: list(team) for side, team in true_teams.items()},
            hidden_worlds=hidden_worlds,
        )

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict) -> "SelfPlayReplay":
        expected_keys = {
            "schema_version", "format", "seed", "winner", "turn_count",
            "reached_turn_cap", "turns", "true_teams", "hidden_worlds",
        }
        if set(value) != expected_keys:
            raise ValueError("Replay has an incompatible schema")
        if value["schema_version"] != REPLAY_SCHEMA_VERSION:
            raise ValueError("Replay schema version is not supported")
        if not isinstance(value["turns"], list):
            raise ValueError("Replay turns must be a list")
        return cls(
            schema_version=value["schema_version"],
            format=value["format"],
            seed=value["seed"],
            winner=value["winner"],
            turn_count=value["turn_count"],
            reached_turn_cap=value["reached_turn_cap"],
            turns=tuple(ReplayTurn(**turn) for turn in value["turns"]),
            true_teams=value["true_teams"],
            hidden_worlds=value["hidden_worlds"],
        )