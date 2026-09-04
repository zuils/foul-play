"""Loss signature mining using wins as a required control population."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from fp import constants
from fp.battle.helpers import type_effectiveness_modifier
from fp.data import all_move_json
from fp.stratagem.replay.schema import SelfPlayReplay


@dataclass(frozen=True)
class LossSignature:
    """A behavior that appears meaningfully more often in losses than wins."""

    name: str
    loss_support: float
    win_support: float
    lift: float


def mine_loss_signatures(
    replays: Iterable[SelfPlayReplay],
    *,
    min_loss_support: float,
    min_lift: float,
) -> tuple[LossSignature, ...]:
    """Return only signatures supported by both loss and win control populations."""
    if not 0.0 < min_loss_support <= 1.0:
        raise ValueError("min_loss_support must be in (0, 1]")
    if min_lift <= 1.0:
        raise ValueError("min_lift must be greater than 1")
    completed_replays = [
        replay for replay in replays if replay.winner in {"side_one", "side_two"}
    ]
    winners = {replay.winner for replay in completed_replays}
    if winners != {"side_one", "side_two"}:
        raise ValueError(
            "Loss mining requires both side_one and side_two wins as control populations"
        )

    candidates = {"wasted_turn", "immune_move"}
    signatures = []
    for side in ("side_one", "side_two"):
        losses = [replay for replay in completed_replays if replay.winner != side]
        wins = [replay for replay in completed_replays if replay.winner == side]
        for candidate in candidates:
            loss_support = sum(candidate in _replay_signatures(replay, side) for replay in losses) / len(losses)
            win_support = sum(candidate in _replay_signatures(replay, side) for replay in wins) / len(wins)
            lift = float("inf") if win_support == 0.0 else loss_support / win_support
            if loss_support >= min_loss_support and lift >= min_lift:
                signatures.append(
                    LossSignature(
                        name=f"{side}:{candidate}",
                        loss_support=loss_support,
                        win_support=win_support,
                        lift=lift,
                    )
                )
    return tuple(sorted(signatures, key=lambda signature: signature.name))


def _replay_signatures(replay: SelfPlayReplay, side: str) -> set[str]:
    detected = set()
    for turn in replay.turns:
        if side == "side_one":
            observation = turn.side_one_public_observation
            action = turn.side_one_action
            action_values = turn.side_one_mcts_values
            progress = turn.side_one_strategic_progress
        else:
            observation = turn.side_two_public_observation
            action = turn.side_two_action
            action_values = turn.side_two_mcts_values
            progress = turn.side_two_strategic_progress
        if _is_immune_move(observation, action):
            detected.add("immune_move")
        if action_values and progress <= 0.0:
            best_value = max(action_values.values())
            if action_values.get(action, best_value) < best_value:
                detected.add("wasted_turn")
    return detected


def _is_immune_move(observation: dict, action: str) -> bool:
    opponent = observation.get("opponent_active", {})
    move = all_move_json.get(action)
    if not move or move[constants.CATEGORY] == constants.MoveCategory.STATUS:
        return False
    types = opponent.get("types", [])
    if opponent.get("is_terastallized") and opponent.get("tera_type"):
        types = [opponent["tera_type"]]
    return bool(opponent.get("revealed")) and type_effectiveness_modifier(move[constants.TYPE], types) == 0