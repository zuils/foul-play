"""Engine-only local self-play with private, public-observation agent decisions."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import TYPE_CHECKING, Dict, List, Protocol, Sequence

from poke_engine import State, generate_instructions

from fp import constants
from fp.stratagem.config import CONFIG
from fp.stratagem.core import Observation
from fp.stratagem.core.actions import StrategicActionSelector
from fp.stratagem.engine.adapter import build_state, mcts_action_to_engine_input
from fp.stratagem.engine.aggregation import WorldAggregator
from fp.stratagem.inference import Belief
from fp.stratagem.learning.rewards import RewardBreakdown, RewardBuilder, RewardSignals

if TYPE_CHECKING:
    from fp.stratagem.learning.trainer import StratagemTrainer


class SelfPlayAgent(Protocol):
    """An agent that selects from a public observation without simulator access."""

    def select_action(self, observation: Observation) -> str: ...


@dataclass(frozen=True)
class SelfPlayDecision:
    """An agent action and its own MCTS evidence, never revealed to the opponent."""

    action: str
    action_values: Dict[str, float]


@dataclass(frozen=True)
class SelfPlayTurn:
    """A public pre-turn snapshot and post-turn strategic result for offline learning."""

    side_one_observation: Observation
    side_two_observation: Observation
    side_one_decision: SelfPlayDecision
    side_two_decision: SelfPlayDecision
    side_one_strategic_progress: float
    side_two_strategic_progress: float


@dataclass(frozen=True)
class SelfPlayResult:
    """Offline simulator output. `final_state` is never exposed to agents."""

    winner: str | None
    turns: int
    reached_turn_cap: bool
    actions: tuple[tuple[str, str], ...]
    turns_data: tuple[SelfPlayTurn, ...]
    final_state: State

    def add_to_trainer(
        self, trainer: "StratagemTrainer", *, side_one: bool
    ) -> tuple[RewardBreakdown, ...]:
        """Add one side's completed local-game experience to a trainer."""
        terminal_outcome = self._terminal_outcome(side_one)
        rewards = []
        for index, turn in enumerate(self.turns_data):
            observation = turn.side_one_observation if side_one else turn.side_two_observation
            decision = turn.side_one_decision if side_one else turn.side_two_decision
            strategic_progress = (
                turn.side_one_strategic_progress
                if side_one
                else turn.side_two_strategic_progress
            )
            reward = RewardBuilder.build(
                RewardSignals(
                    terminal_outcome=terminal_outcome,
                    strategic_progress=strategic_progress,
                    engine_action_value=decision.action_values.get(decision.action),
                )
            )
            next_observation = None
            if index + 1 < len(self.turns_data):
                next_turn = self.turns_data[index + 1]
                next_observation = (
                    next_turn.side_one_observation
                    if side_one
                    else next_turn.side_two_observation
                )
            trainer.add_experience(
                observation,
                decision.action,
                reward.total,
                next_observation,
                done=index == len(self.turns_data) - 1,
                action_values=decision.action_values,
            )
            rewards.append(reward)
        return tuple(rewards)

    def _terminal_outcome(self, side_one: bool) -> float:
        if self.winner == "draw" or self.winner is None:
            return 0.0
        if self.winner == ("side_one" if side_one else "side_two"):
            return 1.0
        return -1.0


class MctsSelfPlayAgent:
    """Select actions from a private belief and public Observation only."""

    def __init__(
        self,
        own_team: List[Dict],
        candidate_worlds: Sequence[List[Dict]],
        *,
        search_time_ms: int = 1,
        learned_model=None,
        random_seed: int | None = None,
    ) -> None:
        if not own_team or not candidate_worlds:
            raise ValueError("Self-play agents require a team and at least one candidate world")
        self.own_team = own_team
        self.belief = Belief(team_sampler=None, world_count=len(candidate_worlds), random_seed=random_seed)
        self.belief.worlds = [list(world) for world in candidate_worlds]
        self.belief.weights = [0.0] * len(candidate_worlds)
        self.search_time_ms = search_time_ms
        self.learned_model = learned_model
        self.selector = StrategicActionSelector(random_seed=random_seed)

    def select_action(self, observation: Observation) -> str:
        return self.select_decision(observation).action

    def select_decision(self, observation: Observation) -> SelfPlayDecision:
        """Return a move and the private MCTS values used to select it."""
        self.belief.update_with_evidence(observation)
        scores = WorldAggregator(self.belief, self.learned_model).aggregate_worlds(
            observation,
            self.own_team,
            search_time_ms=self.search_time_ms,
            prediction_horizon=0,
        )
        legal_scores = self.selector.legal_actions(observation, scores)
        return SelfPlayDecision(
            action=self.selector.select_action(observation, legal_scores),
            action_values=legal_scores,
        )


class LocalSelfPlayGame:
    """Resolve a local game while keeping true engine state outside both agents."""

    _SIDE_CONDITIONS = (
        "aurora_veil", "crafty_shield", "healing_wish", "light_screen",
        "lucky_chant", "lunar_dance", "mat_block", "mist", "protect",
        "quick_guard", "reflect", "safeguard", "spikes", "stealth_rock",
        "sticky_web", "tailwind", "toxic_count", "toxic_spikes", "wide_guard",
    )

    def __init__(
        self,
        state: State,
        side_one_agent: SelfPlayAgent,
        side_two_agent: SelfPlayAgent,
        *,
        max_turns: int = CONFIG.max_turns,
        random_seed: int | None = None,
    ) -> None:
        if not 1 <= max_turns <= 100:
            raise ValueError("max_turns must be between 1 and 100")
        self._state = state
        self._side_one_agent = side_one_agent
        self._side_two_agent = side_two_agent
        self.max_turns = max_turns
        self.rng = random.Random(random_seed)
        self._seen_side_one_moves: Dict[str, set[str]] = {}
        self._seen_side_two_moves: Dict[str, set[str]] = {}

    @classmethod
    def from_teams(
        cls,
        side_one_team: List[Dict],
        side_two_team: List[Dict],
        side_one_agent: SelfPlayAgent,
        side_two_agent: SelfPlayAgent,
        **kwargs,
    ) -> "LocalSelfPlayGame":
        if len(side_one_team) != 6 or len(side_two_team) != 6:
            raise ValueError("Local self-play requires exactly six Pokemon per team")
        initial_observation = Observation.from_public_snapshot(
            player="user",
            active_pokemon={"name": side_one_team[0]["species"]},
            reserve_pokemon=[],
            hidden_reserve_count=5,
            opponent_active={"name": side_two_team[0]["species"]},
            opponent_reserve_revealed=[],
            opponent_hidden_reserve_count=5,
        )
        return cls(
            build_state(initial_observation, side_two_team, side_one_team),
            side_one_agent,
            side_two_agent,
            **kwargs,
        )

    def run(
        self,
        side_one_trainer: "StratagemTrainer | None" = None,
        side_two_trainer: "StratagemTrainer | None" = None,
    ) -> SelfPlayResult:
        actions: list[tuple[str, str]] = []
        turns_data: list[SelfPlayTurn] = []
        for turn in range(1, self.max_turns + 1):
            winner = self._winner()
            if winner is not None:
                return self._complete_result(
                    winner, turn - 1, False, actions, turns_data, side_one_trainer, side_two_trainer
                )

            side_one_observation = self._public_observation(side_one=True, turn=turn)
            side_two_observation = self._public_observation(side_one=False, turn=turn)
            starting_progress = self._health_differential()
            side_one_decision = self._select_decision(
                self._side_one_agent, side_one_observation
            )
            side_two_decision = self._select_decision(
                self._side_two_agent, side_two_observation
            )
            transitions = generate_instructions(
                self._state,
                mcts_action_to_engine_input(side_one_decision.action),
                mcts_action_to_engine_input(side_two_decision.action),
            )
            valid_transitions = [transition for transition in transitions if transition.percentage > 0]
            if not valid_transitions:
                raise ValueError("Selected simultaneous actions have no engine-valid transition")
            transition = self.rng.choices(
                valid_transitions,
                weights=[instruction.percentage for instruction in valid_transitions],
                k=1,
            )[0]
            self._record_revealed_moves(
                side_one_decision.action, side_two_decision.action
            )
            self._state = self._state.apply_instructions(transition)
            ending_progress = self._health_differential()
            actions.append((side_one_decision.action, side_two_decision.action))
            turns_data.append(
                SelfPlayTurn(
                    side_one_observation=side_one_observation,
                    side_two_observation=side_two_observation,
                    side_one_decision=side_one_decision,
                    side_two_decision=side_two_decision,
                    side_one_strategic_progress=(ending_progress - starting_progress) / 2.0,
                    side_two_strategic_progress=(starting_progress - ending_progress) / 2.0,
                )
            )

        return self._complete_result(
            self._winner(), self.max_turns, True, actions, turns_data, side_one_trainer, side_two_trainer
        )

    @staticmethod
    def _select_decision(agent: SelfPlayAgent, observation: Observation) -> SelfPlayDecision:
        decision_method = getattr(agent, "select_decision", None)
        if callable(decision_method):
            decision = decision_method(observation)
            if not isinstance(decision, SelfPlayDecision):
                raise TypeError("Self-play select_decision must return SelfPlayDecision")
            return decision
        return SelfPlayDecision(action=agent.select_action(observation), action_values={})

    def _complete_result(
        self,
        winner: str | None,
        turns: int,
        reached_turn_cap: bool,
        actions: list[tuple[str, str]],
        turns_data: list[SelfPlayTurn],
        side_one_trainer: "StratagemTrainer | None",
        side_two_trainer: "StratagemTrainer | None",
    ) -> SelfPlayResult:
        result = SelfPlayResult(
            winner, turns, reached_turn_cap, tuple(actions), tuple(turns_data), self._state
        )
        if side_one_trainer is not None:
            result.add_to_trainer(side_one_trainer, side_one=True)
        if side_two_trainer is not None:
            result.add_to_trainer(side_two_trainer, side_one=False)
        return result

    def _winner(self) -> str | None:
        side_one_alive = any(pokemon.hp > 0 for pokemon in self._state.side_one.pokemon)
        side_two_alive = any(pokemon.hp > 0 for pokemon in self._state.side_two.pokemon)
        if side_one_alive and side_two_alive:
            return None
        if side_one_alive:
            return "side_one"
        if side_two_alive:
            return "side_two"
        return "draw"

    def _health_differential(self) -> float:
        side_one_health = self._team_health_fraction(self._state.side_one.pokemon)
        side_two_health = self._team_health_fraction(self._state.side_two.pokemon)
        return side_one_health - side_two_health

    @staticmethod
    def _team_health_fraction(pokemon) -> float:
        battle_pokemon = [member for member in pokemon if member.id != "none"]
        if not battle_pokemon:
            raise ValueError("A local self-play side must contain at least one Pokemon")
        return sum(member.hp / member.maxhp for member in battle_pokemon) / len(battle_pokemon)

    def _public_observation(self, *, side_one: bool, turn: int) -> Observation:
        own_side = self._state.side_one if side_one else self._state.side_two
        opponent_side = self._state.side_two if side_one else self._state.side_one
        seen_moves = self._seen_side_two_moves if side_one else self._seen_side_one_moves
        own_active_pokemon = self._active_pokemon(own_side)
        opponent_active_pokemon = self._active_pokemon(opponent_side)
        own_active_index = self._active_index(own_side)
        opponent_active_index = self._active_index(opponent_side)
        own_active = self._pokemon_snapshot(
            own_active_pokemon, private=True, seen_moves=set()
        )
        own_reserves = [
            self._pokemon_snapshot(pokemon, private=True, seen_moves=set())
            for index, pokemon in enumerate(own_side.pokemon)
            if index != own_active_index and pokemon.id != "none"
        ]
        opponent_active = self._pokemon_snapshot(
            opponent_active_pokemon,
            private=False,
            seen_moves=seen_moves.get(opponent_active_pokemon.id, set()),
        )
        return Observation.from_public_snapshot(
            player="user" if side_one else "opponent",
            active_pokemon=own_active,
            reserve_pokemon=own_reserves,
            hidden_reserve_count=0,
            opponent_active=opponent_active,
            opponent_reserve_revealed=[],
            opponent_hidden_reserve_count=sum(
                pokemon.id != "none"
                for index, pokemon in enumerate(opponent_side.pokemon)
                if index != opponent_active_index
            ),
            weather=str(self._state.weather),
            weather_turns_remaining=self._state.weather_turns_remaining,
            field=str(self._state.terrain),
            field_turns_remaining=self._state.terrain_turns_remaining,
            trick_room=self._state.trick_room,
            trick_room_turns_remaining=self._state.trick_room_turns_remaining,
            side_conditions=self._side_conditions(own_side),
            opponent_side_conditions=self._side_conditions(opponent_side),
            turn=turn,
            available_moves=[
                move.id for move in own_active_pokemon.moves if not move.disabled
            ],
            disabled_moves=[
                move.id for move in own_active_pokemon.moves if move.disabled
            ],
        )

    @staticmethod
    def _active_index(side) -> int:
        return int(str(side.active_index))

    @classmethod
    def _active_pokemon(cls, side):
        return side.pokemon[cls._active_index(side)]

    def _pokemon_snapshot(self, pokemon, *, private: bool, seen_moves: set[str]) -> Dict:
        moves = [move.id for move in pokemon.moves if private or move.id in seen_moves]
        move_pp = [
            move.pp for move in pokemon.moves if private or move.id in seen_moves
        ]
        snapshot = {
            "name": pokemon.id,
            "level": pokemon.level,
            "revealed": True,
            "is_alive": pokemon.hp > 0,
            "hp": pokemon.hp,
            "max_hp": pokemon.maxhp,
            "hp_fraction": pokemon.hp / pokemon.maxhp if pokemon.maxhp else 0.0,
            "status": str(pokemon.status).lower() if str(pokemon.status) != "None" else None,
            "types": list(pokemon.types),
            "moves": moves,
            "move_pp": move_pp,
            "disabled_moves": [move.id for move in pokemon.moves if move.disabled and private],
            "is_mega": pokemon.mega_evolved,
            "is_terastallized": pokemon.terastallized,
            "tera_type": pokemon.tera_type if pokemon.terastallized else None,
            "volatile_statuses": [],
            "volatile_status_durations": {},
        }
        if private:
            snapshot.update({
                "ability": pokemon.ability,
                "item": pokemon.item,
                "nature": pokemon.nature,
                "evs": list(pokemon.evs),
                "stats": {
                    constants.ATTACK: pokemon.attack,
                    constants.DEFENSE: pokemon.defense,
                    constants.SPECIAL_ATTACK: pokemon.special_attack,
                    constants.SPECIAL_DEFENSE: pokemon.special_defense,
                    constants.SPEED: pokemon.speed,
                    constants.HITPOINTS: pokemon.maxhp,
                },
                "boosts": {},
                "can_mega_evo": False,
                "can_terastallize": False,
                "forme_changed": pokemon.mega_evolved,
            })
        else:
            snapshot.update({
                "ability": None,
                "item": constants.UNKNOWN_ITEM,
                "nature": None,
                "evs": None,
                "stats": None,
                "boosts": None,
                "can_mega_evo": None,
                "can_terastallize": None,
                "forme_changed": False,
            })
        return snapshot

    @classmethod
    def _side_conditions(cls, side) -> Dict[str, int | bool]:
        return {
            name: getattr(side.side_conditions, name)
            for name in cls._SIDE_CONDITIONS
            if getattr(side.side_conditions, name)
        }

    def _record_revealed_moves(self, side_one_action: str, side_two_action: str) -> None:
        if not side_one_action.startswith(f"{constants.SWITCH_STRING} "):
            self._seen_side_one_moves.setdefault(
                self._active_pokemon(self._state.side_one).id, set()
            ).add(side_one_action)
        if not side_two_action.startswith(f"{constants.SWITCH_STRING} "):
            self._seen_side_two_moves.setdefault(
                self._active_pokemon(self._state.side_two).id, set()
            ).add(side_two_action)