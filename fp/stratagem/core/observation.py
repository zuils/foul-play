"""
Observation layer for Stratagem system.
Extracts publicly visible battle state without leaking hidden information.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Set, Optional
from fp.battle.state import Battle, Pokemon
from fp import constants


class Observation:
    """
    Represents the publicly observable state of a battle from one player's perspective.
    Contains only information that would be available through normal battle observation.
    """

    def __init__(self, battle: Battle, player: str = "user"):
        """
        Initialize observation from a battle state.

        Args:
            battle: The Battle object to observe
            player: Either "user" or "opponent" indicating which side to observe from
        """
        self.player = player
        self._extract_observable_state(battle)

    @classmethod
    def from_public_snapshot(
        cls,
        *,
        player: str,
        active_pokemon: Dict,
        reserve_pokemon: List[Dict],
        hidden_reserve_count: int,
        opponent_active: Dict,
        opponent_reserve_revealed: List[Dict],
        opponent_hidden_reserve_count: int,
        weather: Optional[str] = None,
        weather_turns_remaining: int = 0,
        field: Optional[str] = None,
        field_turns_remaining: int = 0,
        trick_room: bool = False,
        trick_room_turns_remaining: int = 0,
        side_conditions: Optional[Dict] = None,
        opponent_side_conditions: Optional[Dict] = None,
        turn: int = 0,
        team_preview: bool = False,
        available_moves: Optional[List[str]] = None,
        disabled_moves: Optional[List[str]] = None,
        last_used_move: Optional[str] = None,
        active_effective_speed: Optional[int] = None,
        opponent_effective_speed: Optional[int] = None,
        active_hp_change: Optional[int] = None,
        opponent_hp_change: Optional[int] = None,
        active_is_choice_locked: bool = False,
        opponent_is_choice_locked: bool = False,
        active_locked_move: Optional[str] = None,
        opponent_locked_move: Optional[str] = None,
    ) -> "Observation":
        """Construct an Observation from validated public data without live state."""
        snapshot = {
            "active_pokemon": active_pokemon,
            "reserve_pokemon": reserve_pokemon,
            "hidden_reserve_count": hidden_reserve_count,
            "opponent_active": opponent_active,
            "opponent_reserve_revealed": opponent_reserve_revealed,
            "opponent_hidden_reserve_count": opponent_hidden_reserve_count,
            "weather": weather,
            "weather_turns_remaining": weather_turns_remaining,
            "field": field,
            "field_turns_remaining": field_turns_remaining,
            "trick_room": trick_room,
            "trick_room_turns_remaining": trick_room_turns_remaining,
            "side_conditions": side_conditions or {},
            "opponent_side_conditions": opponent_side_conditions or {},
            "turn": turn,
            "team_preview": team_preview,
            "available_moves": available_moves or [],
            "disabled_moves": disabled_moves or [],
            "last_used_move": last_used_move,
            "active_effective_speed": active_effective_speed,
            "opponent_effective_speed": opponent_effective_speed,
            "active_hp_change": active_hp_change,
            "opponent_hp_change": opponent_hp_change,
            "active_is_choice_locked": active_is_choice_locked,
            "opponent_is_choice_locked": opponent_is_choice_locked,
            "active_locked_move": active_locked_move,
            "opponent_locked_move": opponent_locked_move,
        }
        cls._validate_public_snapshot(snapshot)
        observation = cls.__new__(cls)
        observation.player = player
        for name, value in deepcopy(snapshot).items():
            setattr(observation, name, value)
        observation.status = observation.active_pokemon.get("status")
        observation.opponent_status = observation.opponent_active.get("status")
        observation.can_switch = any(
            pokemon.get("is_alive", False) for pokemon in observation.reserve_pokemon
        )
        observation.can_terastallize = bool(
            observation.active_pokemon.get("can_terastallize", False)
        )
        observation.is_terastallized = bool(
            observation.active_pokemon.get("is_terastallized", False)
        )
        observation.tera_type = observation.active_pokemon.get("tera_type")
        observation.can_mega_evolve = bool(
            observation.active_pokemon.get("can_mega_evo", False)
        )
        observation.is_mega_evolved = bool(
            observation.active_pokemon.get("is_mega", False)
        )
        observation.can_use_z_move = False
        observation.active_volatile_statuses = list(
            observation.active_pokemon.get("volatile_statuses", [])
        )
        observation.opponent_active_volatile_statuses = list(
            observation.opponent_active.get("volatile_statuses", [])
        )
        observation._visible_pokemon = {
            "self": observation._pokemon_by_name(
                [observation.active_pokemon, *observation.reserve_pokemon]
            ),
            "opponent": observation._pokemon_by_name(
                [observation.opponent_active, *observation.opponent_reserve_revealed]
            ),
        }
        return observation

    @staticmethod
    def _validate_public_snapshot(value) -> None:
        if value is None or isinstance(value, (str, int, float, bool)):
            return
        if isinstance(value, list):
            for item in value:
                Observation._validate_public_snapshot(item)
            return
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError("Public snapshot dictionary keys must be strings")
                Observation._validate_public_snapshot(item)
            return
        raise TypeError(f"Public snapshot contains a non-serializable value: {type(value)!r}")

    def _extract_observable_state(self, battle: Battle) -> None:
        """Extract all publicly observable information from the battle state."""
        battler = battle.user if self.player == "user" else battle.opponent
        opponent = battle.opponent if self.player == "user" else battle.user

        # Active Pokemon information
        self.active_pokemon = self._observable_pokemon(battler.active)

        # Reserve Pokemon information (only revealed ones)
        self.reserve_pokemon = [
            self._observable_pokemon(pkmn)
            for pkmn in battler.reserve
            if pkmn.revealed
        ]

        # Hidden reserve Pokemon count (unrevealed)
        self.hidden_reserve_count = len([
            pkmn for pkmn in battler.reserve
            if not pkmn.revealed
        ])

        # Opponent's active Pokemon (only what's visible)
        self.opponent_active = self._observable_pokemon(opponent.active)

        # Opponent's revealed reserve Pokemon
        self.opponent_reserve_revealed = [
            self._observable_pokemon(pkmn)
            for pkmn in opponent.reserve
            if pkmn.revealed
        ]

        # Opponent's hidden reserve count
        self.opponent_hidden_reserve_count = len([
            pkmn for pkmn in opponent.reserve
            if not pkmn.revealed
        ])

        # Battle field information
        self.weather = battle.weather
        self.weather_turns_remaining = battle.weather_turns_remaining
        self.field = battle.field
        self.field_turns_remaining = battle.field_turns_remaining
        self.trick_room = battle.trick_room
        self.trick_room_turns_remaining = battle.trick_room_turns_remaining

        # Side conditions (both sides)
        self.side_conditions = dict(battler.side_conditions)
        self.opponent_side_conditions = dict(opponent.side_conditions)

        # Battle metadata
        self.turn = battle.turn
        self.team_preview = battle.team_preview

        # Volatile statuses on active Pokemon (handle None active Pokemon)
        if battler.active:
            self.active_volatile_statuses = list(battler.active.volatile_statuses)
        else:
            self.active_volatile_statuses = []

        if opponent.active:
            self.opponent_active_volatile_statuses = list(opponent.active.volatile_statuses)
        else:
            self.opponent_active_volatile_statuses = []

        # Available moves for active Pokemon (handle None active Pokemon)
        if battler.active:
            self.available_moves = [
                move.name for move in battler.active.moves
                if not move.disabled
            ]
            self.disabled_moves = [
                move.name for move in battler.active.moves
                if move.disabled
            ]
        else:
            self.available_moves = []
            self.disabled_moves = []

        # Last used move (if any)
        self.last_used_move = (
            battler.last_used_move.move
            if battler.last_used_move.move
            else None
        )

        # Can the active Pokemon terastallize? (handle None active Pokemon)
        if battler.active:
            self.can_terastallize = battler.active.can_terastallize
            self.is_terastallized = battler.active.terastallized
            self.tera_type = battler.active.tera_type if battler.active.terastallized else None

            # Mega evolution info
            self.can_mega_evolve = battler.active.can_mega_evo
            self.is_mega_evolved = battler.active.is_mega

            # Z-move info
            self.can_use_z_move = any(
                getattr(move, 'can_z', False)
                for move in battler.active.moves
            )
        else:
            self.can_terastallize = False
            self.is_terastallized = False
            self.tera_type = None
            self.can_mega_evolve = False
            self.is_mega_evolved = False
            self.can_use_z_move = False

        # Snapshot evidence while the battle is available, then discard every live
        # Battle, Battler, and Pokemon reference. Opponent speed and Choice-item
        # state are hidden unless they were directly revealed in protocol evidence.
        self.active_effective_speed = (
            battle.get_effective_speed(battler)
            if battler.active and battler.active.revealed
            else None
        )
        self.opponent_effective_speed = None
        self.active_hp_change = self._public_hp_change(battler.active)
        self.opponent_hp_change = self._public_hp_change(opponent.active)
        self.active_is_choice_locked = self._is_user_choice_locked(battler)
        self.opponent_is_choice_locked = False
        self.active_locked_move = (
            battler.last_used_move.move if self.active_is_choice_locked else None
        )
        self.opponent_locked_move = None

        # Status conditions that affect action choice
        self.status = battler.active.status if battler.active else None
        self.opponent_status = opponent.active.status if opponent.active else None

        self.can_switch = any(pkmn.is_alive() for pkmn in battler.reserve)
        self._visible_pokemon = {
            "self": self._pokemon_by_name([self.active_pokemon, *self.reserve_pokemon]),
            "opponent": self._pokemon_by_name(
                [self.opponent_active, *self.opponent_reserve_revealed]
            ),
        }

    @staticmethod
    def _pokemon_by_name(pokemon: List[Dict]) -> Dict[str, Dict]:
        return {
            pkmn["name"]: pkmn
            for pkmn in pokemon
            if pkmn and pkmn.get("name")
        }

    def _observable_pokemon(self, pokemon: Pokemon) -> Dict:
        """
        Extract only publicly observable information from a Pokemon.

        Args:
            pokemon: The Pokemon to observe

        Returns:
            Dictionary containing only observable properties
        """
        if pokemon is None:
            return {}

        # Basic info that's always visible once Pokemon is revealed
        observable = {
            "name": pokemon.name,
            "level": pokemon.level,
            "revealed": pokemon.revealed,
            "is_alive": pokemon.is_alive(),
            "hp": pokemon.hp if pokemon.revealed else None,
            "max_hp": pokemon.max_hp if pokemon.revealed else None,
            "hp_fraction": pokemon.hp / pokemon.max_hp if pokemon.revealed and pokemon.max_hp > 0 else None,
            "status": pokemon.status,
            "types": list(pokemon.types),
            "moves": [move.name for move in pokemon.moves],
            "move_pp": [move.current_pp for move in pokemon.moves],
            "disabled_moves": [move.name for move in pokemon.moves if move.disabled],
        }

        # Only add these if the Pokemon is revealed
        if pokemon.revealed:
            # Normalize item to match constant form (lowercase, no spaces or hyphens)
            normalized_item = None
            if pokemon.item:
                normalized_item = pokemon.item.lower().replace(" ", "").replace("-", "")
            observable.update({
                "ability": pokemon.ability.lower() if pokemon.ability else None,
                "item": normalized_item,
                "nature": pokemon.nature,
                "evs": list(pokemon.evs) if pokemon.evs else None,
                "stats": {
                    constants.ATTACK: pokemon.stats[constants.ATTACK],
                    constants.DEFENSE: pokemon.stats[constants.DEFENSE],
                    constants.SPECIAL_ATTACK: pokemon.stats[constants.SPECIAL_ATTACK],
                    constants.SPECIAL_DEFENSE: pokemon.stats[constants.SPECIAL_DEFENSE],
                    constants.SPEED: pokemon.stats[constants.SPEED],
                    constants.HITPOINTS: pokemon.max_hp,  # HP is stored in max_hp after being popped from stats
                },
                "boosts": dict(pokemon.boosts),
                "can_mega_evo": pokemon.can_mega_evo,
                "can_terastallize": pokemon.can_terastallize,
                "is_mega": pokemon.is_mega,
                "is_terastallized": pokemon.terastallized,
                "tera_type": pokemon.tera_type,
                "forme_changed": pokemon.forme_changed,
            })

            # Add volatile status info
            observable["volatile_statuses"] = list(pokemon.volatile_statuses)
            observable["volatile_status_durations"] = dict(pokemon.volatile_status_durations)
        else:
            # For unrevealed Pokemon, we only know it exists
            observable.update({
                "ability": None,
                "item": constants.UNKNOWN_ITEM,
                "nature": None,
                "evs": None,
                "stats": None,
                "boosts": None,
                "can_mega_evo": None,
                "can_terastallize": None,
                "is_mega": False,
                "is_terastallized": False,
                "tera_type": None,
                "forme_changed": False,
                "volatile_statuses": [],
                "volatile_status_durations": {},
            })

        return observable

    def get_active_pokemon_info(self) -> Dict:
        """Get information about the player's active Pokemon."""
        return self.active_pokemon

    def get_opponent_active_pokemon_info(self) -> Dict:
        """Get information about the opponent's active Pokemon."""
        return self.opponent_active

    def get_revealed_reserve_pokemon(self) -> List[Dict]:
        """Get information about revealed reserve Pokemon."""
        return self.reserve_pokemon

    def get_opponent_revealed_reserve_pokemon(self) -> List[Dict]:
        """Get information about opponent's revealed reserve Pokemon."""
        return self.opponent_reserve_revealed

    def get_hidden_reserve_count(self) -> int:
        """Get count of unrevealed reserve Pokemon."""
        return self.hidden_reserve_count

    def get_opponent_hidden_reserve_count(self) -> int:
        """Get count of opponent's unrevealed reserve Pokemon."""
        return self.opponent_hidden_reserve_count

    def is_action_available(self, action_name: str) -> bool:
        """
        Check if an action (move or switch) is available.

        Args:
            action_name: Name of the move or "switch" for switching

        Returns:
            True if the action is available, False otherwise
        """
        if action_name == "switch":
            return self.can_switch
        return action_name in self.available_moves

    def get_available_actions(self) -> List[str]:
        """
        Get list of all available actions (moves and switches).

        Returns:
            List of available action names
        """
        actions = []

        # Add available moves
        actions.extend(self.available_moves)

        # Add switch option if available
        if self.can_switch:
            actions.append("switch")

        return actions

    def is_pokemon_revealed(self, pokemon_name: str, opponent: bool = False) -> bool:
        """
        Check if a specific Pokemon has been revealed.

        Args:
            pokemon_name: Name of the Pokemon to check
            opponent: Whether to check opponent's team (True) or player's team (False)

        Returns:
            True if the Pokemon has been revealed, False otherwise
        """
        side = "opponent" if opponent else "self"
        pkmn = self._visible_pokemon[side].get(pokemon_name)
        return bool(pkmn and pkmn["revealed"])

    def get_known_moves_of_pokemon(self, pokemon_name: str, opponent: bool = False) -> Set[str]:
        """
        Get set of known moves for a specific Pokemon.

        Args:
            pokemon_name: Name of the Pokemon
            opponent: Whether to check opponent's team (True) or player's team (False)

        Returns:
            Set of known move names
        """
        side = "opponent" if opponent else "self"
        pkmn = self._visible_pokemon[side].get(pokemon_name)
        if not pkmn or not pkmn["revealed"]:
            return set()
        return set(pkmn["moves"])

    def to_dict(self) -> Dict:
        """
        Convert observation to dictionary for logging/debugging.

        Returns:
            Dictionary representation of the observation
        """
        base_dict = {
            "active_pokemon": self.active_pokemon,
            "reserve_pokemon": self.reserve_pokemon,
            "hidden_reserve_count": self.hidden_reserve_count,
            "opponent_active": self.opponent_active,
            "opponent_reserve_revealed": self.opponent_reserve_revealed,
            "opponent_hidden_reserve_count": self.opponent_hidden_reserve_count,
            "weather": self.weather,
            "weather_turns_remaining": self.weather_turns_remaining,
            "field": self.field,
            "field_turns_remaining": self.field_turns_remaining,
            "side_conditions": self.side_conditions,
            "opponent_side_conditions": self.opponent_side_conditions,
            "turn": self.turn,
            "team_preview": self.team_preview,
            "available_moves": self.available_moves,
            "disabled_moves": self.disabled_moves,
            "last_used_move": self.last_used_move,
            "can_terastallize": self.can_terastallize,
            "is_terastallized": self.is_terastallized,
            "tera_type": self.tera_type,
            "can_mega_evolve": self.can_mega_evolve,
            "is_mega_evolved": self.is_mega_evolved,
            "can_use_z_move": self.can_use_z_move,
            # Additional evidence signals for belief updates
            "status": self.status,
            "opponent_status": self.opponent_status,
            "can_switch": self.can_switch,
        }

        base_dict.update({
            "active_effective_speed": self.active_effective_speed,
            "opponent_effective_speed": self.opponent_effective_speed,
            "active_hp_change": self.active_hp_change,
            "opponent_hp_change": self.opponent_hp_change,
            "active_is_choice_locked": self.active_is_choice_locked,
            "opponent_is_choice_locked": self.opponent_is_choice_locked,
            "active_locked_move": self.active_locked_move,
            "opponent_locked_move": self.opponent_locked_move,
        })

        return base_dict

    @staticmethod
    def _public_hp_change(pokemon: Optional[Pokemon]) -> Optional[int]:
        """Snapshot the public HP difference from the visible switch-in total."""
        if not pokemon or not pokemon.revealed:
            return None
        if hasattr(pokemon, 'hp_at_switch_in'):
            return pokemon.hp_at_switch_in - pokemon.hp
        return None

    @staticmethod
    def _is_user_choice_locked(battler) -> bool:
        """Determine the user's Choice lock without retaining the battler."""
        if not battler.active or not battler.active.revealed:
            return False
        choice_items = {constants.CHOICE_BAND.lower(), constants.CHOICE_SPECS.lower(), constants.CHOICE_SCARF.lower()}
        normalized_item = battler.active.item.lower().replace(" ", "").replace("-", "") if battler.active.item else ""
        has_choice_item = normalized_item in choice_items
        if not has_choice_item:
            return False
        if not battler.last_used_move or battler.last_used_move.pokemon_name != battler.active.name:
            return False
        enabled_moves = [move.name for move in battler.active.moves if not move.disabled]
        return len(enabled_moves) >= 1 and battler.last_used_move.move in enabled_moves

    def __str__(self) -> str:
        """String representation of the observation."""
        return f"Observation(turn={self.turn}, player={self.player})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"Observation(\n"
                f"  turn={self.turn},\n"
                f"  player={self.player},\n"
                f"  active_pokemon={self.active_pokemon.get('name', 'None')},\n"
                f"  opponent_active={self.opponent_active.get('name', 'None')},\n"
                f"  hidden_reserve_count={self.hidden_reserve_count},\n"
                f"  opponent_hidden_reserve_count={self.opponent_hidden_reserve_count}\n"
                f")")