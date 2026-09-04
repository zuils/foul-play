"""
Adapter to convert Stratagem observation and hidden team hypotheses to poke-engine state.
"""
from __future__ import annotations

from typing import Dict, List, Optional
from fp.battle.helpers import calculate_stats, normalize_name
from fp import constants
from fp.data import pokedex
from fp.stratagem.core import Observation
from poke_engine import (
    State as PokeEngineState,
    Side as PokeEngineSide,
    SideConditions as PokeEngineSideConditions,
    VolatileStatusDurations as PokeEngineVolatileStatusDurations,
    Pokemon as PokeEnginePokemon,
    Move as PokeEngineMove,
)


def _species_to_id(species: str) -> str:
    """Convert species name to poke-engine ID (lowercase)."""
    return species.lower()


def mcts_action_to_engine_input(action: str) -> str:
    """Convert poke-engine's MCTS display action to its transition parser input.

    The v0.0.48 PyO3 binding displays `MoveChoice::Switch` as `switch <species>`
    but `MoveChoice::from_string()` accepts the raw non-active species name.
    All other MCTS action strings are already accepted unchanged.
    """
    if not isinstance(action, str) or not action:
        raise ValueError("Engine action must be a non-empty string")
    if action == "No Move":
        return "none"
    if action.startswith(f"{constants.SWITCH_STRING} "):
        target = action.removeprefix(f"{constants.SWITCH_STRING} ")
        if not target:
            raise ValueError("Switch action is missing its target species")
        return target
    return action


def _item_to_id(item: Optional[str]) -> str:
    """Convert item name to poke-engine ID."""
    if not item or item == constants.UNKNOWN_ITEM:
        return "None"
    return item.lower().replace(" ", "").replace("-", "")


def _ability_to_id(ability: Optional[str]) -> str:
    """Convert ability name to poke-engine ID."""
    if not ability:
        return "None"
    return ability.lower()


def _nature_to_id(nature: Optional[str]) -> str:
    """Convert nature name to poke-engine ID."""
    if not nature:
        return "Hardy"
    return nature.capitalize()


def _status_to_id(status: Optional[str]) -> str:
    """Convert status string to poke-engine status."""
    if not status:
        return "None"
    mapping = {
        "sleep": "Sleep",
        "burn": "Burn",
        "freeze": "Freeze",
        "paralyze": "Paralyze",
        "poison": "Poison",
        "tox": "Toxic",
        "toxicity": "Toxic",
    }
    key = status.lower()
    return mapping.get(key, status.capitalize())


def _create_pokemon_from_spec(
    species: str,
    item: Optional[str],
    ability: Optional[str],
    nature: Optional[str],
    evs: List[int],
    moves: List[str],
    tera_type: Optional[str],
    level: int = 100,
    mega_evolved: bool = False,
    terastallized: bool = False,
    hp_override: Optional[int] = None,
    max_hp_override: Optional[int] = None,
    stats_override: Optional[Dict[str, int]] = None,
    status_override: Optional[str] = None,
    boosts_override: Optional[Dict[str, int]] = None,
    volatile_statuses_override: Optional[List[str]] = None,
    volatile_durations_override: Optional[Dict[str, int]] = None,
) -> PokeEnginePokemon:
    """
    Create a PokeEnginePokemon from spec dictionaries and optional overrides.
    """
    # Defaults
    if evs is None:
        evs = [0, 0, 0, 0, 0, 0]
    if moves is None:
        moves = []
    if boosts_override is None:
        boosts_override = {}
    if volatile_statuses_override is None:
        volatile_statuses_override = []
    if volatile_durations_override is None:
        volatile_durations_override = {}

    species = normalize_name(species)
    base_stats = pokedex[species][constants.BASESTATS]
    normalized_evs = tuple(int(value or 0) for value in evs)
    calculated_stats = calculate_stats(
        base_stats, level, evs=normalized_evs, nature=_nature_to_id(nature).lower()
    )
    stats = stats_override or calculated_stats
    max_hp = int(max_hp_override or stats[constants.HITPOINTS])
    hp = int(hp_override if hp_override is not None else max_hp)

    # Types
    base_types = pokedex[species][constants.TYPES]
    if len(base_types) == 1:
        base_types = (base_types[0], "typeless")
    else:
        base_types = (base_types[0], base_types[1])

    # Moves
    pkmn_moves = []
    for move_name in moves[:4]:
        pkmn_moves.append(PokeEngineMove(id=normalize_name(move_name), disabled=False, pp=16))
    while len(pkmn_moves) < 4:
        pkmn_moves.append(PokeEngineMove(id="none", disabled=True, pp=0))

    # Ability
    ability_id = _ability_to_id(ability)

    # Item
    item_id = _item_to_id(item)

    # Nature
    nature_id = _nature_to_id(nature)

    # Status
    status_id = _status_to_id(status_override) if status_override else "None"

    # Build the pokemon
    pkmn = PokeEnginePokemon(
        id=species,
        level=level,
        types=tuple(base_types),
        base_types=tuple(base_types),
        hp=hp,
        maxhp=max_hp,
        ability=ability_id,
        base_ability=ability_id,  # assume no ability change
        item=item_id,
        nature=nature_id,
        evs=normalized_evs,
        attack=int(stats[constants.ATTACK]),
        defense=int(stats[constants.DEFENSE]),
        special_attack=int(stats[constants.SPECIAL_ATTACK]),
        special_defense=int(stats[constants.SPECIAL_DEFENSE]),
        speed=int(stats[constants.SPEED]),
        status=status_id,
        rest_turns=0,
        sleep_turns=0,
        weight_kg=float(pokedex.get(species, {}).get(constants.WEIGHT, 0.0)),
        moves=pkmn_moves,
        tera_type=tera_type.lower() if tera_type else "typeless",
        mega_evolved=mega_evolved,
        terastallized=terastallized,
    )
    return pkmn


def _build_side_from_team(
    team_specs: List[Dict],
    active_observable: Dict,
    reserve_observables: List[Dict],
    side_conditions: Dict,
    last_used_move: str = "move:none",
) -> PokeEngineSide:
    """
    Build a poke-engine Side from team specs and observable Pokemon states.
    The active Pokemon is placed in engine slot zero. Revealed reserve entries are
    matched by species; all remaining sampled team members follow in stable order.
    """
    assert len(team_specs) == 6
    ordered_specs = _order_team_specs(team_specs, active_observable, reserve_observables)
    observable_by_species = {
        observable["name"]: observable
        for observable in [active_observable, *reserve_observables]
        if observable and observable.get("name")
    }

    pokemon_list = []
    for spec in ordered_specs:
        obs = observable_by_species.get(normalize_name(spec["species"]))
        hp_override = None
        max_hp_override = None
        stats_override = None
        status_override = None
        boosts_override = None
        volatile_statuses_override = None
        volatile_durations_override = None
        if obs:
            if obs.get("hp") is not None:
                hp_override = obs["hp"]
            max_hp_override = obs.get("max_hp")
            stats_override = obs.get("stats")
            status_override = obs.get("status")
            if obs.get("boosts"):
                boosts_override = obs["boosts"]
            if obs.get("volatile_statuses") is not None:
                volatile_statuses_override = obs["volatile_statuses"]
            if obs.get("volatile_status_durations") is not None:
                volatile_durations_override = obs["volatile_status_durations"]
        pkmn = _create_pokemon_from_spec(
            species=spec.get("species", ""),
            item=spec.get("item"),
            ability=spec.get("ability"),
            nature=spec.get("nature"),
            evs=spec.get("evs"),
            moves=spec.get("moves"),
            tera_type=(
                obs.get("tera_type")
                if obs and obs.get("is_terastallized")
                else spec.get("tera_type")
            ),
            level=int(spec.get("level") or 100),
            mega_evolved=bool(
                spec.get("mega_evolved", False) or (obs and obs.get("is_mega", False))
            ),
            terastallized=bool(obs and obs.get("is_terastallized", False)),
            hp_override=hp_override,
            max_hp_override=max_hp_override,
            stats_override=stats_override,
            status_override=status_override,
            boosts_override=boosts_override,
            volatile_statuses_override=list(volatile_statuses_override) if volatile_statuses_override else [],
            volatile_durations_override=volatile_durations_override,
        )
        pokemon_list.append(pkmn)

    side_conditions = PokeEngineSideConditions(
        **_convert_side_conditions(side_conditions)
    )

    vol_status_dur = PokeEngineVolatileStatusDurations(
        **_convert_volatile_durations(active_observable.get("volatile_status_durations", {}))
    )

    side = PokeEngineSide(
        active_index="0",
        baton_passing=False,
        shed_tailing=False,
        pokemon=pokemon_list,
        side_conditions=side_conditions,
        wish=(0, 0),
        future_sight=(0, "0"),
        force_switch=False,
        force_trapped=False,
        slow_uturn_move=False,
        volatile_statuses=set(active_observable.get("volatile_statuses", [])),
        volatile_status_durations=vol_status_dur,
        substitute_health=0,
        attack_boost=(active_observable.get("boosts") or {}).get(constants.ATTACK, 0),
        defense_boost=(active_observable.get("boosts") or {}).get(constants.DEFENSE, 0),
        special_attack_boost=(active_observable.get("boosts") or {}).get(constants.SPECIAL_ATTACK, 0),
        special_defense_boost=(active_observable.get("boosts") or {}).get(constants.SPECIAL_DEFENSE, 0),
        speed_boost=(active_observable.get("boosts") or {}).get(constants.SPEED, 0),
        accuracy_boost=0,
        evasion_boost=0,
        last_used_move=last_used_move,
        switch_out_move_second_saved_move="NONE",
    )
    return side


def _order_team_specs(
    team_specs: List[Dict], active_observable: Dict, reserve_observables: List[Dict]
) -> List[Dict]:
    remaining = list(team_specs)
    ordered = []
    for observable in [active_observable, *reserve_observables]:
        if not observable or not observable.get("name"):
            continue
        name = normalize_name(observable["name"])
        for index, spec in enumerate(remaining):
            if normalize_name(spec["species"]) == name:
                ordered.append(remaining.pop(index))
                break
    return ordered + remaining


def _convert_side_conditions(side_conditions: Dict) -> Dict[str, int | bool]:
    converted = {
        "aurora_veil": 0, "crafty_shield": 0, "healing_wish": 0,
        "light_screen": 0, "lucky_chant": 0, "lunar_dance": 0,
        "mat_block": 0, "mist": 0, "protect": 0, "quick_guard": 0,
        "reflect": 0, "safeguard": 0, "spikes": 0, "stealth_rock": 0,
        "sticky_web": 0, "tailwind": 0, "toxic_count": 0,
        "toxic_spikes": 0, "wide_guard": 0,
    }
    condition_mapping = {
        "auroraveil": "aurora_veil", "craftyshield": "crafty_shield",
        "healingwish": "healing_wish", "lightscreen": "light_screen",
        "luckychant": "lucky_chant", "lunardance": "lunar_dance",
        "matblock": "mat_block", "quickguard": "quick_guard",
        "stealthrock": "stealth_rock", "stickyweb": "sticky_web",
        "toxicspikes": "toxic_spikes", "wideguard": "wide_guard",
    }
    for name, value in side_conditions.items():
        normalized_name = normalize_name(str(name))
        field_name = condition_mapping.get(normalized_name, normalized_name)
        if field_name in converted:
            converted[field_name] = value
    return converted


def _convert_volatile_durations(durations: Dict) -> Dict[str, int]:
    return {
        "confusion": durations.get("confusion", 0),
        "lockedmove": durations.get("lockedmove", 0),
        "encore": durations.get("encore", 0),
        "slowstart": durations.get("slowstart", 0),
        "taunt": durations.get("taunt", 0),
        "yawn": durations.get("yawn", 0),
    }


def build_state(
    observation: Observation,
    hidden_team: List[Dict],
    our_team: List[Dict],
) -> PokeEngineState:
    """
    Build a poke-engine State from observation and team hypotheses.

    Args:
        observation: Stratagem observation (public info)
        hidden_team: list of 6 dicts representing opponent's full team (spec)
        our_team: list of 6 dicts representing our own full team (spec)
    Returns:
        PokeEngineState ready for simulation
    """
    # Weather and field from observation
    weather = observation.weather
    weather_turns = getattr(observation, "weather_turns_remaining", 0)
    field = observation.field
    field_turns = getattr(observation, "field_turns_remaining", 0)
    trick_room = getattr(observation, "trick_room", False)
    trick_room_turns = getattr(observation, "trick_room_turns_remaining", 0)

    # Convert weather from observation to poke-engine format
    if weather is None:
        poke_weather = "none"
    else:
        # Handle Foul Play Weather enum
        weather_mapping = {
            "raindance": "rain",
            "sunnyday": "sun",
            "sandstorm": "sand",
            "hail": "hail",
            "snowscape": "snow",
            "desolateland": "desolateland",
            "primordialsea": "primordialsea",
        }
        weather_str = str(weather)
        poke_weather = weather_mapping.get(weather_str, weather_str.lower())

    # Convert field/terrain from observation to poke-engine format
    if field is None:
        poke_terrain = "none"
    else:
        # Handle Foul Play Terrain enum
        terrain_mapping = {
            "electricterrain": "electric",
            "grassyterrain": "grassy",
            "mistyterrain": "misty",
            "psychicterrain": "psychic",
        }
        field_str = str(field)
        poke_terrain = terrain_mapping.get(field_str, field_str.lower())

    # Build our side (user)
    our_side = _build_side_from_team(
        team_specs=our_team,
        active_observable=observation.active_pokemon,
        reserve_observables=observation.reserve_pokemon,
        side_conditions=observation.side_conditions,
        last_used_move="move:none",
    )

    # Build opponent side
    opp_side = _build_side_from_team(
        team_specs=hidden_team,
        active_observable=observation.opponent_active,
        reserve_observables=observation.opponent_reserve_revealed,
        side_conditions=observation.opponent_side_conditions,
    )

    state = PokeEngineState(
        side_one=our_side,
        side_two=opp_side,
        weather=poke_weather,
        weather_turns_remaining=weather_turns,
        terrain=poke_terrain,
        terrain_turns_remaining=field_turns,
        trick_room=trick_room,
        trick_room_turns_remaining=trick_room_turns,
        team_preview=getattr(observation, "team_preview", False),
    )
    return state

