import random

from fp.battle.state import Battle, Pokemon
from fp.config import FoulPlayConfig
from fp.data.sets import MOVES_STRING, PokemonSet
from fp.modes.standard_battle import StandardBattleMode
from fp.search.poke_engine_helpers import battle_to_poke_engine_state
from fp.search.standard_battles import _sample_pokemon

from poke_engine import monte_carlo_tree_search


def concrete_world(opponent_item, opponent_moves):
    FoulPlayConfig.pokemon_format = "gen7ou"

    battle = Battle(None)
    battle.generation = "gen7"
    battle.pokemon_format = "gen7ou"

    battle.user.active = Pokemon("pikachu", 100)
    battle.user.active.ability = "static"
    battle.user.active.item = "lightball"
    battle.user.active.add_move("thunderbolt")

    battle.opponent.active = Pokemon("vaporeon", 100)
    battle.opponent.active.ability = "waterabsorb"
    battle.opponent.active.item = opponent_item
    for move in opponent_moves:
        battle.opponent.active.add_move(move)

    # This is format/request gating only. The engine still decides whether this
    # concrete item's moves create a legal Z action.
    battle.opponent.can_z_move = True
    return battle


def smogon_sampled_opponent_world(item):
    FoulPlayConfig.pokemon_format = "gen7ou"
    mode = StandardBattleMode()
    mode.smogon_sets.pkmn_sets = {
        "vaporeon": [
            PokemonSet(
                ability="waterabsorb",
                item=item,
                nature="modest",
                evs=(0, 0, 0, 252, 4, 252),
                count=1,
                tera_type=None,
            )
        ]
    }
    mode.smogon_sets.raw_pkmn_sets = {
        "vaporeon": {MOVES_STRING: [("surf", 1.0)]}
    }

    opponent = Pokemon("vaporeon", 100)
    random.seed(0)
    _sample_pokemon(opponent, mode)

    battle = concrete_world(opponent.item, [move.name for move in opponent.moves])
    return opponent, battle


def opponent_actions(battle):
    state = battle_to_poke_engine_state(battle)
    result = monte_carlo_tree_search(state, iterations=100)
    return {option.move_choice for option in result.side_two}


def test_concrete_world_without_z_crystal_has_no_z_action():
    actions = opponent_actions(concrete_world("leftovers", ["surf"]))

    assert "surf" in actions
    assert "surf-z" not in actions


def test_concrete_world_with_compatible_z_crystal_has_z_action():
    actions = opponent_actions(concrete_world("wateriumz", ["surf"]))

    assert "surf" in actions
    assert "surf-z" in actions


def test_concrete_world_with_incompatible_z_crystal_has_no_z_action():
    actions = opponent_actions(concrete_world("firiumz", ["surf"]))

    assert "surf" in actions
    assert "surf-z" not in actions


def test_mixed_concrete_worlds_keep_distinct_z_action_spaces():
    non_z_world = concrete_world("leftovers", ["surf"])
    z_world = concrete_world("wateriumz", ["surf"])

    assert non_z_world is not z_world
    assert battle_to_poke_engine_state(non_z_world).to_string() != (
        battle_to_poke_engine_state(z_world).to_string()
    )
    assert "surf-z" not in opponent_actions(non_z_world)
    assert "surf-z" in opponent_actions(z_world)


def test_smogon_trait_and_move_sampling_keeps_z_action_world_specific():
    non_z_opponent, non_z_world = smogon_sampled_opponent_world("leftovers")
    z_opponent, z_world = smogon_sampled_opponent_world("wateriumz")

    assert non_z_opponent.item == "leftovers"
    assert z_opponent.item == "wateriumz"
    assert [move.name for move in non_z_opponent.moves] == ["surf"]
    assert [move.name for move in z_opponent.moves] == ["surf"]
    assert "surf-z" not in opponent_actions(non_z_world)
    assert "surf-z" in opponent_actions(z_world)
