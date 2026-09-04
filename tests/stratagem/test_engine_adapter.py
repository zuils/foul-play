"""Unittest integration coverage for Stratagem's poke-engine state adapter."""

import unittest

from poke_engine import State, generate_instructions, monte_carlo_tree_search

from fp import constants
from fp.battle.state import Battle, Pokemon
from fp.config import FoulPlayConfig
from fp.modes.standard_battle import StandardBattleMode
from fp.stratagem.core import Observation
from fp.stratagem.engine.adapter import build_state, mcts_action_to_engine_input
from fp.stratagem.engine.search import search_world, search_world_with_visits


def team_spec(species, moves):
    return {
        "species": species,
        "item": "lifeorb",
        "ability": "blaze",
        "nature": "timid",
        "evs": [252, 0, 0, 252, 4, 252],
        "moves": moves,
        "tera_type": "fire",
        "level": 50,
    }


class EngineAdapterTests(unittest.TestCase):
    def setUp(self):
        self.previous_format = FoulPlayConfig.pokemon_format
        FoulPlayConfig.pokemon_format = "gen9ou"

    def tearDown(self):
        FoulPlayConfig.pokemon_format = self.previous_format

    def test_build_state_preserves_public_active_state(self):
        battle = Battle(None)
        battle.generation = "gen9"
        battle.mode = StandardBattleMode()
        battle.user.side_conditions[constants.SPIKES] = 2

        user = Pokemon("gengar", level=50, nature="timid", evs=[0, 0, 0, 252, 4, 252])
        user.revealed = True
        user.ability = "levitate"
        user.item = "lifeorb"
        user.terastallized = True
        user.tera_type = "water"
        user.add_move("shadowball")
        user.add_move("sludgebomb")
        user.hp = 81
        user.boosts[constants.SPECIAL_ATTACK] = 2
        battle.user.active = user

        opponent = Pokemon("charizard", level=50, nature="timid", evs=[0, 0, 0, 252, 4, 252])
        opponent.revealed = True
        opponent.ability = "blaze"
        opponent.item = "lifeorb"
        opponent.add_move("flamethrower")
        opponent.add_move("airslash")
        opponent.hp = 73
        battle.opponent.active = opponent

        observation = Observation(battle)
        our_team = [
            team_spec("gengar", ["shadowball", "sludgebomb"]),
            team_spec("pikachu", ["thunderbolt"]),
            team_spec("blastoise", ["surf"]),
            team_spec("venusaur", ["gigadrain"]),
            team_spec("machamp", ["closecombat"]),
            team_spec("charizard", ["flamethrower"]),
        ]
        our_team[0]["mega_evolved"] = True
        our_team[0]["tera_type"] = "fire"
        hidden_team = [
            team_spec("charizard", ["flamethrower", "airslash"]),
            team_spec("gengar", ["shadowball"]),
            team_spec("pikachu", ["thunderbolt"]),
            team_spec("blastoise", ["surf"]),
            team_spec("venusaur", ["gigadrain"]),
            team_spec("machamp", ["closecombat"]),
        ]

        state = build_state(observation, hidden_team, our_team)

        self.assertEqual(state.side_one.pokemon[0].id, "gengar")
        self.assertEqual(state.side_one.pokemon[0].hp, 81)
        self.assertEqual(state.side_one.pokemon[0].maxhp, user.max_hp)
        self.assertEqual(state.side_one.pokemon[0].special_attack, user.stats[constants.SPECIAL_ATTACK])
        self.assertTrue(state.side_one.pokemon[0].mega_evolved)
        self.assertTrue(state.side_one.pokemon[0].terastallized)
        self.assertEqual(state.side_one.pokemon[0].tera_type, "water")
        clone = State.from_string(state.to_string())
        self.assertTrue(clone.side_one.pokemon[0].mega_evolved)
        self.assertTrue(clone.side_one.pokemon[0].terastallized)
        self.assertEqual(clone.side_one.pokemon[0].tera_type.lower(), "water")
        self.assertEqual(state.side_one.special_attack_boost, 2)
        self.assertEqual(state.side_one.side_conditions.spikes, 2)
        self.assertEqual(state.side_two.pokemon[0].id, "charizard")
        self.assertEqual(state.side_two.pokemon[0].hp, 73)

    def test_search_world_invokes_mcts_with_legal_actions(self):
        battle = Battle(None)
        battle.generation = "gen9"
        battle.mode = StandardBattleMode()

        user = Pokemon("gengar", level=50, nature="timid", evs=[0, 0, 0, 252, 4, 252])
        user.revealed = True
        user.add_move("shadowball")
        user.add_move("sludgebomb")
        battle.user.active = user

        opponent = Pokemon("charizard", level=50, nature="timid", evs=[0, 0, 0, 252, 4, 252])
        opponent.revealed = True
        opponent.add_move("flamethrower")
        opponent.add_move("airslash")
        battle.opponent.active = opponent

        our_team = [
            team_spec("gengar", ["shadowball", "sludgebomb"]),
            team_spec("pikachu", ["thunderbolt"]),
            team_spec("blastoise", ["surf"]),
            team_spec("venusaur", ["gigadrain"]),
            team_spec("machamp", ["closecombat"]),
            team_spec("charizard", ["flamethrower"]),
        ]
        hidden_team = [
            team_spec("charizard", ["flamethrower", "airslash"]),
            team_spec("gengar", ["shadowball"]),
            team_spec("pikachu", ["thunderbolt"]),
            team_spec("blastoise", ["surf"]),
            team_spec("venusaur", ["gigadrain"]),
            team_spec("machamp", ["closecombat"]),
        ]

        scores = search_world(Observation(battle), hidden_team, our_team, search_time_ms=1)

        self.assertTrue(scores)
        self.assertTrue(
            set(scores).issubset(
                {
                    "shadowball", "sludgebomb", "switch pikachu", "switch blastoise",
                    "switch venusaur", "switch machamp", "switch charizard",
                }
            ),
            scores,
        )
        _, visits = search_world_with_visits(
            Observation(battle), hidden_team, our_team, search_time_ms=1
        )
        self.assertEqual(set(scores), set(visits))
        self.assertGreater(sum(visits.values()), 0)

    def test_every_mcts_action_round_trips_to_a_real_engine_transition(self):
        observation = Observation.from_public_snapshot(
            player="user",
            active_pokemon={"name": "gengar", "revealed": True},
            reserve_pokemon=[],
            hidden_reserve_count=5,
            opponent_active={"name": "charizard", "revealed": True},
            opponent_reserve_revealed=[],
            opponent_hidden_reserve_count=5,
        )
        side_one = [
            team_spec("gengar", ["shadowball", "sludgebomb"]),
            team_spec("pikachu", ["thunderbolt"]),
            team_spec("blastoise", ["surf"]),
            team_spec("venusaur", ["gigadrain"]),
            team_spec("machamp", ["closecombat"]),
            team_spec("charizard", ["flamethrower"]),
        ]
        side_two = [
            team_spec("charizard", ["flamethrower", "airslash"]),
            team_spec("gengar", ["shadowball"]),
            team_spec("pikachu", ["thunderbolt"]),
            team_spec("blastoise", ["surf"]),
            team_spec("venusaur", ["gigadrain"]),
            team_spec("machamp", ["closecombat"]),
        ]
        state = build_state(observation, side_two, side_one)
        result = monte_carlo_tree_search(state, duration_ms=25)
        opponent_action = result.side_two[0].move_choice
        switch_action = next(
            option.move_choice
            for option in result.side_one
            if option.move_choice.startswith("switch ")
        )

        with self.assertRaisesRegex(ValueError, "Invalid move"):
            generate_instructions(state, switch_action, opponent_action)
        for option in result.side_one:
            transitions = generate_instructions(
                state,
                mcts_action_to_engine_input(option.move_choice),
                mcts_action_to_engine_input(opponent_action),
            )
            self.assertTrue(transitions, option.move_choice)

        switched_state = State.from_string(state.to_string()).apply_instructions(
            generate_instructions(
                state,
                mcts_action_to_engine_input(switch_action),
                mcts_action_to_engine_input(opponent_action),
            )[0]
        )
        switch_target = mcts_action_to_engine_input(switch_action)
        self.assertEqual(
            str(
                switched_state.side_one.pokemon[
                    int(str(switched_state.side_one.active_index))
                ].id
            ).lower(),
            switch_target,
        )
        self.assertEqual(state.side_one.pokemon[0].id, "gengar")
        self.assertEqual(mcts_action_to_engine_input("No Move"), "none")

    def test_mcts_mega_action_round_trips_and_persists_in_engine_state(self):
        observation = Observation.from_public_snapshot(
            player="user",
            active_pokemon={"name": "charizard", "revealed": True},
            reserve_pokemon=[],
            hidden_reserve_count=5,
            opponent_active={"name": "blastoise", "revealed": True},
            opponent_reserve_revealed=[],
            opponent_hidden_reserve_count=5,
        )
        side_one = [
            team_spec("charizard", ["flamethrower"]),
            team_spec("pikachu", ["thunderbolt"]),
            team_spec("blastoise", ["surf"]),
            team_spec("venusaur", ["gigadrain"]),
            team_spec("machamp", ["closecombat"]),
            team_spec("gengar", ["shadowball"]),
        ]
        side_one[0]["item"] = "charizarditex"
        side_two = [
            team_spec("blastoise", ["surf"]),
            team_spec("pikachu", ["thunderbolt"]),
            team_spec("charizard", ["flamethrower"]),
            team_spec("venusaur", ["gigadrain"]),
            team_spec("machamp", ["closecombat"]),
            team_spec("gengar", ["shadowball"]),
        ]
        state = build_state(observation, side_two, side_one)
        result = monte_carlo_tree_search(state, duration_ms=25)
        mega_action = next(
            option.move_choice
            for option in result.side_one
            if option.move_choice.endswith("-mega")
        )
        opponent_action = result.side_two[0].move_choice

        next_state = State.from_string(state.to_string()).apply_instructions(
            generate_instructions(
                state,
                mcts_action_to_engine_input(mega_action),
                mcts_action_to_engine_input(opponent_action),
            )[0]
        )

        self.assertEqual(mcts_action_to_engine_input(mega_action), mega_action)
        self.assertTrue(next_state.side_one.pokemon[0].mega_evolved)


if __name__ == "__main__":
    unittest.main()