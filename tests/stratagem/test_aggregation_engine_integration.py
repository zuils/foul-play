"""Integration coverage for real multi-world poke-engine aggregation."""

import math
import unittest

from fp.battle.state import Battle, Pokemon
from fp.config import FoulPlayConfig
from fp.modes.standard_battle import StandardBattleMode
from fp.stratagem.core import Observation
from fp.stratagem.engine.aggregation import WorldAggregator
from fp.stratagem.inference import Belief, TeamSampler


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


class NoopSets:
    pkmn_sets = {}


class WorldAggregationEngineTests(unittest.TestCase):
    def setUp(self):
        self.previous_format = FoulPlayConfig.pokemon_format
        FoulPlayConfig.pokemon_format = "gen9ou"

    def tearDown(self):
        FoulPlayConfig.pokemon_format = self.previous_format

    def test_aggregates_live_mcts_results_for_two_distinct_worlds(self):
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
        first_world = [
            team_spec("charizard", ["flamethrower", "airslash"]),
            team_spec("gengar", ["shadowball"]),
            team_spec("pikachu", ["thunderbolt"]),
            team_spec("blastoise", ["surf"]),
            team_spec("venusaur", ["gigadrain"]),
            team_spec("machamp", ["closecombat"]),
        ]
        second_world = [
            team_spec("charizard", ["fireblast", "airslash"]),
            team_spec("machamp", ["closecombat"]),
            team_spec("venusaur", ["gigadrain"]),
            team_spec("blastoise", ["surf"]),
            team_spec("pikachu", ["thunderbolt"]),
            team_spec("gengar", ["shadowball"]),
        ]
        belief = Belief(TeamSampler(NoopSets()), world_count=2, random_seed=7)
        belief.worlds = [first_world, second_world]
        belief.weights = [math.log(0.75), math.log(0.25)]
        belief._evidence_applied = True

        scores, visits = WorldAggregator(belief).aggregate_worlds_with_visits(
            Observation(battle), our_team, search_time_ms=1
        )

        self.assertTrue(scores)
        self.assertEqual(set(scores), set(visits))
        self.assertGreaterEqual(sum(visits.values()), 2)
        self.assertTrue(
            set(scores).issubset(
                {
                    "shadowball", "sludgebomb", "switch pikachu", "switch blastoise",
                    "switch venusaur", "switch machamp", "switch charizard",
                }
            )
        )


if __name__ == "__main__":
    unittest.main()