"""Unittest coverage for TeamSampler's real SmogonSets integration contract."""

import unittest

from fp.battle.state import Battle
from fp.config import FoulPlayConfig
from fp.data.sets import PokemonSet
from fp.modes.standard_battle import StandardBattleMode
from fp.stratagem.core import Observation
from fp.stratagem.inference import TeamSampler


class StaticSmogonSets:
    species = ("charizard", "gengar", "blastoise", "venusaur", "machamp", "pikachu")

    def __init__(self):
        self.pkmn_sets = {species: [object()] for species in self.species}

    def get_all_remaining_trait_combinations(self, pkmn):
        return [
            PokemonSet(
                ability="blaze" if pkmn.name == "charizard" else "static",
                item="lifeorb",
                nature="timid",
                evs=(252, 0, 0, 252, 4, 252),
                count=1,
                tera_type="fire",
            )
        ]

    def move_usage_rates(self, pkmn):
        if pkmn.name == "charizard":
            return [("flamethrower", 0.8), ("airslash", 0.6), ("roost", 0.5), ("focusblast", 0.4)]
        return [("tackle", 1.0)]


class TeamSamplerSmogonApiTests(unittest.TestCase):
    def setUp(self):
        self.previous_format = FoulPlayConfig.pokemon_format
        FoulPlayConfig.pokemon_format = "gen9ou"

    def tearDown(self):
        FoulPlayConfig.pokemon_format = self.previous_format

    def test_samples_real_pokemon_set_values_and_retains_observed_moves(self):
        battle = Battle(None)
        battle.generation = "gen9"
        battle.mode = StandardBattleMode()
        observation = Observation(battle)
        observation.opponent_active = {
            "name": "charizard",
            "revealed": True,
            "ability": "blaze",
            "item": "lifeorb",
            "nature": "timid",
            "evs": [252, 0, 0, 252, 4, 252],
            "moves": ["flamethrower"],
        }

        team = TeamSampler(StaticSmogonSets(), random_seed=7).sample_team(observation)

        charizard = next(pkmn for pkmn in team if pkmn["species"] == "charizard")
        self.assertEqual(charizard["item"], "lifeorb")
        self.assertIn("flamethrower", charizard["moves"])
        self.assertEqual(len(team), 6)

    def test_unknown_public_item_does_not_eliminate_real_set_candidates(self):
        battle = Battle(None)
        battle.generation = "gen9"
        battle.mode = StandardBattleMode()
        observation = Observation(battle)
        observation.opponent_active = {
            "name": "charizard",
            "revealed": True,
            "ability": None,
            "item": "unknownitem",
            "nature": None,
            "evs": None,
            "moves": [],
        }

        team = TeamSampler(StaticSmogonSets(), random_seed=7).sample_team(observation)

        self.assertEqual(
            next(pokemon for pokemon in team if pokemon["species"] == "charizard")["item"],
            "lifeorb",
        )


if __name__ == "__main__":
    unittest.main()