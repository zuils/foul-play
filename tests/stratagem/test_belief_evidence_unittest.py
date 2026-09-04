"""Regression coverage for Belief updates from public Observation evidence."""

import math
import unittest

from fp.battle.state import Battle, Pokemon
from fp.config import FoulPlayConfig
from fp.modes.standard_battle import StandardBattleMode
from fp.stratagem.core import Observation
from fp.stratagem.inference import Belief


class BeliefEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.previous_format = FoulPlayConfig.pokemon_format
        FoulPlayConfig.pokemon_format = "gen9ou"

    def tearDown(self):
        FoulPlayConfig.pokemon_format = self.previous_format

    def test_update_eliminates_incompatible_public_evidence_and_ranks_known_move(self):
        battle = Battle(None)
        battle.generation = "gen9"
        battle.mode = StandardBattleMode()

        opponent_active = Pokemon(
            "charizard", level=100, nature="serious", evs=[0, 0, 0, 0, 0, 0]
        )
        opponent_active.revealed = True
        opponent_active.ability = "blaze"
        opponent_active.item = "leftovers"
        opponent_active.terastallized = True
        opponent_active.tera_type = "fire"
        opponent_active.add_move("flamethrower")
        battle.opponent.active = opponent_active

        revealed_reserve = Pokemon("gengar", level=100)
        revealed_reserve.revealed = True
        revealed_reserve.ability = "levitate"
        revealed_reserve.item = "blacksludge"
        revealed_reserve.add_move("shadowball")
        battle.opponent.reserve = [revealed_reserve]

        observation = Observation(battle)
        active_candidate = {
            "species": "charizard",
            "ability": "blaze",
            "item": "leftovers",
            "nature": "serious",
            "evs": [0, 0, 0, 0, 0, 0],
            "moves": ["flamethrower", "airslash"],
            "tera_type": "fire",
        }
        reserve_candidate = {
            "species": "gengar",
            "ability": "levitate",
            "item": "blacksludge",
            "nature": "serious",
            "evs": [85, 85, 85, 85, 85, 85],
            "moves": ["shadowball"],
            "tera_type": "ghost",
        }
        missing_revealed_move = {**active_candidate, "moves": ["airslash"]}
        wrong_tera_type = {**active_candidate, "tera_type": "water"}
        wrong_item = {**active_candidate, "item": "lifeorb"}
        wrong_species = {**active_candidate, "species": "venusaur"}
        wrong_reserve_ability = {**reserve_candidate, "ability": "cursedbody"}

        belief = Belief(team_sampler=None, world_count=6)
        belief.worlds = [
            [active_candidate, reserve_candidate],
            [missing_revealed_move, reserve_candidate],
            [wrong_tera_type, reserve_candidate],
            [active_candidate, wrong_reserve_ability],
            [wrong_item, reserve_candidate],
            [wrong_species, reserve_candidate],
        ]
        belief.weights = [0.0] * len(belief.worlds)

        belief.update_with_evidence(observation)

        self.assertTrue(belief._evidence_applied)
        self.assertGreater(belief.weights[0], belief.weights[1])
        for weight in belief.weights[1:]:
            self.assertTrue(math.isinf(weight) and weight < 0)


if __name__ == "__main__":
    unittest.main()