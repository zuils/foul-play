"""Unittest coverage for Stratagem's hidden-information boundary."""

import unittest

from fp.battle.state import Battle, Pokemon
from fp.config import FoulPlayConfig
from fp.modes.standard_battle import StandardBattleMode
from fp.stratagem.core import Observation


class ObservationBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.previous_format = FoulPlayConfig.pokemon_format
        FoulPlayConfig.pokemon_format = "gen9ou"

    def tearDown(self):
        FoulPlayConfig.pokemon_format = self.previous_format

    def test_observation_cannot_reach_hidden_opponent_state(self):
        battle = Battle(None)
        battle.generation = "gen9"
        battle.mode = StandardBattleMode()

        hidden_opponent = Pokemon("charizard", level=50)
        hidden_opponent.revealed = False
        hidden_opponent.item = "charizarditey"
        hidden_opponent.ability = "blaze"
        battle.opponent.active = hidden_opponent

        observation = Observation(battle)

        self.assertFalse(hasattr(observation, "_battle"))
        self.assertFalse(hasattr(observation, "_active_battler"))
        self.assertFalse(hasattr(observation, "_opponent_battler"))
        self.assertFalse(hasattr(observation, "_active_pokemon_obj"))
        self.assertFalse(hasattr(observation, "_opponent_pokemon_obj"))
        self.assertNotIn("charizarditey", repr(observation.__dict__))
        self.assertNotIn("charizarditey", repr(observation.to_dict()))
        self.assertIsNone(observation.opponent_active["ability"])
        self.assertIsNone(observation.opponent_effective_speed)
        self.assertFalse(observation.opponent_is_choice_locked)


if __name__ == "__main__":
    unittest.main()