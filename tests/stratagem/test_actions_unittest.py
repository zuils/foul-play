"""Unittest coverage for public-information action guards and mixed selection."""

import unittest

from fp.battle.state import Battle, Pokemon
from fp.config import FoulPlayConfig
from fp.modes.standard_battle import StandardBattleMode
from fp.stratagem.core import Observation, StrategicActionSelector


class StrategicActionSelectorTests(unittest.TestCase):
    def setUp(self):
        self.previous_format = FoulPlayConfig.pokemon_format
        FoulPlayConfig.pokemon_format = "gen9ou"

    def tearDown(self):
        FoulPlayConfig.pokemon_format = self.previous_format

    def _observation(self):
        battle = Battle(None)
        battle.generation = "gen9"
        battle.mode = StandardBattleMode()

        user = Pokemon("gengar", level=50)
        user.revealed = True
        user.add_move("shadowball")
        user.add_move("sludgebomb")
        battle.user.active = user
        reserve = Pokemon("pikachu", level=50)
        reserve.revealed = True
        battle.user.reserve = [reserve]

        opponent = Pokemon("snorlax", level=50)
        opponent.revealed = True
        battle.opponent.active = opponent
        return Observation(battle)

    def test_filters_unavailable_and_revealed_immune_actions(self):
        observation = self._observation()
        selector = StrategicActionSelector(random_seed=7)
        observation.opponent_active["types"] = ["normal"]

        legal = selector.legal_actions(
            observation,
            {"shadowball": 1.0, "sludgebomb": 0.9, "thunderbolt": 2.0, "switch pikachu": 0.8},
        )

        self.assertEqual(legal, {"sludgebomb": 0.9, "switch pikachu": 0.8})

    def test_mixed_selection_is_seeded_and_preserves_clear_preference(self):
        observation = self._observation()
        observation.opponent_active["types"] = ["fighting"]
        scores = {"shadowball": 2.0, "sludgebomb": -20.0, "switch pikachu": -20.0}
        first = StrategicActionSelector(random_seed=19)
        second = StrategicActionSelector(random_seed=19)

        self.assertEqual(first.select_action(observation, scores), "shadowball")
        self.assertEqual(second.select_action(observation, scores), "shadowball")

    def test_retains_engine_mechanic_actions_only_when_publicly_available(self):
        observation = self._observation()
        observation.opponent_active["types"] = ["normal"]
        observation.can_terastallize = True
        observation.can_mega_evolve = True
        selector = StrategicActionSelector(random_seed=7)

        legal = selector.legal_actions(
            observation,
            {
                "shadowball-tera": 1.0,
                "sludgebomb-tera": 0.9,
                "sludgebomb-mega": 0.8,
            },
        )

        self.assertEqual(legal, {"sludgebomb-tera": 0.9, "sludgebomb-mega": 0.8})
        observation.can_terastallize = False
        observation.can_mega_evolve = False
        self.assertRaises(
            ValueError,
            selector.legal_actions,
            observation,
            {"sludgebomb-tera": 0.9, "sludgebomb-mega": 0.8},
        )


if __name__ == "__main__":
    unittest.main()