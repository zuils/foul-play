"""Unittest coverage for weighted and engine-backed opponent predictions."""

import math
import unittest

from fp.battle.state import Battle, Pokemon
from fp.config import FoulPlayConfig
from fp.modes.standard_battle import StandardBattleMode
from fp.stratagem.core import Observation
from fp.stratagem.inference import Belief, OpponentModel, TeamSampler


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


class OpponentModelTests(unittest.TestCase):
    def setUp(self):
        self.previous_format = FoulPlayConfig.pokemon_format
        FoulPlayConfig.pokemon_format = "gen9ou"
        self.our_team = [
            team_spec("gengar", ["shadowball", "sludgebomb"]),
            team_spec("pikachu", ["thunderbolt"]),
            team_spec("blastoise", ["surf"]),
            team_spec("venusaur", ["gigadrain"]),
            team_spec("machamp", ["closecombat"]),
            team_spec("charizard", ["flamethrower"]),
        ]
        self.world_one = [
            team_spec("pikachu", ["thunderbolt"]),
            team_spec("blastoise", ["surf"]),
            team_spec("venusaur", ["gigadrain"]),
            team_spec("machamp", ["closecombat"]),
            team_spec("gengar", ["shadowball"]),
            team_spec("charizard", ["flamethrower", "airslash"]),
        ]
        self.world_two = [
            team_spec("gengar", ["shadowball"]),
            team_spec("pikachu", ["thunderbolt"]),
            team_spec("blastoise", ["surf"]),
            team_spec("venusaur", ["gigadrain"]),
            team_spec("machamp", ["closecombat"]),
            team_spec("charizard", ["fireblast", "airslash"]),
        ]

    def tearDown(self):
        FoulPlayConfig.pokemon_format = self.previous_format

    def _belief(self):
        belief = Belief(TeamSampler(NoopSets()), world_count=2, random_seed=7)
        belief.worlds = [self.world_one, self.world_two]
        belief.weights = [math.log(0.75), math.log(0.25)]
        belief._evidence_applied = True
        return belief

    def _observation(self):
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
        return Observation(battle)

    def test_lead_and_move_predictions_consider_nonzero_world_slots(self):
        observation = self._observation()
        model = OpponentModel(self._belief(), random_seed=11)

        observation.opponent_active = {}
        lead = model.predict_lead(observation)
        charizard_lead = next(
            alternative
            for alternative in lead["alternatives"]
            if alternative["species"] == "charizard"
        )
        self.assertAlmostEqual(charizard_lead["probability"], 1 / 6)

        observation = self._observation()
        move = model.predict_move(observation)
        self.assertEqual(move["move"], "airslash")
        self.assertAlmostEqual(move["confidence"], 0.5)
        self.assertAlmostEqual(model.score_prediction(move, "fireblast"), 0.125)
        self.assertEqual(model.score_prediction(move, "switch pikachu"), 0.0)
        self.assertEqual(move["source"], "belief")

    def test_sequence_uses_engine_rollouts_for_configured_horizon(self):
        model = OpponentModel(
            self._belief(), our_team=self.our_team, search_time_ms=1, random_seed=13
        )

        team_preview_observation = self._observation()
        team_preview_observation.team_preview = True
        team_preview_observation.active_pokemon = {}
        team_preview_observation.opponent_active = {}
        lead_prediction = model.predict_lead(team_preview_observation)
        self.assertEqual(lead_prediction["source"], "mcts")
        self.assertTrue(lead_prediction["alternatives"])

        action_prediction = model.predict_move(self._observation())
        self.assertEqual(action_prediction["source"], "mcts")
        self.assertTrue(action_prediction["alternatives"])
        self.assertAlmostEqual(sum(action_prediction["probabilities"].values()), 1.0)

        prediction = model.predict_sequence(self._observation(), horizon=2)

        self.assertEqual(len(prediction["sequence"]), 2)
        self.assertEqual(prediction["source"], "engine-rollout")
        self.assertTrue(0.0 <= prediction["confidence"] <= 1.0)
        for step in prediction["sequence"]:
            self.assertIn("action", step)
            self.assertTrue(step["alternatives"])
            self.assertGreater(sum(item["probability"] for item in step["alternatives"]), 0.0)
            self.assertLessEqual(sum(item["probability"] for item in step["alternatives"]), 1.0)
            self.assertEqual(step["action"], step["alternatives"][0]["action"])


if __name__ == "__main__":
    unittest.main()