"""Contract tests for Stratagem's learned model integration."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from fp.battle.state import Battle, Pokemon
from fp.config import FoulPlayConfig
from fp.modes.standard_battle import StandardBattleMode
from fp.stratagem.core import Observation
from fp.stratagem.engine import aggregation
from fp.stratagem.engine.aggregation import WorldAggregator
from fp.stratagem.inference import Belief, TeamSampler
from fp.stratagem.learning.features import StratagemFeatureExtractor
from fp.stratagem.learning.model import StratagemModelWrapper
from fp.stratagem.learning.trainer import StratagemTrainer


class EmptySets:
    pkmn_sets = {}


class KeyedPriors:
    def get_candidate_action_priors(self, observation, candidate_actions):
        return {"tackle": 0.01, "thunderbolt": 0.99}


def make_observation() -> Observation:
    battle = Battle(None)
    battle.generation = "gen9"
    battle.mode = StandardBattleMode()

    user = Pokemon("gengar", level=50, nature="timid", evs=[0, 0, 0, 252, 4, 252])
    user.revealed = True
    user.add_move("shadowball")
    user.add_move("sludgebomb")
    battle.user.active = user

    reserve = Pokemon("pikachu", level=50, nature="timid", evs=[0, 0, 0, 252, 4, 252])
    reserve.revealed = True
    reserve.add_move("thunderbolt")
    battle.user.reserve = [reserve]

    opponent = Pokemon("charizard", level=50, nature="timid", evs=[0, 0, 0, 252, 4, 252])
    opponent.revealed = True
    opponent.add_move("flamethrower")
    battle.opponent.active = opponent
    return Observation(battle)


def make_belief() -> Belief:
    belief = Belief(TeamSampler(EmptySets()), world_count=1)
    belief.worlds = [[{"species": "charizard"}]]
    belief.weights = [0.0]
    belief._evidence_applied = True
    return belief


class LearnedModelContractTests(unittest.TestCase):
    def setUp(self):
        self.previous_format = FoulPlayConfig.pokemon_format
        FoulPlayConfig.pokemon_format = "gen9ou"

    def tearDown(self):
        FoulPlayConfig.pokemon_format = self.previous_format

    def test_real_observation_has_finite_exactly_sized_features(self):
        extractor = StratagemFeatureExtractor()
        features = extractor.extract_features(make_observation())

        self.assertEqual(features.shape, (extractor.feature_size,))
        self.assertTrue(np.isfinite(features).all())

    def test_candidate_action_priors_normalize_over_legal_candidates(self):
        observation = make_observation()
        model = StratagemModelWrapper(hidden_sizes=(8,))
        candidates = ["shadowball", "sludgebomb", "switch pikachu"]

        priors = model.get_candidate_action_priors(observation, candidates)

        self.assertEqual(set(priors), set(candidates))
        self.assertAlmostEqual(sum(priors.values()), 1.0)
        self.assertTrue(all(probability > 0.0 for probability in priors.values()))
        with self.assertRaises(ValueError):
            model.action_to_index("not-a-real-action")

    def test_supplied_priors_change_ranking_but_default_is_pure_mcts(self):
        observation = make_observation()
        mcts_results = ({"tackle": 0.9, "thunderbolt": 0.8}, {"tackle": 9, "thunderbolt": 8})
        with patch(
            "fp.stratagem.engine.aggregation.search_world_with_visits",
            return_value=mcts_results,
        ):
            default_aggregator = WorldAggregator(make_belief())
            default_scores, _ = default_aggregator.aggregate_worlds_with_visits(
                observation, [], search_time_ms=1, prediction_horizon=0
            )
            learned_scores, _ = WorldAggregator(
                make_belief(), learned_model=KeyedPriors()
            ).aggregate_worlds_with_visits(
                observation,
                [],
                search_time_ms=1,
                prediction_horizon=0,
            )

        self.assertIsNone(default_aggregator.learned_model)
        self.assertFalse(hasattr(aggregation, "get_model_wrapper"))
        self.assertGreater(default_scores["tackle"], default_scores["thunderbolt"])
        self.assertGreater(learned_scores["thunderbolt"], learned_scores["tackle"])

    def test_trainer_step_updates_a_parameter_for_a_valid_batch(self):
        model = StratagemModelWrapper(hidden_sizes=(8,))
        trainer = StratagemTrainer(model, batch_size=1)
        before = [parameter.detach().clone() for parameter in model.model.parameters()]

        trainer.add_experience(make_observation(), "shadowball", 0.5, None, True)
        losses = trainer.train_step()

        self.assertGreater(losses["total_loss"], 0.0)
        self.assertTrue(
            any(
                not torch.equal(previous, current)
                for previous, current in zip(before, model.model.parameters())
            )
        )

    def test_trainer_uses_mcts_action_values_as_policy_targets(self):
        model = StratagemModelWrapper(hidden_sizes=(8,))
        trainer = StratagemTrainer(model, batch_size=1)
        trainer.add_experience(
            make_observation(),
            "shadowball",
            0.0,
            None,
            True,
            action_values={"shadowball": 0.1, "sludgebomb": 1.1},
        )

        _, actions, _, _, _, batch = trainer.sample_batch()
        targets = trainer._policy_targets(batch, actions)

        self.assertGreater(
            targets[0, model.action_to_index("sludgebomb")].item(),
            targets[0, model.action_to_index("shadowball")].item(),
        )


if __name__ == "__main__":
    unittest.main()