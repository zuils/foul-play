"""Unittest coverage for isolated parallel local self-play execution."""

import unittest

from fp.config import FoulPlayConfig
from fp.stratagem.learning.model import StratagemModelWrapper
from fp.stratagem.learning.trainer import StratagemTrainer
from fp.stratagem.training.workers import run_parallel_self_play
from tests.stratagem.test_local_self_play_unittest import team


class RecordingPriors:
    def __init__(self):
        self.calls = 0

    def get_candidate_action_priors(self, observation, candidate_actions):
        self.calls += 1
        return {action: 1.0 / len(candidate_actions) for action in candidate_actions}


class ParallelSelfPlayTests(unittest.TestCase):
    def setUp(self):
        self.previous_format = FoulPlayConfig.pokemon_format
        FoulPlayConfig.pokemon_format = "gen9ou"
        self.side_one_team = team("one")
        self.side_two_team = team("two")
        self.side_one_world = [{**pokemon, "item": "unknownitem"} for pokemon in self.side_two_team]
        self.side_two_world = [{**pokemon, "item": "unknownitem"} for pokemon in self.side_one_team]

    def tearDown(self):
        FoulPlayConfig.pokemon_format = self.previous_format

    def test_parallel_games_are_seeded_isolated_and_centrally_trainable(self):
        batch = run_parallel_self_play(
            self.side_one_team,
            self.side_two_team,
            [self.side_one_world],
            [self.side_two_world],
            games=2,
            parallel_games=2,
            max_turns=1,
            search_time_ms=1,
            seed=42,
        )

        self.assertEqual(len(batch.results), 2)
        self.assertEqual(len(set(batch.seeds)), 2)
        self.assertTrue(all(result.turns == 1 for result in batch.results))
        self.assertIsNot(batch.results[0].final_state, batch.results[1].final_state)

        trainer = StratagemTrainer(StratagemModelWrapper(hidden_sizes=(8,)), batch_size=1)
        batch.add_to_trainers(side_one_trainer=trainer)
        self.assertEqual(len(trainer.experience_buffer), 2)
        self.assertGreater(trainer.train_step()["total_loss"], 0.0)

    def test_parallel_configuration_is_validated(self):
        with self.assertRaises(ValueError):
            run_parallel_self_play(
                self.side_one_team,
                self.side_two_team,
                [self.side_one_world],
                [self.side_two_world],
                games=1,
                parallel_games=0,
                max_turns=1,
                search_time_ms=1,
            )

    def test_workers_forward_explicit_learned_priors_to_agent_search(self):
        priors = RecordingPriors()
        run_parallel_self_play(
            self.side_one_team,
            self.side_two_team,
            [self.side_one_world],
            [self.side_two_world],
            games=1,
            parallel_games=1,
            max_turns=1,
            search_time_ms=1,
            seed=42,
            side_one_learned_model=priors,
        )

        self.assertGreater(priors.calls, 0)


if __name__ == "__main__":
    unittest.main()