"""Unittest coverage for Stratagem's multi-signal reward construction."""

import unittest

from fp.stratagem.learning.rewards import RewardBuilder, RewardSignals


class RewardBuilderTests(unittest.TestCase):
    def test_combines_terminal_strategy_engine_and_prediction_signals(self):
        reward = RewardBuilder.build(
            RewardSignals(
                terminal_outcome=1.0,
                strategic_progress=0.4,
                engine_action_value=0.75,
                opponent_action_probability=0.8,
                sequence_action_probability=0.6,
            )
        )

        self.assertAlmostEqual(reward.total, 0.715)
        self.assertEqual(reward.terminal, 1.0)
        self.assertAlmostEqual(reward.engine, 0.5)
        self.assertAlmostEqual(reward.opponent_prediction, 0.6)
        self.assertAlmostEqual(reward.sequence_prediction, 0.2)

    def test_missing_optional_signals_are_renormalized_not_fabricated(self):
        reward = RewardBuilder.build(
            RewardSignals(terminal_outcome=-1.0, strategic_progress=0.2)
        )

        self.assertAlmostEqual(reward.total, -0.6)
        self.assertIsNone(reward.engine)
        self.assertIsNone(reward.opponent_prediction)
        self.assertIsNone(reward.sequence_prediction)

    def test_rejects_invalid_prediction_probabilities(self):
        with self.assertRaises(ValueError):
            RewardBuilder.build(
                RewardSignals(
                    terminal_outcome=None,
                    strategic_progress=0.0,
                    opponent_action_probability=1.1,
                )
            )


if __name__ == "__main__":
    unittest.main()