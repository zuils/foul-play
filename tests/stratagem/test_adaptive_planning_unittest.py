"""Deterministic unit coverage for adaptive future-planning allocation."""

import time
import unittest
from types import SimpleNamespace

from fp.stratagem.config import StratagemConfig
from fp.stratagem.engine.planning import ActionConditionedPlanner


def option(action, visits, value):
    return SimpleNamespace(
        move_choice=action,
        visits=visits,
        total_score=visits * value,
    )


class AdaptivePlanningTests(unittest.TestCase):
    def planner(self, budget, horizon=3, **overrides):
        # Extract expert/debug parameters from overrides
        future_rollouts = overrides.pop('future_rollouts', None)
        future_action_limit = overrides.pop('future_action_limit', None)
        future_candidate_limit = overrides.pop('future_candidate_limit', None)
        planner = ActionConditionedPlanner(
            budget, horizon,
            # Expert/debug parameters - secondary to adaptive budgeting
            _future_rollouts=future_rollouts,
            _future_action_limit=future_action_limit,
            _future_candidate_limit=future_candidate_limit,
            **overrides  # Pass any remaining overrides as random_seed, etc.
        )
        planner._deadline = time.perf_counter() + budget / 1000
        return planner

    def test_normal_config_has_only_conceptual_future_controls(self):
        config = StratagemConfig()
        self.assertEqual(config.prediction_horizon, 3)
        self.assertEqual(config.search_time_ms, 150)
        self.assertFalse(hasattr(config, "future_rollouts"))
        self.assertFalse(hasattr(config, "future_action_limit"))
        self.assertFalse(hasattr(config, "future_candidate_limit"))

    def test_policy_width_changes_with_available_budget(self):
        options = [
            option("a", 40, 0.6),
            option("b", 30, 0.5),
            option("c", 20, 0.4),
            option("d", 10, 0.3),
        ]
        narrow = self.planner(4)._policy_distribution(options)
        broad = self.planner(100)._policy_distribution(options)
        self.assertLess(len(narrow), len(broad))
        self.assertNotEqual(len(broad), 3)

    def test_horizon_changes_depth_parameter_not_policy_width(self):
        options = [option("a", 60, 0.6), option("b", 30, 0.5), option("c", 10, 0.4)]
        short = self.planner(100, horizon=1)._policy_distribution(options)
        deep = self.planner(100, horizon=3)._policy_distribution(options)
        self.assertEqual(short, deep)
        self.assertEqual(self.planner(100, horizon=3).prediction_horizon, 3)

    def test_seed_stably_orders_allocation_inputs(self):
        options = [option("a", 60, 0.6), option("b", 30, 0.5), option("c", 10, 0.4)]
        first = self.planner(100)._policy_distribution(options)
        second = self.planner(100)._policy_distribution(options)
        self.assertEqual(first, second)

    def test_candidate_selection_uses_value_overlap_not_fixed_top_three(self):
        planner = self.planner(100)
        candidates = planner._serious_actions(
            {"a": 100, "b": 100, "c": 5, "d": 1},
            {"a": 0.80, "b": 0.79, "c": 0.10, "d": 0.01},
        )
        self.assertEqual(candidates, {"a", "b"})

    def test_close_candidates_receive_more_budget_than_stable_candidates(self):
        planner = self.planner(100, horizon=2)
        self.assertGreater(
            planner._adaptive_rollout_count(0.5),
            planner._adaptive_rollout_count(0.05),
        )

    def test_high_impact_tail_is_retained(self):
        planner = self.planner(4)
        policy = planner._policy_distribution(
            [
                option("safe", 70, 0.5),
                option("good", 20, 0.6),
                option("catastrophic", 1, 0.0),
                option("other", 9, 0.5),
            ]
        )
        self.assertIn("catastrophic", policy)

    def test_expert_rollout_bound_is_secondary_to_adaptive_allocation(self):
        planner = self.planner(100, future_rollouts=2)
        self.assertEqual(planner._adaptive_rollout_count(1.0), 2)
        self.assertNotEqual(planner._adaptive_rollout_count(0.25), 4)


if __name__ == "__main__":
    unittest.main()
