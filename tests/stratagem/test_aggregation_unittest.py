"""Unittest coverage for belief-weighted multi-world aggregation."""

import math
import unittest
from unittest.mock import patch

from fp.stratagem.engine.aggregation import WorldAggregator
from fp.stratagem.inference import Belief, TeamSampler


class EmptySets:
    pkmn_sets = {}


class AggregationTests(unittest.TestCase):
    def test_all_eliminated_worlds_revert_to_uniform_weights(self):
        belief = Belief(TeamSampler(EmptySets()), world_count=2)
        belief.worlds = [[{"species": "charizard"}], [{"species": "blastoise"}]]
        belief.weights = [-math.inf, -math.inf]
        belief._evidence_applied = True

        self.assertEqual(belief.get_world_weights(), [0.5, 0.5])

    def test_uses_belief_weights_and_real_visits_once_per_duplicate_world(self):
        belief = Belief(TeamSampler(EmptySets()), world_count=3)
        first_world = [{"species": "charizard"}]
        second_world = [{"species": "blastoise"}]
        belief.worlds = [first_world, first_world, second_world]
        belief.weights = [math.log(0.5), math.log(0.25), math.log(0.25)]
        belief._evidence_applied = True

        with patch(
            "fp.stratagem.engine.aggregation.search_world_with_visits",
            side_effect=[
                ({"flamethrower": 0.8, "switch blastoise": 0.2}, {"flamethrower": 8, "switch blastoise": 2}),
                ({"flamethrower": 0.4, "switch blastoise": 0.6}, {"flamethrower": 4, "switch blastoise": 6}),
            ],
        ) as search:
            scores, visits = WorldAggregator(belief).aggregate_worlds_with_visits(
                observation=object(),
                our_team=[],
                search_time_ms=1,
                prediction_horizon=0,
            )

        self.assertEqual(search.call_count, 2)
        self.assertAlmostEqual(scores["flamethrower"], 0.7)
        self.assertAlmostEqual(scores["switch blastoise"], 0.3)
        self.assertEqual(visits["flamethrower"], 12)
        self.assertEqual(visits["switch blastoise"], 8)


if __name__ == "__main__":
    unittest.main()