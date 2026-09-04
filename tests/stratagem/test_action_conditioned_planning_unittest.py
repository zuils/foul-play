"""Real poke-engine coverage for action-conditioned future planning."""

import math
import unittest
from copy import deepcopy

from poke_engine import monte_carlo_tree_search

from fp.battle.state import Battle, Pokemon
from fp.config import FoulPlayConfig
from fp.modes.standard_battle import StandardBattleMode
from fp.stratagem.core import Observation
from fp.stratagem.engine.adapter import build_state
from fp.stratagem.engine.aggregation import WorldAggregator
from fp.stratagem.engine.planning import ActionConditionedPlanner
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


class ActionConditionedPlanningTests(unittest.TestCase):
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
        self.first_world = [
            team_spec("charizard", ["flamethrower", "airslash"]),
            team_spec("gengar", ["shadowball"]),
            team_spec("pikachu", ["thunderbolt"]),
            team_spec("blastoise", ["surf"]),
            team_spec("venusaur", ["gigadrain"]),
            team_spec("machamp", ["closecombat"]),
        ]
        self.second_world = [
            team_spec("charizard", ["fireblast", "airslash"]),
            team_spec("machamp", ["closecombat"]),
            team_spec("venusaur", ["gigadrain"]),
            team_spec("blastoise", ["surf"]),
            team_spec("pikachu", ["thunderbolt"]),
            team_spec("gengar", ["shadowball"]),
        ]

    def tearDown(self):
        FoulPlayConfig.pokemon_format = self.previous_format

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

    def _belief(self):
        belief = Belief(TeamSampler(NoopSets()), world_count=2, random_seed=7)
        belief.worlds = [self.first_world, self.second_world]
        belief.weights = [math.log(0.75), math.log(0.25)]
        belief._evidence_applied = True
        return belief

    def test_joint_branches_preserve_both_opponent_response_probabilities(self):
        planner = ActionConditionedPlanner(
            1, 1,
            # Expert/debug parameters - secondary to adaptive budgeting
            _future_rollouts=1,
            _future_action_limit=3,
            _future_candidate_limit=3,
            random_seed=3
        )
        state = build_state(self._observation(), self.first_world, self.our_team)

        branches = planner._joint_transition_distribution(
            state,
            {"shadowball": 0.7, "sludgebomb": 0.3},
            {"flamethrower": 0.6, "airslash": 0.4},
        )
        probability_by_joint_action = {}
        for branch in branches:
            joint_action = (branch.side_one_action, branch.side_two_action)
            probability_by_joint_action[joint_action] = (
                probability_by_joint_action.get(joint_action, 0.0)
                + branch.probability
            )

        self.assertAlmostEqual(sum(branch.probability for branch in branches), 1.0)
        self.assertAlmostEqual(probability_by_joint_action[("shadowball", "flamethrower")], 0.42)
        self.assertAlmostEqual(probability_by_joint_action[("shadowball", "airslash")], 0.28)
        self.assertAlmostEqual(probability_by_joint_action[("sludgebomb", "flamethrower")], 0.18)
        self.assertAlmostEqual(probability_by_joint_action[("sludgebomb", "airslash")], 0.12)

    def test_future_values_are_consumed_by_aggregated_action_ranking(self):
        aggregator = WorldAggregator(self._belief())
        root_scores, _, root_details = aggregator.aggregate_worlds_with_details(
            self._observation(), self.our_team, 1, prediction_horizon=0
        )
        scores, _, details = aggregator.aggregate_worlds_with_details(
            self._observation(),
            self.our_team,
            10,
            prediction_horizon=2,
            future_rollouts=1,
            future_action_limit=2,
            future_candidate_limit=3,
            random_seed=5,
        )

        planned = [detail for detail in details.values() if detail.planned]
        self.assertTrue(planned)
        self.assertTrue(any(detail.branch_count > 0 for detail in planned))
        for action, detail in details.items():
            self.assertAlmostEqual(scores[action], detail.final_value)
            self.assertAlmostEqual(
                detail.final_value,
                detail.current_engine_value + detail.future_delta,
            )
        self.assertTrue(
            any(
                abs(scores[action] - root_scores[action]) > 1e-9
                for action in scores
                if action in root_scores and details[action].planned
            )
        )
        ranking_changed = False
        for random_seed in range(5, 9):
            sampled_scores, _, sampled_details = aggregator.aggregate_worlds_with_details(
                self._observation(),
                self.our_team,
                10,
                prediction_horizon=2,
                future_rollouts=1,
                future_action_limit=2,
                future_candidate_limit=3,
                random_seed=random_seed,
            )
            root_leader = max(
                sampled_details,
                key=lambda action: sampled_details[action].current_engine_value,
            )
            future_leader = max(sampled_scores, key=sampled_scores.get)
            if root_leader != future_leader:
                ranking_changed = True
                break
        self.assertTrue(ranking_changed)
        self.assertEqual(
            {action: detail.current_engine_value for action, detail in root_details.items()},
            root_scores,
        )

    def test_hypothetical_candidate_evaluation_does_not_mutate_root_state(self):
        planner = ActionConditionedPlanner(
            1, 1,
            # Expert/debug parameters - secondary to adaptive budgeting
            _future_rollouts=1,
            _future_action_limit=2,
            _future_candidate_limit=3,
            random_seed=11
        )
        state = build_state(self._observation(), self.first_world, self.our_team)
        source_state = state.to_string()
        result = monte_carlo_tree_search(state, 1)
        candidate = next(
            option.move_choice
            for option in result.side_one
            if option.move_choice == "shadowball" and option.visits > 0
        )

        future_value, branch_count = planner._evaluate_current_action(state, result, candidate)

        self.assertIsNotNone(future_value)
        self.assertGreater(branch_count, 0)
        self.assertEqual(state.to_string(), source_state)

    def test_future_planning_evaluates_an_actual_mcts_switch_candidate(self):
        planner = ActionConditionedPlanner(
            10,
            1,
            _future_rollouts=1,
            _future_action_limit=2,
            _future_candidate_limit=3,
            random_seed=19,
        )
        state = build_state(self._observation(), self.first_world, self.our_team)
        result = monte_carlo_tree_search(state, 10)
        switch_action = next(
            option.move_choice
            for option in result.side_one
            if option.move_choice.startswith("switch ") and option.visits > 0
        )

        future_value, branch_count = planner._evaluate_current_action(
            state, result, switch_action
        )

        self.assertIsNotNone(future_value)
        self.assertGreater(branch_count, 0)

    def test_diagnostic_sequence_does_not_mutate_observation(self):
        observation = self._observation()
        before = deepcopy(observation.to_dict())
        model = OpponentModel(
            self._belief(), our_team=self.our_team, search_time_ms=1, random_seed=17
        )

        model.predict_sequence(observation, horizon=1)

        self.assertEqual(observation.to_dict(), before)


if __name__ == "__main__":
    unittest.main()