"""Unittest coverage for engine-only local Stratagem self-play."""

import inspect
import unittest
from unittest.mock import patch

from fp.config import FoulPlayConfig
from fp.stratagem.learning.model import StratagemModelWrapper
from fp.stratagem.learning.trainer import StratagemTrainer
from fp.stratagem.replay import SelfPlayReplay
from fp.stratagem.training.self_play import (
    LocalSelfPlayGame,
    MctsSelfPlayAgent,
    SelfPlayDecision,
)
from poke_engine import generate_instructions, monte_carlo_tree_search


def team(species_prefix: str) -> list[dict]:
    species = ["pikachu", "charizard", "blastoise", "venusaur", "machamp", "gengar"]
    return [
        {
            "species": name,
            "item": "leftovers" if index == 0 else "lifeorb",
            "ability": "blaze" if name == "charizard" else "static",
            "nature": "serious",
            "evs": [0, 0, 0, 0, 0, 0],
            "moves": ["tackle"],
            "tera_type": "normal",
            "level": 50,
            "label": f"{species_prefix}-{index}",
        }
        for index, name in enumerate(species)
    ]


class RecordingAgent:
    def __init__(self, action: str = "tackle"):
        self.action = action
        self.observations = []

    def select_action(self, observation):
        self.observations.append(observation)
        self.assert_public_snapshot(observation)
        return self.action

    @staticmethod
    def assert_public_snapshot(observation):
        assert not hasattr(observation, "_state")
        assert not hasattr(observation, "_battle")


class MctsActionAgent:
    """Test agent that replays an exact legal action returned by real MCTS."""

    def __init__(self, action: str):
        self.action = action

    def select_decision(self, observation):
        return SelfPlayDecision(self.action, {self.action: 1.0})


class LocalSelfPlayTests(unittest.TestCase):
    def setUp(self):
        self.previous_format = FoulPlayConfig.pokemon_format
        FoulPlayConfig.pokemon_format = "gen9ou"
        self.side_one_team = team("one")
        self.side_two_team = team("two")

    def tearDown(self):
        FoulPlayConfig.pokemon_format = self.previous_format

    def make_game(self, side_one_agent, side_two_agent, **kwargs):
        return LocalSelfPlayGame.from_teams(
            self.side_one_team,
            self.side_two_team,
            side_one_agent,
            side_two_agent,
            random_seed=7,
            **kwargs,
        )

    def test_local_game_resolves_after_private_simultaneous_selection(self):
        side_one_agent = RecordingAgent()
        side_two_agent = RecordingAgent()
        game = self.make_game(side_one_agent, side_two_agent, max_turns=1)

        with patch(
            "fp.stratagem.training.self_play.generate_instructions",
            wraps=generate_instructions,
        ) as engine_transition:
            result = game.run()

        self.assertEqual(result.turns, 1)
        self.assertTrue(result.reached_turn_cap)
        self.assertEqual(result.actions, (("tackle", "tackle"),))
        engine_transition.assert_called_once()
        self.assertEqual(len(side_one_agent.observations), 1)
        self.assertEqual(len(side_two_agent.observations), 1)
        self.assertEqual(
            side_two_agent.observations[0].opponent_active["hp"],
            side_two_agent.observations[0].opponent_active["max_hp"],
        )

    def test_public_snapshot_hides_true_opponent_team_information(self):
        game = self.make_game(RecordingAgent(), RecordingAgent(), max_turns=1)

        observation = game._public_observation(side_one=True, turn=1)

        self.assertNotIn("leftovers", repr(observation.opponent_active))
        self.assertNotIn("blaze", repr(observation.opponent_active))
        self.assertNotIn("two-1", repr(observation.opponent_active))
        self.assertEqual(observation.opponent_active["item"], "unknownitem")
        self.assertIsNone(observation.opponent_active["ability"])
        self.assertEqual(observation.opponent_active["moves"], [])
        self.assertEqual(observation.opponent_hidden_reserve_count, 5)

    def test_mcts_agent_selects_from_public_snapshot_and_candidate_worlds(self):
        agent = MctsSelfPlayAgent(
            self.side_one_team,
            [self.side_two_team],
            search_time_ms=1,
            random_seed=3,
        )
        game = self.make_game(agent, RecordingAgent(), max_turns=1)

        action = agent.select_action(game._public_observation(side_one=True, turn=1))

        self.assertTrue(action == "tackle" or action.startswith("switch "))

    def test_mcts_self_play_turn_adds_normalized_experience_to_trainer(self):
        model = StratagemModelWrapper(hidden_sizes=(8,))
        trainer = StratagemTrainer(model, batch_size=1)
        side_one_agent = MctsSelfPlayAgent(
            self.side_one_team,
            [self.side_two_team],
            search_time_ms=1,
            random_seed=3,
        )
        game = self.make_game(side_one_agent, RecordingAgent(), max_turns=1)

        result = game.run(side_one_trainer=trainer)

        self.assertEqual(len(result.turns_data), 1)
        self.assertEqual(len(trainer.experience_buffer), 1)
        experience = trainer.experience_buffer[0]
        self.assertTrue(experience.done)
        self.assertTrue(experience.action_values)
        self.assertGreaterEqual(experience.reward, -1.0)
        self.assertLessEqual(experience.reward, 1.0)
        self.assertGreater(trainer.train_step()["total_loss"], 0.0)

    def test_mcts_originated_switch_resolves_in_self_play_and_replay(self):
        game = self.make_game(RecordingAgent(), RecordingAgent(), max_turns=1)
        result = monte_carlo_tree_search(game._state, duration_ms=25)
        switch_action = next(
            option.move_choice
            for option in result.side_one
            if option.move_choice.startswith("switch ")
        )
        game._side_one_agent = MctsActionAgent(switch_action)

        completed = game.run()
        replay = SelfPlayReplay.from_result(
            completed,
            format_name="gen9ou",
            seed=7,
            true_teams={"side_one": self.side_one_team, "side_two": self.side_two_team},
        )

        self.assertEqual(completed.actions[0][0], switch_action)
        self.assertEqual(
            str(
                completed.final_state.side_one.pokemon[
                    int(str(completed.final_state.side_one.active_index))
                ].id
            ).lower(),
            switch_action.removeprefix("switch "),
        )
        self.assertEqual(replay.turns[0].side_one_action, switch_action)

    def test_turn_cap_is_hard_bounded_and_self_play_has_no_network_import(self):
        with self.assertRaises(ValueError):
            self.make_game(RecordingAgent(), RecordingAgent(), max_turns=101)

        import fp.stratagem.training.self_play as self_play

        source = inspect.getsource(self_play).lower()
        self.assertNotIn("websocket", source)
        self.assertNotIn("showdown", source)


if __name__ == "__main__":
    unittest.main()