"""Unittest coverage for Stratagem replay traces and control-aware loss mining."""

import tempfile
import unittest
from pathlib import Path

from fp.config import FoulPlayConfig
from fp.stratagem.learning.loss_mining import mine_loss_signatures
from fp.stratagem.replay import ReplayTurn, SelfPlayReplay, load_replay, save_replay
from fp.stratagem.training.self_play import LocalSelfPlayGame
from tests.stratagem.test_local_self_play_unittest import RecordingAgent, team


def loss_mining_replay(winner: str, action: str) -> SelfPlayReplay:
    turn = ReplayTurn(
        turn=1,
        side_one_public_observation={
            "opponent_active": {
                "revealed": True,
                "types": ["ghost"],
                "is_terastallized": False,
                "tera_type": None,
            }
        },
        side_two_public_observation={"opponent_active": {"revealed": True, "types": ["normal"]}},
        side_one_action=action,
        side_two_action="tackle",
        side_one_mcts_values={"tackle": 0.8, "shadowball": 0.1},
        side_two_mcts_values={"tackle": 0.8},
        side_one_strategic_progress=-0.1,
        side_two_strategic_progress=0.1,
    )
    return SelfPlayReplay(
        schema_version=1,
        format="gen9ou",
        seed=1,
        winner=winner,
        turn_count=1,
        reached_turn_cap=False,
        turns=(turn,),
        true_teams={"side_one": [], "side_two": []},
    )


class ReplayAndLossMiningTests(unittest.TestCase):
    def setUp(self):
        self.previous_format = FoulPlayConfig.pokemon_format
        FoulPlayConfig.pokemon_format = "gen9ou"

    def tearDown(self):
        FoulPlayConfig.pokemon_format = self.previous_format

    def test_real_local_trace_round_trips_as_atomic_json(self):
        side_one_team = team("one")
        side_two_team = team("two")
        result = LocalSelfPlayGame.from_teams(
            side_one_team,
            side_two_team,
            RecordingAgent(),
            RecordingAgent(),
            max_turns=1,
            random_seed=3,
        ).run()
        replay = SelfPlayReplay.from_result(
            result,
            format_name="gen9ou",
            seed=3,
            true_teams={"side_one": side_one_team, "side_two": side_two_team},
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "game.json"
            save_replay(path, replay)
            restored = load_replay(path)

            self.assertEqual(restored.to_dict(), replay.to_dict())
            self.assertEqual(list(Path(temporary_directory).glob(".game.json.*")), [])
        public_state = replay.turns[0].side_one_public_observation["opponent_active"]
        self.assertEqual(public_state["item"], "unknownitem")
        self.assertIsNone(public_state["ability"])

    def test_loss_mining_requires_a_win_control_and_distinguishes_immune_moves(self):
        loss = loss_mining_replay("side_two", "tackle")
        win = loss_mining_replay("side_one", "shadowball")

        signatures = mine_loss_signatures(
            [loss, win], min_loss_support=0.5, min_lift=1.5
        )

        self.assertEqual(len(signatures), 1)
        self.assertEqual(signatures[0].name, "side_one:immune_move")
        self.assertEqual(signatures[0].loss_support, 1.0)
        self.assertEqual(signatures[0].win_support, 0.0)

        with self.assertRaises(ValueError):
            mine_loss_signatures([loss], min_loss_support=0.5, min_lift=1.5)


if __name__ == "__main__":
    unittest.main()