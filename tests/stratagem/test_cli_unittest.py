"""Unittest coverage for local Stratagem CLI commands."""

import io
import json
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from fp.config import FoulPlayConfig
from fp.stratagem import cli
from fp.stratagem.config import get_config, update_config
from fp.stratagem.replay import ReplayTurn, SelfPlayReplay, save_replay
from tests.stratagem.test_local_self_play_unittest import team


def _team_export() -> str:
    return "\n\n".join(
        "\n".join(
            (
                f"{member['species'].title()} @ Leftovers",
                "Ability: Static",
                "Level: 50",
                "Serious Nature",
                "- Tackle",
            )
        )
        for member in team("cli")
    )


class StratagemCliTests(unittest.TestCase):
    def setUp(self):
        self.previous_format = FoulPlayConfig.pokemon_format
        self.previous_config = vars(get_config()).copy()

    def tearDown(self):
        FoulPlayConfig.pokemon_format = self.previous_format
        update_config(**self.previous_config)

    def test_train_runs_local_pipeline_and_persists_checkpoint_and_replay(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            team_path = temporary_path / "team.txt"
            checkpoint_path = temporary_path / "checkpoint.pt"
            replay_directory = temporary_path / "replays"
            team_path.write_text(_team_export(), encoding="utf-8")

            with (
                patch.object(
                    cli,
                    "_sample_candidate_worlds",
                    side_effect=lambda opponent, *_: [
                        [{**member, "item": "unknownitem"} for member in opponent]
                    ],
                ),
                patch.object(
                    sys,
                    "argv",
                    [
                        "stratagem",
                        "train",
                        "--games",
                        "1",
                        "--format",
                        "gen9ou",
                        "--team-path",
                        str(team_path),
                        "--worlds",
                        "1",
                        "--search-time-ms",
                        "1",
                        "--max-turns",
                        "1",
                        "--hidden-size",
                        "8",
                        "--checkpoint-path",
                        str(checkpoint_path),
                        "--replay-dir",
                        str(replay_directory),
                        "--seed",
                        "7",
                    ],
                ),
            ):
                cli.main()

            self.assertTrue(checkpoint_path.is_file())
            self.assertTrue((replay_directory / "game-000001.json").is_file())
            self.assertEqual(get_config().format, "gen9ou")
            self.assertEqual(FoulPlayConfig.pokemon_format, "gen9ou")

    def test_replay_emits_the_strictly_validated_persisted_trace(self):
        replay = SelfPlayReplay(
            schema_version=1,
            format="gen9ou",
            seed=7,
            winner="side_one",
            turn_count=0,
            reached_turn_cap=False,
            turns=(),
            true_teams={"side_one": [], "side_two": []},
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            replay_path = Path(temporary_directory) / "game.json"
            save_replay(replay_path, replay)
            output = io.StringIO()
            with (
                patch.object(sys, "argv", ["stratagem", "replay", "--replay-file", str(replay_path)]),
                redirect_stdout(output),
            ):
                cli.main()

            self.assertEqual(
                json.loads(output.getvalue()), json.loads(json.dumps(replay.to_dict()))
            )

    def test_inspect_worlds_and_predict_use_only_the_replay_public_turn(self):
        side_one_team = team("one")
        side_two_team = team("two")
        public_observation = cli.Observation.from_public_snapshot(
            player="user",
            active_pokemon={"name": "pikachu", "revealed": True},
            reserve_pokemon=[],
            hidden_reserve_count=5,
            opponent_active={"name": "pikachu", "revealed": True, "moves": []},
            opponent_reserve_revealed=[],
            opponent_hidden_reserve_count=5,
            available_moves=["tackle"],
        )
        replay = SelfPlayReplay(
            schema_version=1,
            format="gen9ou",
            seed=7,
            winner=None,
            turn_count=1,
            reached_turn_cap=True,
            turns=(
                ReplayTurn(
                    turn=1,
                    side_one_public_observation=public_observation.to_dict(),
                    side_two_public_observation=public_observation.to_dict(),
                    side_one_action="tackle",
                    side_two_action="tackle",
                    side_one_mcts_values={"tackle": 0.0},
                    side_two_mcts_values={"tackle": 0.0},
                    side_one_strategic_progress=0.0,
                    side_two_strategic_progress=0.0,
                ),
            ),
            true_teams={"side_one": side_one_team, "side_two": side_two_team},
        )
        worlds = [[{**member, "item": "unknownitem"} for member in side_two_team]]
        with tempfile.TemporaryDirectory() as temporary_directory:
            replay_path = Path(temporary_directory) / "game.json"
            save_replay(replay_path, replay)
            with patch.object(cli, "_sample_worlds_for_observation", return_value=worlds):
                inspect_output = self._run_cli(
                    "inspect-worlds", "--battle-state", str(replay_path), "--world-count", "1", "--seed", "7"
                )
                predict_output = self._run_cli(
                    "predict", "--battle-state", str(replay_path), "--world-count", "1",
                    "--search-time-ms", "1", "--horizon", "0", "--seed", "7"
                )

        inspected = json.loads(inspect_output)
        predicted = json.loads(predict_output)
        self.assertEqual(inspected["worlds"], worlds)
        self.assertEqual(predicted["current_action"]["observed_action"], "tackle")
        self.assertEqual(predicted["sequence"]["sequence"], [])

    def test_evaluate_loads_a_checkpoint_and_runs_local_games(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            team_path = temporary_path / "team.txt"
            checkpoint_path = temporary_path / "checkpoint.pt"
            team_path.write_text(_team_export(), encoding="utf-8")
            worlds = [[{**member, "item": "unknownitem"} for member in team("two")]]
            with patch.object(cli, "_sample_candidate_worlds", return_value=worlds):
                training_output = self._run_cli(
                    "train", "--games", "1", "--format", "gen9ou", "--team-path", str(team_path),
                    "--worlds", "1", "--search-time-ms", "1", "--max-turns", "1", "--hidden-size", "8",
                    "--checkpoint-path", str(checkpoint_path), "--seed", "7"
                )
            self.assertIn("Training complete", training_output)
            with patch.object(cli, "_sample_candidate_worlds", return_value=worlds):
                output = self._run_cli(
                    "evaluate", "--checkpoint", str(checkpoint_path), "--games", "1", "--format", "gen9ou",
                    "--team-path", str(team_path), "--worlds", "1", "--search-time-ms", "1",
                    "--max-turns", "1", "--seed", "7"
                )

        results = json.loads(output)
        self.assertEqual(results["games"], 1)
        self.assertEqual(results["turn_caps"], 1)

    def test_play_delegates_authenticated_runtime_arguments_to_foul_play(self):
        args = Namespace(
            opponent="opponent", team="gen9/ou/example", format="gen9ou", weights="model.pt",
            websocket_uri="local", ps_username="bot", ps_password="password", run_count=2,
            worlds=3, search_time_ms=4, prediction_horizon=2, temperature=0.5,
        )
        with patch("fp.main.run_foul_play", new_callable=AsyncMock) as run_foul_play:
            cli.play_command(args)

        runtime_args = run_foul_play.await_args.args[0]
        self.assertIn("--stratagem", runtime_args)
        self.assertIn("--stratagem-weights", runtime_args)
        self.assertIn("--user-to-challenge", runtime_args)
        self.assertIn("opponent", runtime_args)

    @staticmethod
    def _run_cli(*arguments: str) -> str:
        output = io.StringIO()
        with patch.object(sys, "argv", ["stratagem", *arguments]), redirect_stdout(output):
            cli.main()
        return output.getvalue()


if __name__ == "__main__":
    unittest.main()