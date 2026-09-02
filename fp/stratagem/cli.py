"""
Command-line interface for Stratagem system.
"""

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence

from fp.config import FoulPlayConfig
from fp.format_spec import FormatSpec
from fp.stratagem.core import Observation
from fp.stratagem.inference import Belief, OpponentModel
from fp.stratagem.inference.team_sampler import TeamSampler
from fp.stratagem.learning.model import StratagemModelWrapper
from fp.stratagem.learning.trainer import StratagemTrainer
from fp.stratagem.replay import SelfPlayReplay, load_replay, save_replay
from fp.stratagem.training import run_parallel_self_play
from fp.data.sets import SmogonSets
from fp.teams.team_converter import export_to_dict

from .config import CONFIG, update_config


def train_command(args) -> None:
    """Train a model through local, simultaneous poke-engine self-play."""
    update_config(
        format=args.format,
        world_count=args.worlds,
        search_time_ms=args.search_time_ms,
        prediction_horizon=args.prediction_horizon,
        max_turns=args.max_turns,
        parallel_games=args.parallel_games,
        team_path=args.team_path,
        seed=args.seed,
        verbose=args.verbose,
        checkpoint_frequency=args.checkpoint_frequency,
        hidden_size=args.hidden_size,
        weights_path=args.weights,
    )
    FoulPlayConfig.pokemon_format = args.format
    side_one_team, side_two_team = _select_training_teams(args.team_path, args.format, args.seed)
    side_one_worlds = _sample_candidate_worlds(
        side_two_team, args.format, args.worlds, args.seed
    )
    side_two_worlds = _sample_candidate_worlds(
        side_one_team, args.format, args.worlds, None if args.seed is None else args.seed + 1
    )
    model = StratagemModelWrapper(hidden_sizes=(args.hidden_size,))
    trainer = StratagemTrainer(
        model,
        learning_rate=CONFIG.learning_rate,
        batch_size=CONFIG.batch_size,
        experience_buffer_size=CONFIG.buffer_size,
    )
    if args.weights:
        trainer.load_checkpoint(args.weights)

    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else _default_checkpoint_path()
    replay_directory = Path(args.replay_dir) if args.replay_dir else None
    completed_games = 0
    latest_losses: Dict[str, float] = {}
    while completed_games < args.games:
        batch_games = min(args.checkpoint_frequency, args.games - completed_games)
        batch = run_parallel_self_play(
            side_one_team,
            side_two_team,
            side_one_worlds,
            side_two_worlds,
            games=batch_games,
            parallel_games=args.parallel_games,
            max_turns=args.max_turns,
            search_time_ms=args.search_time_ms,
            seed=None if args.seed is None else args.seed + completed_games,
            side_one_learned_model=model,
            side_two_learned_model=model,
        )
        for game_index, result in enumerate(batch.results, start=completed_games + 1):
            side_one_rewards = result.add_to_trainer(trainer, side_one=True)
            side_two_rewards = result.add_to_trainer(trainer, side_one=False)
            trainer.total_reward += sum(reward.total for reward in side_one_rewards)
            trainer.total_reward += sum(reward.total for reward in side_two_rewards)
            if replay_directory is not None:
                save_replay(
                    replay_directory / f"game-{game_index:06d}.json",
                    SelfPlayReplay.from_result(
                        result,
                        format_name=args.format,
                        seed=batch.seeds[game_index - completed_games - 1],
                        true_teams={"side_one": side_one_team, "side_two": side_two_team},
                        hidden_worlds={
                            "side_one": side_one_worlds,
                            "side_two": side_two_worlds,
                        },
                    ),
                )
        trainer.episode_count += len(batch.results)
        latest_losses = trainer.train_epoch()
        completed_games += len(batch.results)
        trainer.save_checkpoint(checkpoint_path)

    print(
        "Training complete: "
        f"games={completed_games} checkpoint={checkpoint_path} "
        f"loss={latest_losses.get('total_loss', 0.0):.6f}"
    )


def _select_training_teams(
    team_path: str | None, format_name: str, seed: int | None
) -> tuple[List[Dict], List[Dict]]:
    """Choose two six-Pokemon teams from an explicit or bundled local path."""
    root = Path(team_path) if team_path else _default_team_directory(format_name)
    if root.is_file():
        team_files = [root]
    elif root.is_dir():
        team_files = sorted(
            path for path in root.rglob("*") if path.is_file() and not path.name.startswith(".")
        )
    else:
        raise FileNotFoundError(f"Training team path does not exist: {root}")
    if not team_files:
        raise ValueError(f"Training team path contains no team files: {root}")
    rng = random.Random(seed)
    return (
        _load_team_specs(rng.choice(team_files)),
        _load_team_specs(rng.choice(team_files)),
    )


def _default_team_directory(format_name: str) -> Path:
    format_spec = FormatSpec.from_format_string(format_name)
    suffix = format_name.removeprefix(format_spec.gen_string)
    return Path(__file__).resolve().parents[1] / "teams" / "teams" / format_spec.gen_string / suffix


def _load_team_specs(path: Path) -> List[Dict]:
    """Convert an existing Foul Play export into the engine adapter's team schema."""
    members = export_to_dict(path.read_text(encoding="utf-8"))
    if len(members) != 6:
        raise ValueError(f"Training team must contain exactly six Pokemon: {path}")
    stat_names = ("hp", "atk", "def", "spa", "spd", "spe")
    return [
        {
            "species": member["species"],
            "item": member["item"] or None,
            "ability": member["ability"] or None,
            "nature": member["nature"] or "serious",
            "evs": [int(member["evs"].get(stat) or 0) for stat in stat_names],
            "moves": member["moves"],
            "tera_type": member["tera_type"] or None,
            "level": int(member["level"] or 100),
        }
        for member in members
    ]


def _sample_candidate_worlds(
    opponent_team: Sequence[Dict],
    format_name: str,
    world_count: int,
    seed: int | None,
) -> List[List[Dict]]:
    """Sample hidden sets using only the opponent's public team-preview species."""
    if len(opponent_team) != 6:
        raise ValueError("Candidate-world sampling requires exactly six public species")
    public_opponents = [
        {"name": member["species"], "revealed": True, "moves": [], "item": "unknownitem"}
        for member in opponent_team
    ]
    preview = Observation.from_public_snapshot(
        player="user",
        active_pokemon={},
        reserve_pokemon=[],
        hidden_reserve_count=0,
        opponent_active=public_opponents[0],
        opponent_reserve_revealed=public_opponents[1:],
        opponent_hidden_reserve_count=0,
        team_preview=True,
    )
    return _sample_worlds_for_observation(preview, format_name, world_count, seed)


def _sample_worlds_for_observation(
    observation: Observation,
    format_name: str,
    world_count: int,
    seed: int | None,
) -> List[List[Dict]]:
    """Sample set hypotheses from public opponent observations only."""
    visible_opponents = [
        pokemon
        for pokemon in [
            observation.opponent_active,
            *observation.opponent_reserve_revealed,
        ]
        if pokemon.get("revealed") and pokemon.get("name")
    ]
    # Sampling unknown reserves requires the complete usage pool. At team preview
    # all six species are public, so limiting the source data remains sufficient.
    species_filter = (
        {pokemon["name"] for pokemon in visible_opponents}
        if observation.opponent_hidden_reserve_count == 0
        else set()
    )
    smogon_sets = SmogonSets()
    smogon_sets.initialize(FormatSpec.from_format_string(format_name), species_filter)
    return TeamSampler(smogon_sets, random_seed=seed).sample_multiple_teams(
        observation, world_count
    )


def _default_checkpoint_path() -> Path:
    return Path(__file__).resolve().parent / "weights" / "latest.pt"


def play_command(args) -> None:
    """Start an authenticated Foul Play session with Stratagem action selection."""
    from fp.main import run_foul_play

    runtime_args = [
        "--websocket-uri", args.websocket_uri,
        "--ps-username", args.ps_username,
        "--bot-mode", "search_ladder" if args.opponent == "ladder" else "challenge_user",
        "--pokemon-format", args.format,
        "--team-name", args.team,
        "--run-count", str(args.run_count),
        "--stratagem",
        "--stratagem-worlds", str(args.worlds),
        "--stratagem-search-time", str(args.search_time_ms),
        "--stratagem-prediction-horizon", str(args.prediction_horizon),
        "--stratagem-temperature", str(args.temperature),
    ]
    if args.ps_password is not None:
        runtime_args.extend(("--ps-password", args.ps_password))
    if args.opponent != "ladder":
        runtime_args.extend(("--user-to-challenge", args.opponent))
    if args.weights is not None:
        runtime_args.extend(("--stratagem-weights", args.weights))
    asyncio.run(run_foul_play(runtime_args))


def evaluate_command(args) -> None:
    """Evaluate a trained checkpoint against pure-MCTS local self-play agents."""
    update_config(
        format=args.format,
        world_count=args.worlds,
        search_time_ms=args.search_time_ms,
        max_turns=args.max_turns,
        parallel_games=args.parallel_games,
        team_path=args.team_path,
        seed=args.seed,
        weights_path=args.checkpoint,
    )
    FoulPlayConfig.pokemon_format = args.format
    side_one_team, side_two_team = _select_training_teams(args.team_path, args.format, args.seed)
    side_one_worlds = _sample_candidate_worlds(
        side_two_team, args.format, args.worlds, args.seed
    )
    side_two_worlds = _sample_candidate_worlds(
        side_one_team, args.format, args.worlds, None if args.seed is None else args.seed + 1
    )
    model = StratagemModelWrapper(model_path=args.checkpoint)
    batch = run_parallel_self_play(
        side_one_team,
        side_two_team,
        side_one_worlds,
        side_two_worlds,
        games=args.games,
        parallel_games=args.parallel_games,
        max_turns=args.max_turns,
        search_time_ms=args.search_time_ms,
        seed=args.seed,
        side_one_learned_model=model,
    )
    winners = [result.winner for result in batch.results]
    print(
        json.dumps(
            {
                "games": len(batch.results),
                "side_one_wins": winners.count("side_one"),
                "side_two_wins": winners.count("side_two"),
                "draws": winners.count("draw"),
                "turn_caps": sum(result.reached_turn_cap for result in batch.results),
            },
            sort_keys=True,
        )
    )


def replay_command(args) -> None:
    """Load, validate, and emit a persisted offline replay trace."""
    replay = load_replay(args.replay_file)
    print(json.dumps(replay.to_dict(), sort_keys=True))


def inspect_worlds_command(args) -> None:
    """Generate weighted hidden-team hypotheses from a public replay turn."""
    replay, observation, _own_team, replay_turn = _load_replay_context(args.battle_state)
    worlds = _sample_worlds_for_observation(
        observation, replay.format, args.world_count, args.seed
    )
    belief = _belief_from_worlds(worlds, observation, args.seed)
    print(
        json.dumps(
            {
                "format": replay.format,
                "turn": replay_turn.turn,
                "weights": belief.get_world_weights(),
                "worlds": worlds,
            },
            sort_keys=True,
        )
    )


def predict_command(args) -> None:
    """Predict the recorded opponent's action and configurable future sequence."""
    replay, observation, own_team, replay_turn = _load_replay_context(args.battle_state)
    worlds = _sample_worlds_for_observation(
        observation, replay.format, args.world_count, args.seed
    )
    belief = _belief_from_worlds(worlds, observation, args.seed)
    model = OpponentModel(
        belief,
        our_team=own_team,
        search_time_ms=args.search_time_ms,
        random_seed=args.seed,
    )
    current_action = model.predict_move(observation)
    current_action["observed_action"] = replay_turn.side_two_action
    current_action["observed_action_probability"] = model.score_prediction(
        current_action, replay_turn.side_two_action
    )
    print(
        json.dumps(
            {
                "format": replay.format,
                "turn": replay_turn.turn,
                "current_action": current_action,
                "sequence": model.predict_sequence(observation, horizon=args.horizon),
            },
            sort_keys=True,
        )
    )


def _load_replay_context(
    replay_path: str,
) -> tuple[SelfPlayReplay, Observation, List[Dict], object]:
    """Build a side-one decision context from the latest stored public turn."""
    replay = load_replay(replay_path)
    FoulPlayConfig.pokemon_format = replay.format
    if not replay.turns:
        raise ValueError("Replay has no turns to inspect or predict")
    own_team = replay.true_teams.get("side_one")
    if not isinstance(own_team, list) or len(own_team) != 6:
        raise ValueError("Replay does not contain a six-Pokemon side_one team")
    replay_turn = replay.turns[-1]
    return (
        replay,
        _observation_from_replay_dict(replay_turn.side_one_public_observation),
        own_team,
        replay_turn,
    )


def _observation_from_replay_dict(value: Dict) -> Observation:
    """Recreate the public Observation contract stored in a replay turn."""
    required_keys = {
        "active_pokemon",
        "reserve_pokemon",
        "hidden_reserve_count",
        "opponent_active",
        "opponent_reserve_revealed",
        "opponent_hidden_reserve_count",
    }
    missing = required_keys.difference(value)
    if missing:
        raise ValueError(f"Replay public observation is missing keys: {sorted(missing)}")
    return Observation.from_public_snapshot(
        player="user",
        active_pokemon=value["active_pokemon"],
        reserve_pokemon=value["reserve_pokemon"],
        hidden_reserve_count=value["hidden_reserve_count"],
        opponent_active=value["opponent_active"],
        opponent_reserve_revealed=value["opponent_reserve_revealed"],
        opponent_hidden_reserve_count=value["opponent_hidden_reserve_count"],
        weather=value.get("weather"),
        weather_turns_remaining=value.get("weather_turns_remaining", 0),
        field=value.get("field"),
        field_turns_remaining=value.get("field_turns_remaining", 0),
        trick_room=value.get("trick_room", False),
        trick_room_turns_remaining=value.get("trick_room_turns_remaining", 0),
        side_conditions=value.get("side_conditions", {}),
        opponent_side_conditions=value.get("opponent_side_conditions", {}),
        turn=value.get("turn", 0),
        team_preview=value.get("team_preview", False),
        available_moves=value.get("available_moves", []),
        disabled_moves=value.get("disabled_moves", []),
        last_used_move=value.get("last_used_move"),
        active_effective_speed=value.get("active_effective_speed"),
        opponent_effective_speed=value.get("opponent_effective_speed"),
        active_hp_change=value.get("active_hp_change"),
        opponent_hp_change=value.get("opponent_hp_change"),
        active_is_choice_locked=value.get("active_is_choice_locked", False),
        opponent_is_choice_locked=value.get("opponent_is_choice_locked", False),
        active_locked_move=value.get("active_locked_move"),
        opponent_locked_move=value.get("opponent_locked_move"),
    )


def _belief_from_worlds(
    worlds: List[List[Dict]], observation: Observation, seed: int | None
) -> Belief:
    """Apply the recorded public evidence to independently sampled worlds."""
    belief = Belief(team_sampler=None, world_count=len(worlds), random_seed=seed)
    belief.worlds = worlds
    belief.weights = [0.0] * len(worlds)
    belief.update_with_evidence(observation)
    return belief


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Stratagem - Self-learning Pokemon battle AI"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train Stratagem through self-play")
    train_parser.add_argument(
        "--games", type=int, default=1000, help="Number of training games (default: 1000)"
    )
    train_parser.add_argument(
        "--format",
        type=str,
        default="gen9ou",
        help="Battle format used by Foul Play and Smogon data (default: gen9ou)",
    )
    train_parser.add_argument(
        "--parallel-games",
        type=int,
        default=1,
        help="Number of parallel games (default: 1)",
    )
    train_parser.add_argument(
        "--worlds",
        type=int,
        default=32,
        help="Number of worlds to sample per decision (default: 32)",
    )
    train_parser.add_argument(
        "--search-time-ms",
        type=int,
        default=150,
        help="Search time per world in milliseconds (default: 150)",
    )
    train_parser.add_argument(
        "--prediction-horizon",
        type=int,
        default=3,
        help="Prediction horizon for opponent moves (default: 3)",
    )
    train_parser.add_argument(
        "--max-turns",
        type=int,
        default=100,
        help="Maximum turns per game (default: 100)",
    )
    train_parser.add_argument(
        "--team-path",
        type=str,
        default=None,
        help="Path to team file or directory (default: use built-in teams)",
    )
    train_parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to a compatible checkpoint to resume (default: new model)",
    )
    train_parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Destination checkpoint path (default: fp/stratagem/weights/latest.pt)",
    )
    train_parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=100,
        help="Games between central training/checkpoint batches (default: 100)",
    )
    train_parser.add_argument(
        "--replay-dir",
        type=str,
        default=None,
        help="Optional directory for versioned local replay traces",
    )
    train_parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="Hidden layer width for a newly initialized model (default: 128)",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: random)",
    )
    train_parser.set_defaults(func=train_command)

    # Play command
    play_parser = subparsers.add_parser("play", help="Play against Stratagem")
    play_parser.add_argument(
        "--opponent",
        type=str,
        required=True,
        help="Username to challenge or 'ladder' for ladder play",
    )
    play_parser.add_argument(
        "--team",
        type=str,
        required=True,
        help="Team to use for battles",
    )
    play_parser.add_argument(
        "--format",
        type=str,
        default="gen9randombattle",
        help="Battle format (default: gen9randombattle)",
    )
    play_parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to weights file (default: use latest checkpoint)",
    )
    play_parser.add_argument("--websocket-uri", required=True)
    play_parser.add_argument("--ps-username", required=True)
    play_parser.add_argument("--ps-password", default=None)
    play_parser.add_argument("--run-count", type=int, default=1)
    play_parser.add_argument("--worlds", type=int, default=32)
    play_parser.add_argument("--search-time-ms", type=int, default=150)
    play_parser.add_argument("--prediction-horizon", type=int, default=3)
    play_parser.add_argument("--temperature", type=float, default=1.0)
    play_parser.set_defaults(func=play_command)

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate Stratagem performance")
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint to evaluate",
    )
    eval_parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Number of evaluation games (default: 100)",
    )
    eval_parser.add_argument(
        "--parallel-games",
        type=int,
        default=1,
        help="Number of parallel games (default: 1)",
    )
    eval_parser.add_argument("--format", type=str, default="gen9ou")
    eval_parser.add_argument("--team-path", type=str, default=None)
    eval_parser.add_argument("--worlds", type=int, default=32)
    eval_parser.add_argument("--search-time-ms", type=int, default=150)
    eval_parser.add_argument("--max-turns", type=int, default=100)
    eval_parser.add_argument("--seed", type=int, default=None)
    eval_parser.set_defaults(func=evaluate_command)

    # Replay command
    replay_parser = subparsers.add_parser("replay", help="Replay a saved battle")
    replay_parser.add_argument(
        "--replay-file", type=str, required=True, help="Path to replay file"
    )
    replay_parser.set_defaults(func=replay_command)

    # Inspect worlds command
    inspect_parser = subparsers.add_parser(
        "inspect-worlds", help="Inspect world generation for a battle state"
    )
    inspect_parser.add_argument(
        "--battle-state",
        type=str,
        required=True,
        help="Path to a versioned local self-play replay JSON file",
    )
    inspect_parser.add_argument(
        "--world-count",
        type=int,
        default=10,
        help="Number of worlds to generate (default: 10)",
    )
    inspect_parser.add_argument("--seed", type=int, default=None)
    inspect_parser.set_defaults(func=inspect_worlds_command)

    # Predict command
    predict_parser = subparsers.add_parser(
        "predict", help="Make predictions for a battle state"
    )
    predict_parser.add_argument(
        "--battle-state",
        type=str,
        required=True,
        help="Path to a versioned local self-play replay JSON file",
    )
    predict_parser.add_argument(
        "--horizon",
        type=int,
        default=3,
        help="Prediction horizon (default: 3)",
    )
    predict_parser.add_argument(
        "--world-count", type=int, default=10, help="Number of sampled worlds (default: 10)"
    )
    predict_parser.add_argument(
        "--search-time-ms", type=int, default=150, help="MCTS time per world (default: 150)"
    )
    predict_parser.add_argument("--seed", type=int, default=None)
    predict_parser.set_defaults(func=predict_command)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Update global verbose setting
    if args.verbose:
        update_config(verbose=True)

    # Call the appropriate command function
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()