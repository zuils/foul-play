# Stratagem

Stratagem is an in-progress Pokémon battle AI built on top of Foul Play and poke-engine. It combines public observations, hidden-opponent-team hypotheses, poke-engine MCTS, and optional learned value/policy components.

## Current Status

Implemented and tested: public observation boundaries, Smogon-backed world sampling and belief updates, poke-engine state adaptation and MCTS aggregation, opponent prediction, strategic action guards, optional learned action priors, local engine-only self-play, reward targets, strict versioned checkpoints, replay/tracing and loss mining, parallel training, and local CLI training/evaluation/replay-inspection/world-inspection/prediction commands.

The `play` command delegates authenticated Showdown sessions to Foul Play's existing runtime and accepts its required websocket and account options. It is covered by a non-network handoff test; controlled-endpoint validation is still pending. Local self-play supports engine-selected voluntary switches through the binding's raw-species transition input. Tera and Mega action strings are engine-supported; Z-move action branching is not exposed by the examined poke-engine `MoveChoice` API.

See `IMPLEMENTATION_PLAN.md` for the verified current tracker.

## Features

- **Hidden Team Modeling**: Maintains belief distributions over complete opponent team hypotheses using Smogon set machinery
- **Multi-world Search**: Samples multiple consistent hidden worlds and runs MCTS independently in each
- **Optional Learned Components**: Provides a trainable value/policy model whose priors are opt-in for live play
- **Multi-step Prediction**: Predicts opponent action sequences with configurable horizons
- **Strategic Guards**: Prevents obvious blunders through deterministic checks
- **Mixed Strategy**: Uses probability-weighted selection for near-tied actions to prevent exploitable patterns
- **Local Self-play**: Resolves simultaneous move and engine-selected switch turns through poke-engine without Showdown connections

## Installation

Stratagem is designed to work as a drop-in enhancement to the existing Foul Play system.

### Prerequisites

- Python 3.8+
- Rust toolchain (for building poke-engine)
- PyTorch or similar ML framework (will be specified in requirements)

### Setup

1. Clone the Foul Play repository (if you haven't already)
2. Ensure poke-engine is built: `make poke_engine GEN=gen9`
3. Install dependencies: `pip install -r requirements.txt`
4. Stratagem will be available as a Python module

## Usage

### Local Training

```bash
python -m fp.stratagem.cli train \
    --games 10000 \
    --parallel-games 4 \
    --worlds 32 \
    --search-time-ms 150 \
    --prediction-horizon 3 \
    --team-path fp/teams/ \
    --weights fp/stratagem/weights/latest.pt \
    --seed 42
```

### Commands

- `train`: Train through bounded local self-play, with optional replays and checkpoints
- `evaluate`: Run a saved checkpoint against pure-MCTS local opponents
- `replay`: Validate and print a persisted local replay
- `inspect-worlds`: Sample worlds from the latest public replay turn
- `predict`: Produce current-action and sequence predictions from the latest public replay turn
- `play`: Run an authenticated Foul Play/Showdown session through Stratagem
        (`--websocket-uri`, `--ps-username`, optional `--ps-password`, `--opponent`,
        `--team`, and optional checkpoint/search options)

## Architecture

Implemented decision paths follow this information flow:

```
Foul Play Battle State
        |
        v
Stratagem Observation Layer (public info only)
        |
        +----------------------+
        |                      |
        v                      v
Hidden Team Belief      Current Battle State
Generation/Filtering         |
        |                      |
        +----------+-----------+
                   |
                   v
           Construct Engine State
                   |
                   v
              poke-engine
                   |
                   v
          MCTS / State Evaluation
                   |
                   v
         Stratagem Aggregation
                   |
          +--------+---------+
          |                  |
          v                  v
 Optional Learned Priors   Strategic Guards
          |                  |
          +--------+---------+
                   |
                   v
              Final Action Selection
```

### Key Architectural Constraints

1. **No Battle Recreation**: Stratagem does not recreate the Pokémon battle engine. It uses poke-engine for all simulation, damage calculation, mechanics, etc.
2. **Information Stratification**: The acting agent only receives information available at decision time; hidden information is never leaked.
3. **Simultaneous Decision Making**: Opponents never see our action before choosing theirs.
4. **Mechanic Persistence**: Existing serialized Mega and Tera state is preserved in future search states. Encoding a new future Tera/Mega/Z action remains unimplemented.
5. **Learned Augmentation**: A supplied trained model enhances but does not replace poke-engine's tactical search.

## Configuration

All configuration is centralized in `fp/stratagem/config.py`. Key parameters include:

- `world_count`: Number of hidden worlds to sample per decision
- `search_time_ms`: MCTS search time per world
- `prediction_horizon`: How many future turns to predict
- `max_turns`: Maximum turns per training game
- `parallel_games`: Number of simultaneous training games
- `learning_rate`, `hidden_size`: Neural network parameters
- `checkpoint_frequency`: How often to save checkpoints

See `fp/stratagem/config.py` for the full list and detailed descriptions.

## Replay Format

The planned replay system will save JSON training battles containing:

- Battle metadata (format, generation, seed)
- Team identities and true hidden teams (for simulator only)
- Per-turn public observations
- Selected and actual opponent actions
- Engine outcomes and state transitions
- MCTS summaries and world weights
- Prediction sequences and accuracy
- Learned model outputs
- Final outcome and turn count
- Detected loss signatures

## Development

See `IMPLEMENTATION_PLAN.md` for detailed implementation milestones and progress tracking.

## Testing

Run the existing test suite to ensure no regressions:
```bash
.venv\Scripts\python.exe -m unittest discover -s tests
.venv\Scripts\python.exe -m unittest discover -s tests/stratagem
```

All Stratagem tests live in `tests/stratagem/`. The existing legacy pytest tests can also be run with:
```bash
.venv\Scripts\python.exe -m pytest tests/stratagem -q
```