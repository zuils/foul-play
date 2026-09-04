"""
Configuration module for Stratagem system.
Centralizes all configurable parameters.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class StratagemConfig:
    """Configuration for Stratagem battle system."""

    # World sampling parameters
    world_count: int = 32
    prediction_horizon: int = 3

    # MCTS parameters
    search_time_ms: int = 150  # Approximate target per world
    search_threads: int = 1

    # Training parameters
    max_turns: int = 100
    parallel_games: int = 1
    parallel_worlds: bool = True

    # Learning parameters
    hidden_size: int = 128
    learning_rate: float = 0.001
    batch_size: int = 32
    buffer_size: int = 10000
    gamma: float = 0.99

    # Optional trained learned-model weights for live policy priors.
    weights_path: Optional[str] = None

    # Checkpointing
    checkpoint_frequency: int = 100  # Save every N games

    # Randomness
    seed: Optional[int] = None

    # Verbosity
    verbose: bool = False

    # Team loading
    team_path: Optional[str] = None  # If None, uses default teams

    # Format
    format: str = "gen9randombattle"  # Default format

    # Mixed strategy temperature (for action selection)
    temperature: float = 1.0

    # Strategic guard thresholds
    useless_action_threshold: float = 0.01  # Below this value, action considered useless
    once_per_game_mechanic_threshold: float = 0.1  # Minimum advantage to use once-per-game mechanic

    # Loss mining parameters
    loss_signature_min_support: float = 0.1  # Minimum support to consider a pattern
    loss_signature_min_lift: float = 1.5   # Minimum lift (vs win rate) to consider significant

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.world_count <= 0:
            raise ValueError("world_count must be positive")
        if self.prediction_horizon < 0:
            raise ValueError("prediction_horizon must be non-negative")
        if self.search_time_ms <= 0:
            raise ValueError("search_time_ms must be positive")
        if self.max_turns <= 0:
            raise ValueError("max_turns must be positive")
        if self.parallel_games <= 0:
            raise ValueError("parallel_games must be positive")
        if not (0 < self.learning_rate <= 1):
            raise ValueError("learning_rate must be in (0, 1]")
        if not (0 <= self.gamma <= 1):
            raise ValueError("gamma must be in [0, 1]")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        if self.checkpoint_frequency <= 0:
            raise ValueError("checkpoint_frequency must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not (0 <= self.useless_action_threshold <= 1):
            raise ValueError("useless_action_threshold must be in [0, 1]")
        if not (0 <= self.once_per_game_mechanic_threshold <= 1):
            raise ValueError("once_per_game_mechanic_threshold must be in [0, 1]")
        if self.loss_signature_min_support <= 0:
            raise ValueError("loss_signature_min_support must be positive")
        if self.loss_signature_min_lift <= 1:
            raise ValueError("loss_signature_min_lift must be > 1")


# Global configuration instance
CONFIG = StratagemConfig()


def get_config() -> StratagemConfig:
    """Get the global configuration instance."""
    return CONFIG


def update_config(**kwargs) -> None:
    """Update configuration parameters."""
    global CONFIG
    for key, value in kwargs.items():
        if hasattr(CONFIG, key):
            setattr(CONFIG, key, value)
        else:
            raise AttributeError(f"Unknown configuration parameter: {key}")

    # Re-validate after updates
    CONFIG.__post_init__()