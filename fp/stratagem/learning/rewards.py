"""Bounded learning rewards derived from observable decisions and game outcomes."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RewardSignals:
    """Signals available after a local transition or completed game."""

    terminal_outcome: float | None
    strategic_progress: float
    engine_action_value: float | None = None
    opponent_action_probability: float | None = None
    sequence_action_probability: float | None = None


@dataclass(frozen=True)
class RewardBreakdown:
    """Normalized reward and its weighted, auditable components."""

    total: float
    terminal: float | None
    strategic: float
    engine: float | None
    opponent_prediction: float | None
    sequence_prediction: float | None


class RewardBuilder:
    """Build a bounded reward without fabricating unavailable learning signals."""

    TERMINAL_WEIGHT = 0.50
    STRATEGIC_WEIGHT = 0.25
    ENGINE_WEIGHT = 0.15
    OPPONENT_PREDICTION_WEIGHT = 0.05
    SEQUENCE_PREDICTION_WEIGHT = 0.05

    @classmethod
    def build(cls, signals: RewardSignals) -> RewardBreakdown:
        """Combine supplied signals into a $[-1, 1]$ target.

        Weights are renormalized over available signals, so missing prediction or
        engine information neither becomes a neutral fabricated value nor changes
        the scale of the resulting reward.
        """
        components: list[tuple[float, float]] = [
            (cls.STRATEGIC_WEIGHT, cls._bounded_signed(signals.strategic_progress)),
        ]
        terminal = None
        engine = None
        opponent_prediction = None
        sequence_prediction = None

        if signals.terminal_outcome is not None:
            terminal = cls._bounded_signed(signals.terminal_outcome)
            components.append((cls.TERMINAL_WEIGHT, terminal))
        if signals.engine_action_value is not None:
            engine = cls._normalize_engine_value(signals.engine_action_value)
            components.append((cls.ENGINE_WEIGHT, engine))
        if signals.opponent_action_probability is not None:
            opponent_prediction = cls._normalize_probability(
                signals.opponent_action_probability
            )
            components.append((cls.OPPONENT_PREDICTION_WEIGHT, opponent_prediction))
        if signals.sequence_action_probability is not None:
            sequence_prediction = cls._normalize_probability(
                signals.sequence_action_probability
            )
            components.append((cls.SEQUENCE_PREDICTION_WEIGHT, sequence_prediction))

        weight_total = sum(weight for weight, _ in components)
        if weight_total <= 0:
            raise ValueError("Reward construction requires at least one signal")
        total = sum(weight * value for weight, value in components) / weight_total
        return RewardBreakdown(
            total=cls._bounded_signed(total),
            terminal=terminal,
            strategic=cls._bounded_signed(signals.strategic_progress),
            engine=engine,
            opponent_prediction=opponent_prediction,
            sequence_prediction=sequence_prediction,
        )

    @staticmethod
    def _bounded_signed(value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Reward signals must be finite")
        return max(-1.0, min(1.0, float(value)))

    @classmethod
    def _normalize_engine_value(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Engine values must be finite")
        normalized = (2.0 * value) - 1.0 if 0.0 <= value <= 1.0 else value
        return cls._bounded_signed(normalized)

    @classmethod
    def _normalize_probability(cls, value: float) -> float:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("Prediction probabilities must be in [0, 1]")
        return cls._bounded_signed((2.0 * value) - 1.0)