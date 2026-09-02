"""
Inference module for Stratagem system.
Contains team sampling, belief, and opponent modeling components.
"""

from .team_sampler import TeamSampler
from .belief import Belief
from .opponent_model import OpponentModel

__all__ = ["TeamSampler", "Belief", "OpponentModel"]