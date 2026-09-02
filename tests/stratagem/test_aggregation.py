"""
Tests for the Stratagem aggregation module.
"""

import pytest
from unittest.mock import MagicMock
from fp.stratagem.engine.aggregation import WorldAggregator, aggregate_worlds_simple
from fp.stratagem.inference.belief import Belief
from fp.stratagem.inference.team_sampler import TeamSampler


class MockSmogonSets:
    """Mock SmogonSets for testing that returns predefined sets."""

    def __init__(self):
        self.initialized = False

    def initialize(self, format_spec, pkmn_names):
        self.initialized = True
        self.format_spec = format_spec
        self.pkmn_names = pkmn_names

    def get_all_remaining_trait_combinations(self, pkmn):
        """Return some simple sets for common test Pokemon."""
        # Return a few simple sets for common test Pokemon
        if pkmn.name.lower() == "charizard":
            return [
                MagicMock(
                    pkmn=MagicMock(name="Charizard"),
                    item="Life Orb",
                    ability="Blaze",
                    nature="Timid",
                    evs=(252, 0, 0, 0, 4, 252),
                    moveset=[MagicMock(name="Flamethrower"), MagicMock(name="Air Slash")]
                ),
                MagicMock(
                    pkmn=MagicMock(name="Charizard"),
                    item="Charizardite Y",
                    ability="Drought",
                    nature="Modest",
                    evs=(252, 0, 0, 252, 4, 0),
                    moveset=[MagicMock(name="Fire Blast"), MagicMock(name="Focus Blast")]
                )
            ]
        elif pkmn.name.lower() == "gengar":
            return [
                MagicMock(
                    pkmn=MagicMock(name="Gengar"),
                    item="Life Orb",
                    ability="Levitate",
                    nature="Timid",
                    evs=(252, 0, 0, 0, 4, 252),
                    moveset=[MagicMock(name="Shadow Ball"), MagicMock(name="Sludge Bomb")]
                )
            ]
        else:
            # Return a generic set for other Pokemon
            return [
                MagicMock(
                    pkmn=MagicMock(name=pkmn.name),
                    item="None",
                    ability="Sturdy",
                    nature="Serious",
                    evs=(0, 0, 0, 0, 0, 0),
                    moveset=[MagicMock(name="Tackle")]
                )
            ]


def test_aggregator_creation():
    """Test that WorldAggregator can be created."""
    mock_smogon_sets = MockSmogonSets()
    team_sampler = TeamSampler(mock_smogon_sets)
    belief = Belief(team_sampler)
    aggregator = WorldAggregator(belief)
    assert aggregator is not None
    assert aggregator.belief == belief


def test_aggregate_worlds_simple_function():
    """Test the simple aggregate_worlds function."""
    assert callable(aggregate_worlds_simple)


def test_aggregate_worlds_with_visits():
    """Test the aggregate_worlds_with_visits method."""
    mock_smogon_sets = MockSmogonSets()
    team_sampler = TeamSampler(mock_smogon_sets)
    belief = Belief(team_sampler, world_count=5, random_seed=42)
    aggregator = WorldAggregator(belief)

    # Test that the method exists
    assert hasattr(aggregator, 'aggregate_worlds_with_visits')
    assert callable(aggregator.aggregate_worlds_with_visits)


if __name__ == "__main__":
    pytest.main([__file__])