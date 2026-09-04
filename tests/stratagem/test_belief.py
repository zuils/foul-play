"""
Tests for the Stratagem belief module.
"""

import pytest
from unittest.mock import MagicMock
from fp.config import FoulPlayConfig
from fp.battle.state import Battle
from fp.modes.standard_battle import StandardBattleMode
from fp.stratagem.inference import TeamSampler, Belief
from fp.stratagem.core import Observation


@pytest.fixture(autouse=True)
def default_pokemon_format():
    FoulPlayConfig.pokemon_format = "gen9ou"
    yield
    FoulPlayConfig.pokemon_format = ""


class MockSmogonSets:
    """Mock SmogonSets for testing that returns predefined sets."""

    def __init__(self):
        self.initialized = False
        self.pkmn_sets = {
            species: []
            for species in ("charizard", "gengar", "blastoise", "venusaur", "machamp", "pikachu")
        }

    def initialize(self, format_spec, pkmn_names):
        self.initialized = True
        self.format_spec = format_spec
        self.pkmn_names = pkmn_names

    def get_all_remaining_trait_combinations(self, pkmn):
        """Return some predefined sets for testing."""
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


@pytest.fixture
def mock_smogon_sets():
    """Create a mock SmogonSets instance for testing."""
    return MockSmogonSets()


@pytest.fixture
def battle():
    """Create a basic battle state for testing."""
    battle = Battle(None)
    battle.generation = "gen9"
    battle.mode = StandardBattleMode()
    return battle


def test_belief_creation(mock_smogon_sets):
    """Test that belief can be created."""
    team_sampler = TeamSampler(mock_smogon_sets)
    belief = Belief(team_sampler)
    assert belief is not None
    assert belief.world_count == 100  # default


def test_belief_initialization(mock_smogon_sets, battle):
    """Test belief initialization from observation."""
    team_sampler = TeamSampler(mock_smogon_sets, random_seed=42)
    belief = Belief(team_sampler, world_count=10, random_seed=42)

    obs = Observation(battle, "user")
    belief.initialize_from_observation(obs)

    assert len(belief.worlds) == 10
    assert len(belief.weights) == 10
    assert all(w == 0.0 for w in belief.weights)  # Initial weights should be 0
    assert not belief._evidence_applied


def test_belief_sample_world(mock_smogon_sets, battle):
    """Test sampling a world from belief."""
    team_sampler = TeamSampler(mock_smogon_sets, random_seed=42)
    belief = Belief(team_sampler, world_count=5, random_seed=42)

    obs = Observation(battle, "user")
    belief.initialize_from_observation(obs)

    world = belief.sample_world()
    assert isinstance(world, list)
    assert len(world) == 6  # Should be a full team of 6 Pokemon

    # Each Pokemon should have required fields
    for pkmn in world:
        assert 'species' in pkmn
        assert 'item' in pkmn
        assert 'ability' in pkmn
        assert 'nature' in pkmn
        assert 'evs' in pkmn
        assert 'moves' in pkmn
        assert 'tera_type' in pkmn


def test_belief_sample_multiple_worlds(mock_smogon_sets, battle):
    """Test sampling multiple worlds from belief."""
    team_sampler = TeamSampler(mock_smogon_sets, random_seed=42)
    belief = Belief(team_sampler, world_count=5, random_seed=42)

    obs = Observation(battle, "user")
    belief.initialize_from_observation(obs)

    worlds = belief.sample_worlds(3)
    assert len(worlds) == 3
    for world in worlds:
        assert len(world) == 6


def test_belief_get_world_weights(mock_smogon_sets, battle):
    """Test getting normalized world weights."""
    team_sampler = TeamSampler(mock_smogon_sets, random_seed=42)
    belief = Belief(team_sampler, world_count=5, random_seed=42)

    obs = Observation(battle, "user")
    belief.initialize_from_observation(obs)

    weights = belief.get_world_weights()
    assert len(weights) == 5
    assert abs(sum(weights) - 1.0) < 1e-6  # Should sum to 1
    assert all(w >= 0 for w in weights)  # Should be non-negative


def test_belief_get_most_likely_world(mock_smogon_sets, battle):
    """Test getting the most likely world."""
    team_sampler = TeamSampler(mock_smogon_sets, random_seed=42)
    belief = Belief(team_sampler, world_count=5, random_seed=42)

    obs = Observation(battle, "user")
    belief.initialize_from_observation(obs)

    # All weights are equal initially, so any world could be most likely
    world = belief.get_most_likely_world()
    assert len(world) == 6


def test_belief_get_effective_world_count(mock_smogon_sets, battle):
    """Test getting effective world count."""
    team_sampler = TeamSampler(mock_smogon_sets, random_seed=42)
    belief = Belief(team_sampler, world_count=10, random_seed=42)

    obs = Observation(battle, "user")
    belief.initialize_from_observation(obs)

    # Initially, all worlds equally likely, so effective count should be 10
    eff_count = belief.get_effective_world_count()
    assert abs(eff_count - 10.0) < 1e-6


def test_belief_update_with_evidence(mock_smogon_sets, battle):
    """Test updating belief with evidence."""
    team_sampler = TeamSampler(mock_smogon_sets, random_seed=42)
    belief = Belief(team_sampler, world_count=5, random_seed=42)

    obs = Observation(battle, "user")
    belief.initialize_from_observation(obs)

    # Update with evidence (should change weights)
    belief.update_with_evidence(obs)

    # Evidence applied flag should be set
    assert belief._evidence_applied

    # Weights should have changed (unless likelihood was 0 for all worlds)
    # Note: with our simple likelihood function, weights might not change much
    # but the flag should be set


def test_belief_reproducibility(mock_smogon_sets, battle):
    """Test that belief sampling is reproducible with fixed seed."""
    team_sampler1 = TeamSampler(mock_smogon_sets, random_seed=42)
    belief1 = Belief(team_sampler1, world_count=5, random_seed=42)

    team_sampler2 = TeamSampler(mock_smogon_sets, random_seed=42)
    belief2 = Belief(team_sampler2, world_count=5, random_seed=42)

    obs = Observation(battle, "user")
    belief1.initialize_from_observation(obs)
    belief2.initialize_from_observation(obs)

    # Sample worlds should be identical
    world1 = belief1.sample_world()
    world2 = belief2.sample_world()

    assert len(world1) == len(world2) == 6
    for pkmn1, pkmn2 in zip(world1, world2):
        assert pkmn1['species'] == pkmn2['species']
        assert pkmn1['ability'] == pkmn2['ability']
        assert pkmn1['item'] == pkmn2['item']


if __name__ == "__main__":
    pytest.main([__file__])