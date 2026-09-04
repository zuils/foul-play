"""
Tests for the Stratagem opponent model.
"""

import pytest
from unittest.mock import MagicMock
from fp.config import FoulPlayConfig
from fp.battle.state import Battle
from fp.modes.standard_battle import StandardBattleMode
from fp.stratagem.inference import TeamSampler, Belief, OpponentModel
from fp.stratagem.core import Observation


@pytest.fixture(autouse=True)
def default_pokemon_format():
    FoulPlayConfig.pokemon_format = "gen9ou"
    yield
    FoulPlayConfig.pokemon_format = ""


@pytest.fixture
def mock_smogon_sets():
    """Create a mock SmogonSets instance for testing."""
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
            """Return some simple sets for common test Pokemon."""
            # Always return Charizard sets for this test to isolate issues
            return [
                (
                    self._create_pokemon_set_mock("Charizard", "Life Orb", "Blaze", "Timid", (252, 0, 0, 0, 4, 252),
                                                  [("Flamethrower",), ("Air Slash",)]),
                    self._create_moveset_mock([("Flamethrower",), ("Air Slash",)])
                ),
                (
                    self._create_pokemon_set_mock("Charizard", "Charizardite Y", "Drought", "Modest", (252, 0, 0, 252, 4, 0),
                                                  [("Fire Blast",), ("Focus Blast",)]),
                    self._create_moveset_mock([("Fire Blast",), ("Focus Blast",)])
                )
            ]

        def _create_pokemon_set_mock(self, name, item, ability, nature, evs, move_tuples):
            """Create a properly mocked PokemonSet object."""
            pkmn_set_mock = MagicMock()
            # Create the inner Pokemon mock
            pokemon_mock = MagicMock()
            pokemon_mock.name = name
            # Set attributes directly so they return the actual values, not MagicMocks
            pkmn_set_mock.pkmn = pokemon_mock
            pkmn_set_mock.item = item
            pkmn_set_mock.ability = ability
            pkmn_set_mock.nature = nature
            pkmn_set_mock.evs = evs
            pkmn_set_mock.tera_type = None
            return pkmn_set_mock

        def _create_moveset_mock(self, move_tuples):
            """Create a properly mocked PokemonMoveset object."""
            moveset_mock = MagicMock()
            move_mocks = []
            for move_tuple in move_tuples:
                move_name = move_tuple[0]
                move_mock = MagicMock()
                # Set attribute directly
                move_mock.name = move_name
                move_mocks.append(move_mock)
            moveset_mock.moveset = move_mocks
            return moveset_mock

    return MockSmogonSets()


@pytest.fixture
def battle():
    """Create a basic battle state for testing."""
    battle = Battle(None)
    battle.generation = "gen9"
    battle.mode = StandardBattleMode()
    return battle


def test_opponent_model_creation(mock_smogon_sets, battle):
    """Test that opponent model can be created."""
    team_sampler = TeamSampler(mock_smogon_sets)
    belief = Belief(team_sampler)
    opponent_model = OpponentModel(belief)
    assert opponent_model is not None
    assert opponent_model.belief == belief


def test_opponent_model_predict_lead(mock_smogon_sets, battle):
    """Test predicting lead Pokemon."""
    team_sampler = TeamSampler(mock_smogon_sets, random_seed=42)
    belief = Belief(team_sampler, world_count=10, random_seed=42)

    obs = Observation(battle, "user")
    # Set up an observation with some revealed info
    obs.opponent_active = {
        'name': 'Charizard',
        'revealed': True,
        'ability': 'blaze',
        'item': 'lifeorb',
        'moves': ['Flamethrower', 'Air Slash'],
        'nature': 'Timid',
        'evs': [252, 0, 0, 0, 4, 252],
        'status': 'none'
    }
    obs.opponent_reserve_revealed = []

    belief.initialize_from_observation(obs)
    opponent_model = OpponentModel(belief, random_seed=42)

    lead_pred = opponent_model.predict_lead(obs)
    assert 'species' in lead_pred
    assert 'confidence' in lead_pred
    assert 'alternatives' in lead_pred
    assert isinstance(lead_pred['species'], str)
    assert 0.0 <= lead_pred['confidence'] <= 1.0
    assert isinstance(lead_pred['alternatives'], list)


def test_opponent_model_predict_move(mock_smogon_sets, battle):
    """Test predicting opponent move."""
    team_sampler = TeamSampler(mock_smogon_sets, random_seed=42)
    belief = Belief(team_sampler, world_count=10, random_seed=42)

    obs = Observation(battle, "user")
    # Set up an observation with some revealed info
    obs.opponent_active = {
        'name': 'Charizard',
        'revealed': True,
        'ability': 'blaze',
        'item': 'lifeorb',
        'moves': ['Flamethrower', 'Air Slash'],
        'nature': 'Timid',
        'evs': [252, 0, 0, 0, 4, 252],
        'status': 'none'
    }
    obs.opponent_reserve_revealed = []

    belief.initialize_from_observation(obs)
    opponent_model = OpponentModel(belief, random_seed=42)

    move_pred = opponent_model.predict_move(obs)
    assert 'move' in move_pred
    assert 'confidence' in move_pred
    assert 'alternatives' in move_pred
    assert isinstance(move_pred['move'], str)
    assert 0.0 <= move_pred['confidence'] <= 1.0
    assert isinstance(move_pred['alternatives'], list)


def test_opponent_model_predict_sequence(mock_smogon_sets, battle):
    """Test predicting sequence of actions."""
    team_sampler = TeamSampler(mock_smogon_sets, random_seed=42)
    belief = Belief(team_sampler, world_count=10, random_seed=42)

    obs = Observation(battle, "user")
    # Set up an observation with some revealed info
    obs.opponent_active = {
        'name': 'Charizard',
        'revealed': True,
        'ability': 'blaze',
        'item': 'lifeorb',
        'moves': ['Flamethrower', 'Air Slash'],
        'nature': 'Timid',
        'evs': [252, 0, 0, 0, 4, 252],
        'status': 'none'
    }
    obs.opponent_reserve_revealed = []

    belief.initialize_from_observation(obs)
    opponent_model = OpponentModel(belief, random_seed=42)

    with pytest.raises(ValueError, match="full known team"):
        opponent_model.predict_sequence(obs, horizon=3)


def test_opponent_model_reproducibility(mock_smogon_sets, battle):
    """Test that predictions are reproducible with fixed seed."""
    team_sampler1 = TeamSampler(mock_smogon_sets, random_seed=42)
    belief1 = Belief(team_sampler1, world_count=10, random_seed=42)

    team_sampler2 = TeamSampler(mock_smogon_sets, random_seed=42)
    belief2 = Belief(team_sampler2, world_count=10, random_seed=42)

    obs = Observation(battle, "user")
    obs.opponent_active = {
        'name': 'Charizard',
        'revealed': True,
        'ability': 'blaze',
        'item': 'lifeorb',
        'moves': ['Flamethrower', 'Air Slash'],
        'nature': 'Timid',
        'evs': [252, 0, 0, 0, 4, 252],
        'status': 'none'
    }
    obs.opponent_reserve_revealed = []

    belief1.initialize_from_observation(obs)
    belief2.initialize_from_observation(obs)

    opponent_model1 = OpponentModel(belief1, random_seed=42)
    opponent_model2 = OpponentModel(belief2, random_seed=42)

    lead_pred1 = opponent_model1.predict_lead(obs)
    lead_pred2 = opponent_model2.predict_lead(obs)

    assert lead_pred1['species'] == lead_pred2['species']
    assert lead_pred1['confidence'] == lead_pred2['confidence']

    move_pred1 = opponent_model1.predict_move(obs)
    move_pred2 = opponent_model2.predict_move(obs)

    assert move_pred1['move'] == move_pred2['move']
    assert move_pred1['confidence'] == move_pred2['confidence']


if __name__ == "__main__":
    pytest.main([__file__])