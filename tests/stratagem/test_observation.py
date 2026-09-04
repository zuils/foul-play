"""
Tests for the Stratagem observation layer.
"""

import pytest
from fp.config import FoulPlayConfig
from fp.battle.state import Battle, Pokemon
from fp import constants
from fp.modes.standard_battle import StandardBattleMode
from fp.stratagem.core import Observation


@pytest.fixture(autouse=True)
def default_pokemon_format():
    FoulPlayConfig.pokemon_format = "gen9ou"
    yield
    FoulPlayConfig.pokemon_format = ""


def test_observation_creation():
    """Test that observation can be created from a battle."""
    battle = Battle(None)
    battle.generation = "gen9"
    battle.mode = StandardBattleMode()
    obs = Observation(battle, "user")
    assert obs is not None
    assert obs.player == "user"


def test_observation_extracts_basic_info():
    """Test that observation extracts basic battle information."""
    battle = Battle(None)
    battle.generation = "gen9"
    battle.mode = StandardBattleMode()

    # Add some basic state
    battle.weather = "rain"
    battle.weather_turns_remaining = 5
    battle.turn = 10

    obs = Observation(battle, "user")

    assert obs.weather == "rain"
    assert obs.weather_turns_remaining == 5
    assert obs.turn == 10


def test_observation_hides_hidden_information():
    """Test that observation does not leak hidden opponent information."""
    battle = Battle(None)
    battle.generation = "gen9"
    battle.mode = StandardBattleMode()

    # Create a hidden Pokemon for opponent
    hidden_pokemon = Pokemon("Charizard", level=50)
    hidden_pokemon.revealed = False  # Not revealed yet
    hidden_pokemon.ability = "Blaze"  # This should be hidden
    hidden_pokemon.item = "Charizardite Y"  # This should be hidden
    hidden_pokemon.evs = [252, 0, 0, 0, 4, 252]  # This should be hidden

    battle.opponent.active = hidden_pokemon

    obs = Observation(battle, "user")

    # The opponent's active Pokemon should not reveal hidden info
    opponent_active = obs.opponent_active
    assert opponent_active["name"] == "charizard"
    assert not opponent_active["revealed"]
    assert opponent_active["ability"] is None  # Should be hidden
    assert opponent_active["item"] == constants.UNKNOWN_ITEM  # Should be hidden
    assert opponent_active["evs"] is None  # Should be hidden
    assert opponent_active["stats"] is None  # Should be hidden


def test_observation_reveals_visible_information():
    """Test that observation correctly shows revealed information."""
    battle = Battle(None)
    battle.generation = "gen9"
    battle.mode = StandardBattleMode()

    # Create a revealed Pokemon for user
    revealed_pokemon = Pokemon("Gengar", level=50)
    revealed_pokemon.revealed = True  # Revealed
    revealed_pokemon.ability = "Levitate"
    revealed_pokemon.item = "Life Orb"
    revealed_pokemon.evs = [0, 252, 0, 0, 252, 4]
    revealed_pokemon.status = "paralyze"

    battle.user.active = revealed_pokemon

    obs = Observation(battle, "user")

    # The user's active Pokemon should show revealed info
    active_pokemon = obs.active_pokemon
    assert active_pokemon["name"] == "gengar"
    assert active_pokemon["revealed"]
    assert active_pokemon["ability"] == "levitate"
    assert active_pokemon["item"] == "lifeorb"
    assert active_pokemon["evs"] == [0, 252, 0, 0, 252, 4]
    assert active_pokemon["status"] == "paralyze"
    # Check that stats are present and HP is correct
    assert active_pokemon["stats"] is not None
    assert active_pokemon["hp_fraction"] is not None
    assert 0 <= active_pokemon["hp_fraction"] <= 1.0


def test_observation_tracks_hidden_pokemon_count():
    """Test that observation correctly counts hidden Pokemon."""
    battle = Battle(None)
    battle.generation = "gen9"
    battle.mode = StandardBattleMode()

    # User has 2 revealed and 1 hidden reserve Pokemon
    revealed_pokemon1 = Pokemon("Gengar", level=50)
    revealed_pokemon1.revealed = True
    revealed_pokemon2 = Pokemon("Machamp", level=50)
    revealed_pokemon2.revealed = True
    hidden_pokemon = Pokemon("Charizard", level=50)
    hidden_pokemon.revealed = False  # Hidden

    battle.user.reserve = [revealed_pokemon1, revealed_pokemon2, hidden_pokemon]

    # Opponent has 1 revealed and 2 hidden reserve Pokemon
    opp_revealed = Pokemon("Blastoise", level=50)
    opp_revealed.revealed = True
    opp_hidden1 = Pokemon("Venusaur", level=50)
    opp_hidden1.revealed = False
    opp_hidden2 = Pokemon("Charizard", level=50)
    opp_hidden2.revealed = False

    battle.opponent.reserve = [opp_revealed, opp_hidden1, opp_hidden2]

    obs = Observation(battle, "user")

    assert obs.get_hidden_reserve_count() == 1
    assert obs.get_opponent_hidden_reserve_count() == 2
    assert len(obs.get_revealed_reserve_pokemon()) == 2
    assert len(obs.get_opponent_revealed_reserve_pokemon()) == 1


def test_observation_available_actions():
    """Test that observation correctly reports available actions."""
    battle = Battle(None)
    battle.generation = "gen9"
    battle.mode = StandardBattleMode()

    # Set up user with some moves
    active_pokemon = Pokemon("Gengar", level=50)
    active_pokemon.revealed = True

    # Add some moves - we'll reuse existing moves from the Pokemon
    # Just make sure we have some moves available
    # For simplicity, we'll check that the observation logic works
    # by verifying the available actions reporting

    battle.user.active = active_pokemon

    # Add a reserve Pokemon so switching is possible
    reserve_pokemon = Pokemon("Machamp", level=50)
    battle.user.reserve = [reserve_pokemon]

    obs = Observation(battle, "user")

    # Check available actions - we know Gengar has some moves by default
    # and Machamp is available for switching
    # Just test the logic rather than specific moves since move initialization is complex
    switch_available = obs.is_action_available("switch")
    assert switch_available  # Should be able to switch to Machamp

    # Get all available actions
    available = obs.get_available_actions()
    assert "switch" in available  # Switch should be in available actions


def test_observation_team_perspective():
    """Test that observation works correctly from both player perspectives."""
    battle = Battle(None)
    battle.generation = "gen9"
    battle.mode = StandardBattleMode()

    # Set up different active Pokemon for each side
    user_pokemon = Pokemon("Gengar", level=50)
    user_pokemon.revealed = True
    user_pokemon.ability = "Levitate"

    opponent_pokemon = Pokemon("Charizard", level=50)
    opponent_pokemon.revealed = True
    opponent_pokemon.ability = "Blaze"

    battle.user.active = user_pokemon
    battle.opponent.active = opponent_pokemon

    # Observe from user perspective
    user_obs = Observation(battle, "user")
    assert user_obs.active_pokemon["name"] == "gengar"
    assert user_obs.opponent_active["name"] == "charizard"

    # Observe from opponent perspective
    opponent_obs = Observation(battle, "opponent")
    assert opponent_obs.active_pokemon["name"] == "charizard"
    assert opponent_obs.opponent_active["name"] == "gengar"

    # Check that abilities are only visible when revealed
    assert user_obs.active_pokemon["ability"] == "levitate"
    assert user_obs.opponent_active["ability"] == "blaze"

    assert opponent_obs.active_pokemon["ability"] == "blaze"
    assert opponent_obs.opponent_active["ability"] == "levitate"


if __name__ == "__main__":
    pytest.main([__file__])