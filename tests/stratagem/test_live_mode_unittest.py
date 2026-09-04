"""Unittest coverage for live Stratagem mode integration contracts."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock

from fp.battle.state import Battle, Pokemon
from fp.config import FoulPlayConfig
from fp.constants import BattleType
from fp.modes import battle_mode
from fp.modes.stratagem_mode import StratagemMode
from fp.stratagem.config import get_config, update_config
from fp.stratagem.learning.model import StratagemModelWrapper
from fp.stratagem.learning.trainer import StratagemTrainer


class LiveModeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.previous_config = vars(get_config()).copy()
        self.previous_format = FoulPlayConfig.pokemon_format
        FoulPlayConfig.pokemon_format = "gen9ou"

    def tearDown(self):
        FoulPlayConfig.pokemon_format = self.previous_format
        update_config(**self.previous_config)

    def test_runtime_flag_selects_the_registered_stratagem_mode(self):
        previous_value = FoulPlayConfig.use_stratagem
        try:
            FoulPlayConfig.use_stratagem = True
            self.assertIsInstance(battle_mode(BattleType.STRATAGEM), StratagemMode)
        finally:
            FoulPlayConfig.use_stratagem = previous_value

    def test_team_converter_accepts_foul_play_team_list_schema(self):
        mode = StratagemMode.__new__(StratagemMode)
        converted = mode._convert_team_to_specs(
            [{
                "species": "pikachu",
                "item": "lightball",
                "ability": "static",
                "nature": "timid",
                "evs": {"hp": "4", "atk": "0", "def": "0", "spa": "252", "spd": "0", "spe": "252"},
                "moves": ["thunderbolt"],
                "tera_type": "electric",
                "level": "50",
            }]
        )

        self.assertEqual(converted[0]["evs"], [4, 0, 0, 252, 0, 252])
        self.assertEqual(converted[0]["moves"], ["thunderbolt"])
        self.assertEqual(converted[0]["level"], 50)

    def test_live_mode_loads_configured_checkpoint_after_runtime_configuration(self):
        source = StratagemTrainer(StratagemModelWrapper(hidden_sizes=(8,)), batch_size=1)
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "checkpoint.pt"
            source.save_checkpoint(checkpoint)
            update_config(weights_path=str(checkpoint))
            mode = StratagemMode()
            mode._initialize_stratagem_components(None, [])

        self.assertIsNotNone(mode.learned_model)
        self.assertEqual(mode._loaded_weights_path, str(checkpoint))

    def test_team_preview_sends_one_stratagem_ranked_team_order(self):
        mode = StratagemMode()
        mode.belief = Mock()
        mode._get_stratagem_action_values = Mock(return_value={"switch pikachu": 1.0})
        battle = Battle("battle-test")
        battle.generation = "gen9"
        battle.rqid = 4
        battle.user.reserve = [Pokemon("pikachu", 50), Pokemon("gengar", 50)]
        battle.opponent.reserve = [Pokemon("charizard", 50)]
        websocket = AsyncMock()

        import asyncio
        asyncio.run(mode.handle_team_preview(battle, websocket))

        websocket.send_message.assert_awaited_once_with(
            "battle-test", ["/team 12|4"]
        )
        self.assertEqual(battle.user.last_selected_move.move, "switch pikachu")