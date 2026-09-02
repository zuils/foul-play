"""
Stratagem battle mode for Foul Play.
Integrates Stratagem's decision-making process as a Foul Play battle mode.
"""

import logging
from typing import List, Tuple
from fp import constants
from fp.battle.state import Battle, LastUsedMove
from fp.config import FoulPlayConfig
from fp.constants import BattleType
from fp.modes.base import BattleMode, get_first_request_json
from fp.battle.protocol import process_battle_updates
from fp.format_spec import FormatSpec
from fp.stratagem.core import Observation
from fp.stratagem.inference.team_sampler import TeamSampler
from fp.stratagem.inference.belief import Belief
from fp.stratagem.inference.opponent_model import OpponentModel
from fp.stratagem.engine.aggregation import WorldAggregator
from fp.stratagem.core.actions import StrategicActionSelector
from fp.stratagem.config import CONFIG as STRATAGEM_CONFIG
from fp.stratagem.learning.model import StratagemModelWrapper
from fp.data.sets import SmogonSets

logger = logging.getLogger(__name__)


class StratagemMode(BattleMode):
    name = BattleType.STRATAGEM
    requires_team = True

    def __init__(self):
        self.smogon_sets = SmogonSets()
        self.team_sampler = None
        self.belief = None
        self.opponent_model = None
        self.action_selector = StrategicActionSelector(
            temperature=STRATAGEM_CONFIG.temperature
        )
        self.learned_model = None
        self._loaded_weights_path = None
        # Initialize with default values, will be updated in start_battle
        self.our_team_specs = []
        logger.info("StratagemMode initialized")

    async def start_battle(
        self,
        ps_websocket_client,
        pokemon_battle_type,
        team_dict
    ) -> Battle:
        battle, msg = await self.start_battle_common(
            ps_websocket_client, pokemon_battle_type
        )
        battle.user.team_dict = team_dict
        self._initialize_stratagem_components(battle, team_dict)

        if battle.gen.has_team_preview:
            while constants.START_TEAM_PREVIEW not in msg:
                msg = await ps_websocket_client.receive_message()
            opponent_pokemon = [
                line.split("|")[3]
                for line in msg.split(constants.START_TEAM_PREVIEW)[-1].split("\n")
                if line and line.split("|")[1] == constants.TEAM_PREVIEW_POKE
                and line.split("|")[2].strip() == battle.opponent.name
            ]
            await get_first_request_json(ps_websocket_client, battle)
            battle.initialize_team_preview(opponent_pokemon, pokemon_battle_type)
            battle.during_team_preview()
            self._initialize_smogon_sets(battle, pokemon_battle_type)
            await self.handle_team_preview(battle, ps_websocket_client)
        else:
            while constants.START_STRING not in msg:
                msg = await ps_websocket_client.receive_message()
            battle.started = True
            battle.msg_list = [
                message
                for message in msg.split(constants.START_STRING)[1].strip().split("\n")
                if not message.startswith("|switch|{}".format(battle.user.name))
            ]
            await get_first_request_json(ps_websocket_client, battle)
            self._initialize_smogon_sets(battle, pokemon_battle_type)
            process_battle_updates(battle)
            await ps_websocket_client.send_message(
                battle.battle_tag, await self._select_stratagem_move(battle)
            )

        return battle

    def _initialize_smogon_sets(self, battle: Battle, pokemon_battle_type: str) -> None:
        names = {pokemon.name for pokemon in battle.user.reserve + battle.opponent.reserve}
        if battle.user.active is not None:
            names.add(battle.user.active.name)
        if battle.opponent.active is not None:
            names.add(battle.opponent.active.name)
        self.smogon_sets.initialize(
            FormatSpec.from_format_string(FoulPlayConfig.smogon_stats or pokemon_battle_type), names
        )

    def _initialize_stratagem_components(self, battle: Battle, team_dict: dict):
        """Initialize Stratagem components for the battle."""
        if self._loaded_weights_path != STRATAGEM_CONFIG.weights_path:
            self.learned_model = (
                StratagemModelWrapper(model_path=STRATAGEM_CONFIG.weights_path)
                if STRATAGEM_CONFIG.weights_path is not None
                else None
            )
            self._loaded_weights_path = STRATAGEM_CONFIG.weights_path

        # Convert team_dict to Stratagem team specs format
        self.our_team_specs = self._convert_team_to_specs(team_dict)

        # Initialize team sampler
        self.team_sampler = TeamSampler(self.smogon_sets)

        # Initialize belief state
        self.belief = Belief(self.team_sampler, world_count=STRATAGEM_CONFIG.world_count)

        # Initialize opponent model
        self.opponent_model = OpponentModel(
            belief=self.belief,
            our_team=self.our_team_specs,
            search_time_ms=STRATAGEM_CONFIG.search_time_ms
        )

        logger.info("Stratagem components initialized")

    def _convert_team_to_specs(self, team_dict: dict) -> List[dict]:
        """Convert Foul Play team dict to Stratagem team specs format."""
        specs = []
        members = team_dict.get("team", []) if isinstance(team_dict, dict) else team_dict
        for pokemon in members:
            spec = {
                'species': pokemon.get('species', ''),
                'item': pokemon.get('item', None),
                'ability': pokemon.get('ability', None),
                'nature': pokemon.get('nature', None),
                'evs': [int(value or 0) for value in pokemon.get('evs', {}).values()] if isinstance(pokemon.get('evs'), dict) else [int(value or 0) for value in pokemon.get('evs', [0, 0, 0, 0, 0, 0])],
                'moves': [move.get('move', '') if isinstance(move, dict) else move for move in pokemon.get('moves', [])],
                'tera_type': pokemon.get('tera_type', None),
                'level': int(pokemon.get('level') or 100)
            }
            specs.append(spec)
        return specs

    def search_params(self, battle: Battle) -> Tuple[int, int]:
        """Return (parallelism, search_time_ms) for Stratagem search."""
        # Stratagem uses its own internal parallelism for worlds
        # We return 1 for external parallelism since Stratagem handles internal parallelization
        return 1, STRATAGEM_CONFIG.search_time_ms

    def prepare_battles(self, battle: Battle, num_battles: int) -> List[Tuple[Battle, float]]:
        """Prepare multiple battles for parallel search."""
        # Stratagem doesn't benefit from external parallel battles since it does internal world sampling
        # Return the same battle with weight 1.0
        return [(battle, 1.0)]

    async def handle_team_preview(self, battle, ps_websocket_client):
        """Select one team-preview lead from Stratagem's switch candidates."""
        observation = Observation(battle, player="user")
        self.belief.initialize_from_observation(observation)
        action_scores = self._get_stratagem_action_values(battle, observation)
        selected_action = self.action_selector.select_action(
            observation, self.action_selector.legal_actions(observation, action_scores)
        )
        if not selected_action.startswith(f"{constants.SWITCH_STRING} "):
            raise ValueError("Team preview requires a scored switch action")
        selected_name = selected_action.removeprefix(f"{constants.SWITCH_STRING} ")
        for index, pokemon in enumerate(battle.user.reserve, start=1):
            if pokemon.name == selected_name:
                break
        else:
            raise ValueError(f"Team-preview lead is not on the user team: {selected_name}")
        battle.user.last_selected_move = LastUsedMove(
            "teampreview", selected_action, battle.turn
        )
        team_indexes = list(range(1, len(battle.user.reserve) + 1))
        team_indexes.remove(index)
        await ps_websocket_client.send_message(
            battle.battle_tag,
            [f"/team {index}{''.join(str(value) for value in team_indexes)}|{battle.rqid}"],
        )

    async def select_move(self, battle: Battle) -> List[str]:
        """Select a move using Stratagem's decision-making process.

        Overrides the base class method to provide Stratagem-specific move selection.
        """
        return await self._select_stratagem_move(battle)

    async def _select_stratagem_move(self, battle: Battle) -> List[str]:
        """Select a move using Stratagem's decision-making process."""
        # Create observation from current battle state
        observation = Observation(battle, player="user")

        # Update belief with current evidence
        self.belief.update_with_evidence(observation)

        # Get action values from Stratagem's aggregated search
        action_scores = self._get_stratagem_action_values(battle, observation)

        # Apply strategic guards and select action
        legal_actions = self.action_selector.legal_actions(observation, action_scores)
        selected_action = self.action_selector.select_action(observation, legal_actions)

        # Format the decision for Pokemon Showdown
        return self._format_decision(battle, selected_action)

    def _get_stratagem_action_values(self, battle: Battle, observation: Observation) -> dict:
        """Get aggregated action values from Stratagem's multi-world search."""
        if not self.belief.worlds:
            # Initialize belief if not done yet
            self.belief.initialize_from_observation(observation)

        # Aggregate results from multiple worlds using belief weighting
        aggregator = WorldAggregator(self.belief, learned_model=self.learned_model)
        action_scores = aggregator.aggregate_worlds(
            observation,
            self.our_team_specs,
            search_time_ms=STRATAGEM_CONFIG.search_time_ms,
            prediction_horizon=STRATAGEM_CONFIG.prediction_horizon
        )

        return action_scores

    def _format_decision(self, battle: Battle, decision: str) -> List[str]:
        """Format a decision for communication with Pokemon-Showdown."""
        # Reuse the base class formatting logic
        from fp.modes.base import format_decision
        return format_decision(battle, decision)

    # Implement required abstract methods from BattleMode
    def opponent_possible_mega_evolutions(
        self, battle: Battle, smogon_sets: SmogonSets
    ) -> list:
        """Delegate to base class implementation."""
        from fp.modes.base import BattleMode
        return BattleMode.opponent_possible_mega_evolutions(self, battle, smogon_sets)

    def sample_mega_evolution(
        self, battle: Battle, index: int, smogon_sets: SmogonSets
    ):
        """Delegate to base class implementation."""
        from fp.modes.base import BattleMode
        return BattleMode.sample_mega_evolution(self, battle, index, smogon_sets)

    def get_all_remaining_sets(self, pokemon) -> list:
        """Stratagem doesn't use this method directly."""
        raise ValueError("Stratagem mode uses its own team sampling")

    def dataset_possibilities(self, battle) -> tuple:
        """Stratagem doesn't use this method directly."""
        # Return empty lists since we handle team sampling internally
        return [], None, False

    def assume_spread_for_speed_check(self, battle, battle_copy):
        """Stratagem doesn't use this method directly."""
        pass

    def add_revealed_pokemon(self, battle, pkmn):
        """Stratagem handles revealed Pokémon through belief updates."""
        # No action needed - belief is updated with evidence
        pass

    def check_zoroark_from_move(
        self, battle, side, pkmn, move_name, split_msg, zoroark_from_reserves
    ) -> object:
        """Stratagem doesn't use this method directly."""
        return pkmn

    def check_zoroark_from_immune(self, battle, side, pkmn, zoroark_from_reserves):
        """Stratagem doesn't use this method directly."""
        pass

    async def get_first_request_json(
        self, ps_websocket_client, battle: Battle
    ):
        """Get first request JSON."""
        # Reuse base class implementation
        from fp.modes.base import get_first_request_json
        return await get_first_request_json(ps_websocket_client, battle)

    def _switch_active_with_zoroark_from_reserves(
        self, opponent_side: object, zoroark_from_reserves: object
    ):
        """Delegate to base class implementation."""
        from fp.modes.base import BattleMode
        return BattleMode._switch_active_with_zoroark_from_reserves(
            self, opponent_side, zoroark_from_reserves
        )