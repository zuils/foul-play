"""
Team sampler for Stratagem system.
Generates complete opponent team hypotheses using Smogon set machinery.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple
from fp.battle.state import Pokemon
from fp.data.sets.smogon import SmogonSets
from fp.stratagem.core import Observation
from fp import constants
from fp.battle.helpers import normalize_name


class TeamSampler:
    """
    Samples complete opponent team hypotheses consistent with observed battle state.
    Uses Smogon set machinery to generate candidate sets for each Pokemon slot.
    """

    def __init__(self, smogon_sets: SmogonSets, random_seed: Optional[int] = None):
        """
        Initialize team sampler.

        Args:
            smogon_sets: Initialized SmogonSets instance for the current format
            random_seed: Optional seed for reproducible sampling
        """
        self.smogon_sets = smogon_sets
        self.rng = random.Random(random_seed) if random_seed is not None else random.Random()

    def sample_team(self, observation: Observation) -> List[Dict]:
        """
        Sample a complete opponent team consistent with the observation.

        Args:
            observation: Current battle observation

        Returns:
            List of 6 dictionaries, each representing a Pokemon with:
                - species: str
                - item: str
                - ability: str
                - nature: str
                - evs: List[int]
                - moves: List[str]
                - tera_type: Optional[str]
        """
        # Get revealed opponent Pokemon from observation
        revealed_pokemon = self._get_revealed_opponent_pokemon(observation)
        unrevealed_count = 6 - len(revealed_pokemon)

        revealed_species = {normalize_name(pkmn["species"]) for pkmn in revealed_pokemon}
        available_species = [
            species
            for species in self.smogon_sets.pkmn_sets
            if species not in revealed_species
        ]
        if len(available_species) < unrevealed_count:
            raise ValueError(
                "Smogon data does not contain enough distinct species to construct a team"
            )

        # Sample teams for revealed slots (constrained by observation)
        revealed_sets = []
        for pkmn_info in revealed_pokemon:
            pkmn_sets = self._get_constrained_sets(pkmn_info)
            if not pkmn_sets:
                raise ValueError(
                    f"No Smogon set is consistent with observed {pkmn_info['species']} evidence"
                )
            selected_set = self.rng.choice(pkmn_sets)
            revealed_sets.append(selected_set)

        # Sample teams for unrevealed slots
        unrevealed_sets = []
        used_species = revealed_species.copy()
        for _ in range(unrevealed_count):
            # Choose a species that hasn't been used yet
            available_for_slot = [s for s in available_species if s not in used_species]
            if not available_for_slot:
                # If we run out, allow duplicates (shouldn't happen in standard formats)
                available_for_slot = available_species
            species = self.rng.choice(available_for_slot)
            used_species.add(species)

            # Get all possible sets for this species (no constraints)
            pkmn_sets = self._get_blank_sets(species)
            if not pkmn_sets:
                raise ValueError(f"No Smogon sets are available for {species}")
            selected_set = self.rng.choice(pkmn_sets)
            unrevealed_sets.append(selected_set)

        # Combine revealed and unrevealed sets
        all_sets = revealed_sets + unrevealed_sets

        # Shuffle to avoid revealing which slots were revealed vs unrevealed
        self.rng.shuffle(all_sets)

        return all_sets

    def _get_revealed_opponent_pokemon(self, observation: Observation) -> List[Dict]:
        """
        Extract revealed opponent Pokemon from observation.

        Returns:
            List of dictionaries with known information about each revealed Pokemon
        """
        revealed = []

        # Check opponent's active Pokemon
        opp_active = observation.opponent_active
        if isinstance(opp_active, dict) and opp_active.get('revealed', False):
            revealed.append({
                'species': opp_active['name'],
                'known_moves': opp_active.get('moves', []),
                'known_ability': opp_active.get('ability'),
                'known_item': opp_active.get('item'),
                'known_nature': opp_active.get('nature'),
                'known_evs': opp_active.get('evs'),
                'known_status': opp_active.get('status'),
                # Note: we don't know HP fraction as it changes, but we know max_hp from stats
            })

        # Check opponent's revealed reserve Pokemon
        for pkmn in observation.opponent_reserve_revealed:
            if isinstance(pkmn, dict) and pkmn.get('revealed', False):
                revealed.append({
                    'species': pkmn['name'],
                    'known_moves': pkmn.get('moves', []),
                    'known_ability': pkmn.get('ability'),
                    'known_item': pkmn.get('item'),
                    'known_nature': pkmn.get('nature'),
                    'known_evs': pkmn.get('evs'),
                    'known_status': pkmn.get('status'),
                })

        return revealed

    def _is_set_consistent_with_info(self, pkmn_set: Dict, pkmn_info: Dict) -> bool:
        """
        Check if a Pokemon set is consistent with known information.

        Args:
            pkmn_set: Pokemon set dictionary to check
            pkmn_info: Dictionary with known information about the Pokemon

        Returns:
            True if the set is consistent with the known information, False otherwise
        """
        # Check species
        if pkmn_set["species"] != normalize_name(pkmn_info["species"]):
            return False

        # Check known moves
        if pkmn_info.get('known_moves'):
            known_moves = {normalize_name(move) for move in pkmn_info["known_moves"]}
            set_moves = set(pkmn_set["moves"])
            # All known moves must be present in the set's moves
            if not known_moves.issubset(set_moves):
                return False

        # Check known ability
        if pkmn_info.get('known_ability') and pkmn_info['known_ability'] is not None:
            if pkmn_set["ability"] != normalize_name(pkmn_info["known_ability"]):
                return False

        # Check known item
        if pkmn_info.get('known_item') not in (None, constants.UNKNOWN_ITEM):
            if pkmn_set["item"] != normalize_name(pkmn_info["known_item"]):
                return False

        # Check known nature
        if pkmn_info.get('known_nature') and pkmn_info['known_nature'] is not None:
            if pkmn_set["nature"] != normalize_name(pkmn_info["known_nature"]):
                return False

        # Check known EVs
        if pkmn_info.get('known_evs') and pkmn_info['known_evs'] is not None:
            if pkmn_set["evs"] != pkmn_info["known_evs"]:
                return False

        return True

    def _get_constrained_sets(self, pkmn_info: Dict) -> List[Dict]:
        """
        Get possible sets for a Pokemon that are consistent with known information.

        Args:
            pkmn_info: Dictionary with known information about the Pokemon

        Returns:
            List of possible set dictionaries
        """
        species = pkmn_info['species']
        # Create a Pokemon object with known constraints
        pkmn = Pokemon(species, level=100)  # Assume level 100 for team building

        # Apply known constraints so Foul Play's trait filtering handles item,
        # ability, choice-lock, speed, and Tera evidence consistently.
        if pkmn_info.get('known_moves'):
            # Clear default moves and set known ones
            pkmn.moves.clear()
            for move_name in pkmn_info['known_moves']:
                # Add the move to the Pokemon's move list
                pkmn.add_move(move_name)

        if pkmn_info.get('known_ability'):
            # Ability should match exactly as stored in Smogon data (case-sensitive)
            pkmn.ability = normalize_name(pkmn_info['known_ability'])

        if pkmn_info.get('known_item'):
            # Item should match exactly as stored in Smogon data (case-sensitive)
            known_item = pkmn_info["known_item"]
            if known_item != constants.UNKNOWN_ITEM:
                pkmn.item = normalize_name(known_item)

        if pkmn_info.get('known_nature'):
            # Nature should match exactly as stored in Smogon data (case-sensitive)
            pkmn.nature = pkmn_info['known_nature']

        if pkmn_info.get('known_evs'):
            pkmn.evs = pkmn_info['known_evs']

        # Status changes during battle and is not a set-identifying trait.

        raw_sets = self.smogon_sets.get_all_remaining_trait_combinations(pkmn)
        converted = self._convert_raw_sets(species, raw_sets, pkmn_info["known_moves"])
        return [
            pkmn_set
            for pkmn_set in converted
            if self._is_set_consistent_with_info(pkmn_set, pkmn_info)
        ]

    def _get_blank_sets(self, species: str) -> List[Dict]:
        """
        Get all possible sets for a species with no constraints.

        Args:
            species: Pokemon species name

        Returns:
            List of possible set dictionaries
        """
        pkmn = Pokemon(species, level=100)
        raw_sets = self.smogon_sets.get_all_remaining_trait_combinations(pkmn)
        return self._convert_raw_sets(species, raw_sets, [])

    def _convert_raw_sets(
        self, species: str, raw_sets: List, known_moves: List[str]
    ) -> List[Dict]:
        """
        Convert raw PokemonSet and PokemonMoveset objects to dictionaries.

        Args:
            species: Species represented by the trait combinations.
            raw_sets: `PokemonSet` values from SmogonSets.
            known_moves: Publicly revealed moves that every candidate must contain.

        Returns:
            List of set dictionaries
        """
        converted = []
        known_moves = [normalize_name(move) for move in known_moves]
        move_rates = self._move_usage_rates(species)
        for pkmn_set in raw_sets:
            legacy_moves = None
            if isinstance(pkmn_set, tuple):
                pkmn_set, legacy_moves = pkmn_set
            elif hasattr(pkmn_set, "moveset"):
                legacy_moves = pkmn_set
            moves = self._sample_moves(known_moves, move_rates, legacy_moves)
            if moves is None:
                continue
            set_dict = {
                "species": normalize_name(species),
                "item": normalize_name(str(pkmn_set.item)) if pkmn_set.item is not None else constants.UNKNOWN_ITEM,
                "ability": normalize_name(str(pkmn_set.ability)) if pkmn_set.ability is not None else None,
                "nature": normalize_name(str(pkmn_set.nature)) if pkmn_set.nature is not None else None,
                "evs": list(pkmn_set.evs) if pkmn_set.evs else [0, 0, 0, 0, 0, 0],
                "moves": moves,
                "tera_type": normalize_name(str(pkmn_set.tera_type)) if pkmn_set.tera_type else None,
            }
            converted.append(set_dict)
        return converted

    def _move_usage_rates(self, species: str) -> List[Tuple[str, float]]:
        pkmn = Pokemon(species, level=100)
        if not hasattr(self.smogon_sets, "move_usage_rates"):
            return []
        return [
            (normalize_name(move), float(rate))
            for move, rate in self.smogon_sets.move_usage_rates(pkmn)
            if rate > 0
        ]

    def _sample_moves(
        self,
        known_moves: List[str],
        move_rates: List[Tuple[str, float]],
        legacy_moves,
    ) -> Optional[List[str]]:
        if legacy_moves is not None:
            move_rates = [
                (normalize_name(move.name), 1.0)
                for move in legacy_moves.moveset
            ]
        available_moves = {move for move, _ in move_rates}
        if not available_moves or not set(known_moves).issubset(available_moves):
            return None

        moves = list(dict.fromkeys(known_moves))
        remaining_rates = [
            (move, rate) for move, rate in move_rates if move not in moves
        ]
        while len(moves) < 4 and remaining_rates:
            choices, weights = zip(*remaining_rates)
            move = self.rng.choices(choices, weights=weights, k=1)[0]
            moves.append(move)
            remaining_rates = [entry for entry in remaining_rates if entry[0] != move]
        return moves

    def sample_multiple_teams(self, observation: Observation, world_count: int) -> List[List[Dict]]:
        """
        Sample multiple complete opponent teams.

        Args:
            observation: Current battle observation
            world_count: Number of teams to sample

        Returns:
            List of world_count teams, each team being a list of 6 Pokemon dictionaries
        """
        teams = []
        for _ in range(world_count):
            team = self.sample_team(observation)
            teams.append(team)
        return teams