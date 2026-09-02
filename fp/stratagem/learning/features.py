"""Public-observation feature extraction for Stratagem's learned models."""

from __future__ import annotations

import hashlib
from typing import Dict, List, Sequence

import numpy as np

from fp import constants
from fp.battle.helpers import normalize_name
from fp.data import all_move_json, pokedex
from fp.stratagem.core.observation import Observation


class StratagemFeatureExtractor:
    """Encode only fields exposed by :class:`Observation` in a fixed layout."""

    TEAM_SLOTS = 6
    MOVE_SLOTS = 4
    ITEM_HASH_BUCKETS = 64
    BOOST_STATS = (
        constants.ATTACK,
        constants.DEFENSE,
        constants.SPECIAL_ATTACK,
        constants.SPECIAL_DEFENSE,
        constants.SPEED,
        constants.ACCURACY,
        constants.EVASION,
    )
    STATUS_VALUES = tuple(status.value for status in constants.Status)
    WEATHER_VALUES = tuple(weather.value for weather in constants.Weather)
    FIELD_VALUES = tuple(terrain.value for terrain in constants.Terrain)
    SIDE_CONDITIONS = (
        constants.REFLECT,
        constants.LIGHT_SCREEN,
        constants.AURORA_VEIL,
        constants.SAFEGUARD,
        constants.MIST,
        constants.TAILWIND,
        constants.STICKY_WEB,
        constants.WISH,
        constants.FUTURE_SIGHT,
        constants.HEALING_WISH,
        constants.STEALTH_ROCK,
        constants.SPIKES,
        constants.TOXIC_SPIKES,
    )

    def __init__(self) -> None:
        self.species_list = sorted(pokedex)
        self.move_list = sorted(all_move_json)
        self.ability_list = sorted(
            {
                normalize_name(ability)
                for entry in pokedex.values()
                for ability in entry.get(constants.ABILITIES, {}).values()
            }
        )
        self.species_to_idx = {species: index for index, species in enumerate(self.species_list)}
        self.move_to_idx = {move: index for index, move in enumerate(self.move_list)}
        self.ability_to_idx = {ability: index for index, ability in enumerate(self.ability_list)}
        self.feature_size = self._calculate_feature_size()

    def _calculate_feature_size(self) -> int:
        return (
            2 * (self.TEAM_SLOTS * self._pokemon_feature_size() + 1)
            + self._team_feature_size()
            + self._evidence_feature_size()
        )

    def _pokemon_feature_size(self) -> int:
        return (
            len(self.species_list)
            + 4
            + len(self.STATUS_VALUES) + 1
            + len(self.BOOST_STATS)
            + self.MOVE_SLOTS * len(self.move_list)
            + self.MOVE_SLOTS
            + self.ITEM_HASH_BUCKETS
            + len(self.ability_list)
        )

    def _team_feature_size(self) -> int:
        return (
            len(self.WEATHER_VALUES) + 1
            + 1
            + len(self.FIELD_VALUES) + 1
            + 1
            + 2
            + 2 * len(self.SIDE_CONDITIONS)
            + 2
        )

    def _evidence_feature_size(self) -> int:
        return 6 + 2 * len(self.move_list)

    def extract_features(self, observation: Observation) -> np.ndarray:
        features = [
            *self._extract_team_features(
                observation.active_pokemon,
                observation.reserve_pokemon,
                observation.hidden_reserve_count,
            ),
            *self._extract_team_features(
                observation.opponent_active,
                observation.opponent_reserve_revealed,
                observation.opponent_hidden_reserve_count,
            ),
            *self._extract_team_level_features(observation),
            *self._extract_evidence_features(observation),
        ]
        feature_vector = np.asarray(features, dtype=np.float32)
        if feature_vector.size != self.feature_size:
            raise ValueError(
                f"Feature layout produced {feature_vector.size} values; expected {self.feature_size}"
            )
        if not np.isfinite(feature_vector).all():
            raise ValueError("Feature layout produced non-finite values")
        return feature_vector

    def _extract_team_features(
        self,
        active_pokemon: Dict,
        reserve_pokemon: Sequence[Dict],
        hidden_count: int,
    ) -> List[float]:
        team = [active_pokemon, *reserve_pokemon[: self.TEAM_SLOTS - 1]]
        features: List[float] = []
        for pokemon in team:
            features.extend(self._extract_pokemon_features(pokemon))
        for _ in range(self.TEAM_SLOTS - len(team)):
            features.extend([0.0] * self._pokemon_feature_size())
        features.append(self._clamp(float(hidden_count) / (self.TEAM_SLOTS - 1)))
        return features

    def _extract_pokemon_features(self, pokemon: Dict) -> List[float]:
        features: List[float] = []
        species = normalize_name(str(pokemon.get("name", "")))
        features.extend(self._one_hot(self.species_to_idx, species))

        hp = pokemon.get("hp")
        max_hp = pokemon.get("max_hp")
        hp_fraction = float(hp) / float(max_hp) if hp is not None and max_hp else 0.0
        features.extend(
            (
                self._clamp(hp_fraction),
                self._clamp(float(pokemon.get("level", 0)) / 100.0),
                float(bool(pokemon.get("revealed"))),
                float(bool(pokemon.get("is_alive"))),
            )
        )

        status = pokemon.get("status")
        status_index = {name: index + 1 for index, name in enumerate(self.STATUS_VALUES)}
        features.extend(self._one_hot(status_index, str(status) if status else "", include_unknown=True))

        boosts = pokemon.get("boosts") or {}
        features.extend(
            self._clamp((float(boosts.get(stat, 0)) + constants.MAX_BOOSTS) / (2 * constants.MAX_BOOSTS))
            for stat in self.BOOST_STATS
        )

        moves = [normalize_name(str(move)) for move in pokemon.get("moves", [])]
        for move in moves[: self.MOVE_SLOTS]:
            features.extend(self._one_hot(self.move_to_idx, move))
        for _ in range(self.MOVE_SLOTS - min(len(moves), self.MOVE_SLOTS)):
            features.extend([0.0] * len(self.move_list))

        move_pp = pokemon.get("move_pp", [])
        features.extend(
            self._clamp(float(move_pp[index]) / 64.0) if index < len(move_pp) else 0.0
            for index in range(self.MOVE_SLOTS)
        )

        # Items have no repository catalogue; BLAKE2 assigns public IDs to fixed buckets.
        item_features = [0.0] * self.ITEM_HASH_BUCKETS
        item = pokemon.get("item")
        if item:
            item_features[self._item_bucket(normalize_name(str(item)))] = 1.0
        features.extend(item_features)
        features.extend(self._one_hot(self.ability_to_idx, normalize_name(str(pokemon.get("ability") or ""))))
        return features

    def _extract_team_level_features(self, observation: Observation) -> List[float]:
        weather_index = {name: index + 1 for index, name in enumerate(self.WEATHER_VALUES)}
        field_index = {name: index + 1 for index, name in enumerate(self.FIELD_VALUES)}
        weather = observation.weather.value if isinstance(observation.weather, constants.Weather) else observation.weather
        field = observation.field.value if isinstance(observation.field, constants.Terrain) else observation.field
        features = self._one_hot(weather_index, str(weather) if weather else "", include_unknown=True)
        features.append(self._clamp(float(observation.weather_turns_remaining) / 10.0))
        features.extend(self._one_hot(field_index, str(field) if field else "", include_unknown=True))
        features.extend(
            (
                self._clamp(float(observation.field_turns_remaining) / 10.0),
                float(bool(observation.trick_room)),
                self._clamp(float(observation.trick_room_turns_remaining) / 10.0),
            )
        )
        for side_conditions in (observation.side_conditions, observation.opponent_side_conditions):
            features.extend(float(condition in side_conditions) for condition in self.SIDE_CONDITIONS)
        features.extend((self._clamp(float(observation.turn) / 1000.0), float(observation.team_preview)))
        return features

    def _extract_evidence_features(self, observation: Observation) -> List[float]:
        features = [
            self._normalized_speed(observation.active_effective_speed),
            self._normalized_speed(observation.opponent_effective_speed),
            self._clamp_signed(observation.active_hp_change),
            self._clamp_signed(observation.opponent_hp_change),
            float(observation.active_is_choice_locked),
            float(observation.opponent_is_choice_locked),
        ]
        features.extend(self._one_hot(self.move_to_idx, normalize_name(observation.active_locked_move or "")))
        features.extend(self._one_hot(self.move_to_idx, normalize_name(observation.opponent_locked_move or "")))
        return features

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))

    @staticmethod
    def _clamp_signed(value: float | None) -> float:
        return 0.0 if value is None else max(-1.0, min(1.0, float(value) / 0.5))

    @classmethod
    def _normalized_speed(cls, value: float | None) -> float:
        return 0.5 if value is None else cls._clamp((float(value) - 50.0) / 150.0)

    @staticmethod
    def _one_hot(mapping: Dict[str, int], value: str, include_unknown: bool = False) -> List[float]:
        size = len(mapping) + int(include_unknown)
        features = [0.0] * size
        index = mapping.get(value)
        if index is None:
            if include_unknown:
                features[0] = 1.0
        else:
            features[index] = 1.0
        return features

    @classmethod
    def _item_bucket(cls, item: str) -> int:
        digest = hashlib.blake2b(item.encode("ascii", "ignore"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % cls.ITEM_HASH_BUCKETS


_feature_extractor: StratagemFeatureExtractor | None = None


def get_feature_extractor() -> StratagemFeatureExtractor:
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = StratagemFeatureExtractor()
    return _feature_extractor


def extract_features(observation: Observation) -> np.ndarray:
    return get_feature_extractor().extract_features(observation)
