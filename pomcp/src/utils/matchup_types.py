"""Matchup type enums for UNO simulations."""

from enum import Enum


class PlayerType(Enum):
    """Player type enumeration."""

    NAIVE = "naive"
    PARTICLE_POLICY = "particle_policy"


class Matchup:
    """Matchup configuration for two players."""

    def __init__(self, player1_type: PlayerType, player2_type: PlayerType):
        self.player1_type = player1_type
        self.player2_type = player2_type

    def __str__(self) -> str:
        return f"{self.player1_type.value}_vs_{self.player2_type.value}"

    def __repr__(self) -> str:
        return f"Matchup({self.player1_type.value}, {self.player2_type.value})"
