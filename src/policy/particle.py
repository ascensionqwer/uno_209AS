"""Particle class for particle filter."""

from typing import List
from ..uno.cards import Card


class Particle:
    """
    Represents a particle in the particle filter.

    A particle represents a possible complete game state s = (H_1, H_2, D_g, P, P_t, G_o).
    We only store the hidden parts (H_2, D_g) since H_1, P, P_t, G_o are observable.
    When used, combine with observable state to get full state.

    Critical constraint: H_2 and D_g can only contain cards NOT in H_1 or P.
    This ensures we only sample from legally possible states.
    """

    def __init__(self, H_2: List[Card], D_g: List[Card], weight: float = 1.0):
        self.H_2 = H_2  # Opponent's hand (hidden)
        self.D_g = D_g  # Deck (hidden)
        self.weight = weight
