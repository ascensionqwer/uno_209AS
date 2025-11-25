"""Particle class for particle filter."""

from typing import List
from ..uno.cards import Card


class Particle:
    """Represents a particle in the particle filter."""

    def __init__(self, H_2: List[Card], D_g: List[Card], weight: float = 1.0):
        self.H_2 = H_2
        self.D_g = D_g
        self.weight = weight
