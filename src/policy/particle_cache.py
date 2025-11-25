"""Particle cache for dynamic particle filter generation (Player 1 perspective)."""

import random
from typing import List
from ..uno.cards import Card, build_full_deck
from .particle import Particle


class ParticleCache:
    """
    Cache for particle sets keyed by game state.
    Generates particles dynamically at runtime and caches them for reuse.

    Particles represent beliefs about Player 2's hidden hand (H_2).
    Each particle contains a sampled H_2 and corresponding D_g.
    """

    def __init__(self):
        """Initialize empty cache."""
        self.cache = {}  # game_state_key -> List[Particle]

    def get_particles(
        self,
        game_state_key: str,
        H_1: List[Card],
        opponent_size: int,
        deck_size: int,
        P: List[Card],
        num_particles: int,
    ) -> List[Particle]:
        """
        Get particles for a game state, generating them if not cached.

        Args:
            game_state_key: Canonical game state key
            H_1: Player 1's hand
            opponent_size: Size of opponent's hand
            deck_size: Size of deck (will be auto-corrected to |L| - opponent_size)
            P: Played cards
            num_particles: Number of particles to generate

        Returns:
            List of Particle objects
        """
        if game_state_key not in self.cache:
            self.cache[game_state_key] = self._generate_particles(
                H_1, opponent_size, deck_size, P, num_particles
            )
        return self.cache[game_state_key]

    def _generate_particles(
        self,
        H_1: List[Card],
        opponent_size: int,
        deck_size: int,
        P: List[Card],
        num_particles: int,
    ) -> List[Particle]:
        """
        Generate particles dynamically at runtime.
        
        Each particle represents a possible complete game state s = (H_1, H_2, D_g, P, P_t, G_o).
        We only sample the hidden parts (H_2, D_g) since H_1, P, P_t, G_o are known.
        
        Filtering: Only sample from L = D minus (H_1 union P) to ensure legally possible states.
        This means if Player 1 has all blue 3s, no particle will contain blue 3s in H_2 or D_g.
        """
        full_deck = build_full_deck()
        
        # Filter: only cards NOT in Player 1's hand or played pile
        # This is the key constraint - we can only sample from available cards
        L = [c for c in full_deck if c not in H_1 and c not in P]
        
        # Validate: ensure we have enough cards for opponent hand + deck
        if len(L) < opponent_size:
            return []

        # Auto-correct deck_size to match mathematical constraint: |D_g| = |L| - |H_2|
        actual_deck_size = len(L) - opponent_size
        if actual_deck_size < 0:
            return []

        particles = []
        for _ in range(num_particles):
            # Sample opponent hand uniformly from available cards only
            H_2 = random.sample(L, opponent_size)
            
            # Remaining cards go to deck (all cards in L that aren't in H_2)
            remaining = [c for c in L if c not in H_2]
            D_g = remaining.copy()

            # Validation: ensure no cards from H_1 or P appear in particle
            assert all(c not in H_1 for c in H_2), "Particle H_2 contains cards from H_1!"
            assert all(c not in P for c in H_2), "Particle H_2 contains cards from P!"
            assert all(c not in H_1 for c in D_g), "Particle D_g contains cards from H_1!"
            assert all(c not in P for c in D_g), "Particle D_g contains cards from P!"

            particles.append(Particle(H_2, D_g, 1.0 / num_particles))

        return particles

    def clear(self):
        """Clear the cache."""
        self.cache.clear()

    def size(self) -> int:
        """Get number of cached game states."""
        return len(self.cache)
