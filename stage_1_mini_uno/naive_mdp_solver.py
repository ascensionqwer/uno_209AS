import sys
import os
from typing import List, Tuple, Optional
from collections import Counter
import random

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage_1_mini_uno.online_solver_adapter import MiniUnoAI, MiniBelief
from stage_1_mini_uno.mini_uno import MiniUno
from cards import Card
from pomdp import Action

class NaiveBelief(MiniBelief):
    """
    Naive Belief state.
    Instead of updating based on history, it simply regenerates particles 
    consistent with the CURRENT observation (hand size, discards) every time.
    """
    def __init__(self, observation: Tuple, num_particles: int = 100, full_deck: List[Card] = None):
        super().__init__(observation, full_deck=full_deck)
        self.num_particles = num_particles
        self.particles = []

    def update(self, action: Action, observation: Tuple):
        """
        Override update to ignore the action history and just re-initialize 
        from the new observation.
        """
        self.H_1, self.H_2_size, self.D_g, self.P, self.P_t, self.G_o = observation
        # Re-compute L and N_Pt based on new observation
        self.L = self._compute_L()
        self.N_Pt = self._compute_legal_unknown()
        self.posterior_mode = None # Reset posterior mode as we are ignoring history
        
        # Sample new particles
        self.particles = [self.sample_state() for _ in range(self.num_particles)]

class NaiveMiniUnoAI(MiniUnoAI):
    """
    Naive MDP Solver.
    Uses NaiveBelief to ignore history.
    """
    def init_belief(self, game: MiniUno):
        """Initialize belief state from game observation."""
        self.game = game
        observation = game.get_O_space()
        # Pass the specific deck from the game instance
        full_deck = game.build_number_deck()
        # Use NaiveBelief instead of MiniBelief
        self.belief = NaiveBelief(observation, num_particles=self.num_samples, full_deck=full_deck)

    def update_belief(self, opponent_action: Action):
        """
        Update belief.
        For Naive solver, this triggers a full re-sample based on current state,
        effectively ignoring the specific action taken by opponent (other than how it changed the visible state).
        """
        if self.belief:
            new_observation = self.game.get_O_space()
            self.belief.update(opponent_action, new_observation)
