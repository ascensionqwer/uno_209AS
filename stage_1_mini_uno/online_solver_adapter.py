import sys
import os
from typing import List, Tuple, Optional
from collections import Counter

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from belief import Belief
from simulate_games import Uno_AI
from stage_1_mini_uno.mini_uno import MiniUno
from cards import Card
from pomdp import Action

class MiniBelief(Belief):
    """
    Belief state adapted for Mini Uno.
    Overrides _compute_L to use Mini Uno deck.
    """
    def _compute_L(self) -> List[Card]:
        """
        Computes L = D \ (H_1 ∪ P) - cards unknown to Player 1.
        Uses Mini Uno deck by default, or custom deck if provided.
        """
        # Build Deck
        # If self.game is set (we need to pass it to belief), we can use it.
        # But Belief init only takes observation.
        # We should pass the deck to Belief __init__.
        
        # For now, let's try to access the deck from a global or passed context.
        # Better: Update Belief to accept `deck_config` or `full_deck`.
        pass 
        
    # I will rewrite the class below to include __init__ override
    
class MiniBelief(Belief):
    def __init__(self, observation: Tuple, full_deck: List[Card] = None):
        self.full_deck = full_deck
        super().__init__(observation)
        
    def _compute_L(self) -> List[Card]:
        if self.full_deck:
            full_deck = self.full_deck
        else:
            full_deck = MiniUno().build_number_deck()

        # Count known cards: H_1 ∪ P
        known_cards = list(self.H_1) + list(self.P)
        known_counter = Counter(known_cards)

        # L = D \ (H_1 ∪ P)
        L = []
        full_counter = Counter(full_deck)

        for card, count in full_counter.items():
            unknown_count = count - known_counter.get(card, 0)
            if unknown_count < 0:
                # Should not happen if game logic is correct
                # print(f"Warning: More {card} observed than in deck!")
                unknown_count = 0
            L.extend([card] * unknown_count)

        return L

class MiniUnoAI(Uno_AI):
    """
    Uno AI adapted for Mini Uno.
    Uses MiniBelief and MiniUno game logic.
    """
    def init_belief(self, game: MiniUno):
        """Initialize belief state from game observation."""
        self.game = game
        observation = game.get_O_space()
        
        # Get deck from game if possible
        full_deck = None
        if hasattr(game, 'build_number_deck'):
            full_deck = game.build_number_deck()
            
        self.belief = MiniBelief(observation, full_deck=full_deck)

    def expectiminimax(self, state: Tuple, depth: int, player: int) -> float:
        """
        Expectiminimax search from AI perspective.
        Overridden to use MiniUno for simulation.
        """
        H_1, H_2, D_g, P, P_t, G_o = state

        # Terminal conditions
        if depth == 0 or G_o == 'GameOver':
            return self.evaluate_state(state)

        # Get legal actions for current player
        # Use game's class for logic, or fallback to MiniUno
        # If FlexibleUno, we need to pass the deck?
        # FlexibleUno constructor takes custom_deck.
        
        if hasattr(self.game, 'custom_deck'):
            # It's FlexibleUno
            from stage_1_mini_uno.flexible_uno import FlexibleUno
            temp_game = FlexibleUno(custom_deck=self.game.custom_deck)
            temp_game.H_1 = H_1
            temp_game.H_2 = H_2
            temp_game.D_g = D_g
            temp_game.P = P
            temp_game.create_S()
        else:
            temp_game = MiniUno(H_1=H_1, H_2=H_2, D_g=D_g, P=P)
            temp_game.create_S()
            
        actions = temp_game.get_legal_actions(player)

        if len(actions) == 0:
            return self.evaluate_state(state)

        # Determine if maximizing or minimizing
        is_maximizing = (player == self.player_id)

        if is_maximizing:
            max_value = float('-inf')

            for action in actions:
                next_state = self.simulate_action(state, action, player)

                # If draw action, sample possible outcomes
                if action.is_draw() and len(D_g) > 0:
                    n_samples = min(5, len(D_g))
                    total_value = 0.0

                    for _ in range(n_samples):
                        sampled_state = self.simulate_action(state, action, player)
                        value = self.expectiminimax(sampled_state, depth - 1, 3 - player)
                        total_value += value

                    avg_value = total_value / n_samples
                    max_value = max(max_value, avg_value)
                else:
                    value = self.expectiminimax(next_state, depth - 1, 3 - player)
                    max_value = max(max_value, value)

            return max_value

        else:  # Minimizing
            min_value = float('inf')

            for action in actions:
                next_state = self.simulate_action(state, action, player)

                if action.is_draw() and len(D_g) > 0:
                    n_samples = min(5, len(D_g))
                    total_value = 0.0

                    for _ in range(n_samples):
                        sampled_state = self.simulate_action(state, action, player)
                        value = self.expectiminimax(sampled_state, depth - 1, 3 - player)
                        total_value += value

                    avg_value = total_value / n_samples
                    min_value = min(min_value, avg_value)
                else:
                    value = self.expectiminimax(next_state, depth - 1, 3 - player)
                    min_value = min(min_value, value)

            return min_value

if __name__ == "__main__":
    # Simple test
    game = MiniUno()
    game.new_game(seed=42)
    
    ai = MiniUnoAI(player_id=1, num_samples=10, lookahead=2)
    ai.init_belief(game)
    
    print("Initial Belief:")
    print(ai.belief)
    
    action = ai.choose_action()
    print(f"Chosen Action: {action}")
