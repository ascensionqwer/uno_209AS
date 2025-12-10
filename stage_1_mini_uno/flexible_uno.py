import sys
import os
from typing import List, Tuple
from cards import Card, RED, BLUE, GREEN, YELLOW

# Add parent directory to path to import uno
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uno import Uno

class FlexibleUno(Uno):
    """
    Uno implementation with customizable deck parameters.
    """
    def __init__(self, 
                 colors: List[str] = [RED, BLUE], 
                 ranks: List[int] = [0, 1, 2], 
                 copies: int = 2,
                 H_1: List[Card] = None, 
                 H_2: List[Card] = None, 
                 D_g: List[Card] = None, 
                 P: List[Card] = None):
        
        self.custom_colors = colors
        self.custom_ranks = ranks
        self.custom_copies = copies
        super().__init__(H_1, H_2, D_g, P)

    def build_number_deck(self) -> List[Card]:
        """
        Builds the deck based on custom parameters.
        """
        deck: List[Card] = []
        for color in self.custom_colors:
            for rank in self.custom_ranks:
                # Usually 0 has 1 copy, others have 'copies' amount?
                # Or just apply 'copies' to all for simplicity in experiments?
                # Let's apply 'copies' to all for uniform scaling, 
                # or follow standard Uno (1 zero, 2 others).
                # User asked for "different card pools".
                # Let's make it simple: 'copies' amount of EACH card.
                for _ in range(self.custom_copies):
                    deck.append((color, rank))
        return deck

    def new_game(self, seed: int = None, deal: int = 2):
        """
        Initializes a new game state.
        """
        super().new_game(seed=seed, deal=deal)
