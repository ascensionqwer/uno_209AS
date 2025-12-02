import sys
import os

# Add parent directory to path to import uno
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uno import Uno
from cards import Card, RED, BLUE
from typing import List
import random

class MiniUno(Uno):
    """
    Mini Uno implementation for solver verification.
    - 2 Colors: RED, BLUE
    - Ranks: 0, 1, 2
    - Deck composition per color: One 0, Two 1s, Two 2s.
    - Total cards: 10
    - Hand size: 2
    """
    def __init__(self, H_1: List[Card] = None, H_2: List[Card] = None, D_g: List[Card] = None, P: List[Card] = None):
        super().__init__(H_1, H_2, D_g, P)

    def build_number_deck(self) -> List[Card]:
        """
        Builds the Mini Uno deck.
        - One 0 per color.
        - Two each of 1 and 2 per color.
        Returns list of 10 cards.
        """
        deck: List[Card] = []
        colors = [RED, BLUE]
        for color in colors:
            # Add the single 0 for this color
            deck.append((color, 0))
            # Add two of each number 1-2 for this color
            for number in range(1, 3):
                deck.append((color, number))
                deck.append((color, number))
        return deck

    def new_game(self, seed: int = None, deal: int = 2):
        """
        Initializes a new game state on this instance.
        - Shuffles the Mini Uno deck.
        - Deals 'deal' cards to H_1 and H_2 (default 2).
        - D_g gets the remainder.
        - P starts with the first card from D_g (top card).
        """
        # Use the same logic as parent but with default deal=2
        super().new_game(seed=seed, deal=deal)

if __name__ == "__main__":
    # Simple test
    mini_uno = MiniUno()
    mini_uno.new_game(seed=42)
    mini_uno.print_S()
    print(f"Deck size: {len(mini_uno.build_number_deck())}")
