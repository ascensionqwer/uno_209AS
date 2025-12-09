import sys
import os
from typing import List
import random

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uno import Uno
from cards import Card, RED, BLUE, GREEN, YELLOW

class FlexibleUno(Uno):
    """
    Uno variant that allows a custom deck composition.
    Used for testing with varying deck sizes (10, 11, 12...).
    """
    def __init__(self, custom_deck: List[Card] = None):
        """
        Args:
            custom_deck: List of cards to use as the full deck.
        """
        super().__init__()
        self.custom_deck = custom_deck

    def build_number_deck(self) -> List[Card]:
        """
        Returns the custom deck if provided, otherwise defaults to standard Uno deck.
        """
        if self.custom_deck:
            return list(self.custom_deck) # Return copy
        return super().build_number_deck()

    def new_game(self, seed: int = None, deal: int = 2):
        """
        Initializes a new game.
        Args:
            seed: Random seed.
            deal: Number of cards to deal to each player (default 2 for Mini Uno style).
        """
        super().new_game(seed=seed, deal=deal)

def generate_deck(size: int) -> List[Card]:
    """
    Generates a balanced deck of a specific size.
    Starts with Mini Uno base (10 cards) and adds cards to reach size.
    
    Mini Uno Base (10):
    - Red: 0, 1, 1, 2, 2
    - Blue: 0, 1, 1, 2, 2
    
    Expansion strategy:
    - Add cards in round-robin fashion across colors and ranks to maintain balance.
    """
    # Base Mini Uno Deck
    deck = []
    colors = [RED, BLUE]
    
    # We want to deterministically generate a deck of size N.
    # Let's define a sequence of cards to add.
    # Base 10:
    for c in colors:
        deck.append((c, 0))
        deck.append((c, 1)); deck.append((c, 1))
        deck.append((c, 2)); deck.append((c, 2))
        
    if size < 10:
        return deck[:size] # Truncate if smaller (unlikely)
        
    if size == 10:
        return deck
        
    # Add more cards
    # Sequence of potential additions:
    # (R, 3), (B, 3), (R, 4), (B, 4)...
    # Or add duplicates of existing?
    # Adding new ranks is probably better to avoid "too many 1s".
    
    extra_needed = size - 10
    
    # Generate pool of extra cards
    extras = []
    for rank in range(3, 10): # 3 to 9
        for c in colors:
            extras.append((c, rank))
            extras.append((c, rank)) # Two of each
            
    # If we need even more (e.g. > 10 + 14 = 24), we can add Green/Yellow
    # But for this experiment, N=10, 11, 12... likely won't exceed 24 quickly.
    
    if extra_needed > len(extras):
        # Add Green/Yellow if needed
        more_colors = [GREEN, YELLOW]
        for c in more_colors:
            extras.append((c, 0))
            for r in range(1, 10):
                extras.append((c, r))
                extras.append((c, r))
                
    deck.extend(extras[:extra_needed])
    
    return deck

if __name__ == "__main__":
    # Test
    for n in [10, 11, 12, 15]:
        d = generate_deck(n)
        print(f"Deck size {n}: {len(d)} cards")
        print(d)
