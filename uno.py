from typing import List
import random
from cards import RED, YELLOW, GREEN, BLUE, Card

class Uno:
    """
    Simplified UNO initialization with only number cards (no specials).
    D = H_1 ⊔ H_2 ⊔ D_g ⊔ P : entire deck partitioned into hands, draw pile, played pile.
    Builds 76-card deck: 4 colors, (1 zero + 2 each of 1-9).
    """
    def __init__(self, H_1: List[Card] = None, H_2: List[Card] = None, D_g: List[Card] = None, P: List[Card] = None):
        self.H_1 = H_1 if H_1 is not None else [] # Player 1 hand
        self.H_2 = H_2 if H_2 is not None else [] # Player 2 hand
        self.D_g = D_g if D_g is not None else [] # Current draw deck (top card at end)
        self.P = P if P is not None else [] # Played cards (top card at end)

    def build_number_deck(self) -> List[Card]:
        """
        Builds the full number-only deck.
        - One 0 per color.
        - Two each of 1 through 9 per color.
        Returns list of 76 cards.
        """
        deck: List[Card] = []
        colors = [RED, YELLOW, GREEN, BLUE]
        for color in colors:
            # Add the single 0 for this color
            deck.append((color, 0))
            # Add two of each number 1-9 for this color
            for number in range(1, 10):
                deck.append((color, number))
                deck.append((color, number))
        return deck

    def new_game(self, seed: int = None, deal: int = 7):
        """
        Initializes a new game state on this instance.
        - Shuffles the full number deck.
        - Deals 'deal' cards to H_1 and H_2.
        - D_g gets the remainder.
        - P starts with the first card from D_g (top card).
        """
        rng = random.Random(seed)
        full_deck = self.build_number_deck()
        rng.shuffle(full_deck)
        
        # Deal to hands
        self.H_1 = full_deck[:deal]
        self.H_2 = full_deck[deal:2 * deal]
        remaining = full_deck[2 * deal:]
        
        # Start played pile with top of remaining (pop from front)
        self.P = [remaining.pop(0)]
        self.D_g = remaining
        
        print(f"H_1: {self.H_1}, {len(self.H_1)}")
        print(f"H_2: {self.H_2}, {len(self.H_2)}")
        print(f"D_g: {self.D_g}, {len(self.D_g)}")
        print(f"P: {self.P}, {len(self.P)}")
        

uno = Uno()
uno.new_game()