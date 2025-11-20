from typing import Tuple, Union, Optional

# Color codes
RED, YELLOW, GREEN, BLUE = 'R', 'Y', 'G', 'B'
BLACK = None  # For Wild cards

# Special card values
SKIP = 'SKIP'
REVERSE = 'REVERSE'
DRAW_2 = 'DRAW_2'
WILD = 'WILD'
WILD_DRAW_4 = 'WILD_DRAW_4'

# Card is a tuple: (color: Optional[str], value: Union[int, str])
# - Number cards: (color, 0-9)
# - Special cards: (color, 'SKIP'|'REVERSE'|'DRAW_2')
# - Wild cards: (None, 'WILD'|'WILD_DRAW_4')
Card = Tuple[Optional[str], Union[int, str]]

def build_full_deck() -> list[Card]:
    """
    Builds the full 108-card UNO deck:
    - Number cards: 0-9 in 4 colors (1 zero, 2 each of 1-9 per color) = 76 cards
    - Skip: 2 per color = 8 cards
    - Reverse: 2 per color = 8 cards
    - Draw 2: 2 per color = 8 cards
    - Wild: 4 cards
    - Wild Draw 4: 4 cards
    Total: 108 cards
    """
    deck: list[Card] = []
    colors = [RED, YELLOW, GREEN, BLUE]
    
    # Number cards: 0-9
    for color in colors:
        # One 0 per color
        deck.append((color, 0))
        # Two of each 1-9 per color
        for number in range(1, 10):
            deck.append((color, number))
            deck.append((color, number))
    
    # Special cards: Skip, Reverse, Draw 2 (2 each per color)
    for color in colors:
        deck.append((color, SKIP))
        deck.append((color, SKIP))
        deck.append((color, REVERSE))
        deck.append((color, REVERSE))
        deck.append((color, DRAW_2))
        deck.append((color, DRAW_2))
    
    # Wild cards (no color)
    for _ in range(4):
        deck.append((BLACK, WILD))
    for _ in range(4):
        deck.append((BLACK, WILD_DRAW_4))
    
    return deck

def card_to_string(card: Card) -> str:
    """Convert card to readable string."""
    color, value = card
    if color is None:
        return str(value)
    if isinstance(value, int):
        return f"{color}{value}"
    return f"{color}_{value}"

