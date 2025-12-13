"""Simplified UNO deck with numbered cards only (no special/wild cards)."""

from typing import List
from .cards import Card, RED, YELLOW, GREEN, BLUE


def build_simplified_deck(max_number: int = 9, max_colors: int = 4) -> List[Card]:
    """
    Builds simplified UNO deck with numbered cards only.

    Args:
        max_number: Maximum card number (default 9, creates cards 0-9)
        max_colors: Number of colors to use (default 4, uses all colors)

    Returns:
        List of cards with pattern:
        - 1 zero per color
        - 2 of each number (1 to max_number) per color

    Example:
        max_number=9, max_colors=4 → 4*(1 + 2*9) = 76 cards
    """
    deck: List[Card] = []
    all_colors = [RED, YELLOW, GREEN, BLUE]
    colors = all_colors[:max_colors]

    # Number cards: 0 to max_number
    for color in colors:
        # One 0 per color
        deck.append((color, 0))
        # Two of each 1-max_number per color
        for number in range(1, max_number + 1):
            deck.append((color, number))
            deck.append((color, number))

    return deck
