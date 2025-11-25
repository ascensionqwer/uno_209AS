"""Tests for UNO card and deck building logic."""

from .cards import (
    build_full_deck,
    card_to_string,
    RED,
    YELLOW,
    GREEN,
    BLUE,
    SKIP,
    REVERSE,
    DRAW_2,
    WILD,
    WILD_DRAW_4,
    BLACK,
)


def test_deck_size():
    """Verify full deck has exactly 108 cards."""
    deck = build_full_deck()
    assert len(deck) == 108


def test_deck_distribution():
    """Verify correct distribution of card types."""
    deck = build_full_deck()

    number_cards = [c for c in deck if isinstance(c[1], int)]
    skip_cards = [c for c in deck if c[1] == SKIP]
    reverse_cards = [c for c in deck if c[1] == REVERSE]
    draw_2_cards = [c for c in deck if c[1] == DRAW_2]
    wild_cards = [c for c in deck if c[1] == WILD]
    wild_draw_4_cards = [c for c in deck if c[1] == WILD_DRAW_4]

    assert len(number_cards) == 76  # 4 colors * (1 zero + 2 each of 1-9)
    assert len(skip_cards) == 8  # 2 per color * 4 colors
    assert len(reverse_cards) == 8  # 2 per color * 4 colors
    assert len(draw_2_cards) == 8  # 2 per color * 4 colors
    assert len(wild_cards) == 4
    assert len(wild_draw_4_cards) == 4


def test_number_card_distribution():
    """Verify one 0 per color, two of each 1-9 per color."""
    deck = build_full_deck()
    colors = [RED, YELLOW, GREEN, BLUE]

    for color in colors:
        zeros = [c for c in deck if c == (color, 0)]
        assert len(zeros) == 1, f"Should have exactly 1 zero card in {color}"

        for number in range(1, 10):
            number_cards = [c for c in deck if c == (color, number)]
            assert len(number_cards) == 2, (
                f"Should have exactly 2 {color}{number} cards"
            )


def test_special_card_distribution():
    """Verify special cards (Skip, Reverse, Draw 2) have 2 per color."""
    deck = build_full_deck()
    colors = [RED, YELLOW, GREEN, BLUE]
    special_types = [SKIP, REVERSE, DRAW_2]

    for color in colors:
        for special_type in special_types:
            cards = [c for c in deck if c == (color, special_type)]
            assert len(cards) == 2, (
                f"Should have exactly 2 {color} {special_type} cards"
            )


def test_wild_card_distribution():
    """Verify Wild cards have no color and correct count."""
    deck = build_full_deck()

    wild_cards = [c for c in deck if c[1] == WILD]
    wild_draw_4_cards = [c for c in deck if c[1] == WILD_DRAW_4]

    assert len(wild_cards) == 4
    assert len(wild_draw_4_cards) == 4

    # All wild cards should have BLACK (None) color
    for card in wild_cards + wild_draw_4_cards:
        assert card[0] == BLACK, "Wild cards should have BLACK (None) color"


def test_card_to_string():
    """Test card_to_string() function for various card types."""
    # Number card
    assert card_to_string((RED, 5)) == "R5"
    assert card_to_string((YELLOW, 0)) == "Y0"

    # Special cards
    assert card_to_string((GREEN, SKIP)) == "G_SKIP"
    assert card_to_string((BLUE, REVERSE)) == "B_REVERSE"
    assert card_to_string((RED, DRAW_2)) == "R_DRAW_2"

    # Wild cards
    assert card_to_string((BLACK, WILD)) == "WILD"
    assert card_to_string((BLACK, WILD_DRAW_4)) == "WILD_DRAW_4"
