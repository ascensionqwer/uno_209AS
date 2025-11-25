"""Tests for UNO game logic and rules."""

import pytest
from .game import Uno
from .cards import (
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
from .state import Action


# Helper functions for test setup
def create_game_with_state(H_1, H_2, top_card, deck):
    """Helper to create game with specific state."""
    game = Uno(H_1=H_1, H_2=H_2, D_g=deck, P=[top_card])
    game.current_color = top_card[0] if top_card[0] is not None else RED
    game.create_S()
    return game


# Game Initialization Tests
def test_game_initialization():
    """Verify 7 cards dealt to each player, initial top card is non-Wild, game state is Active."""
    game = Uno()
    game.new_game(seed=42)
    
    assert len(game.H_1) == 7, "Player 1 should have 7 cards"
    assert len(game.H_2) == 7, "Player 2 should have 7 cards"
    assert len(game.P) == 1, "Should have one card in played pile"
    assert game.P_t is not None, "Top card should exist"
    assert game.P_t[0] is not None, "Top card should not be Wild (should have color)"
    assert len(game.D_g) > 0, "Deck should have remaining cards"
    assert game.G_o == "Active", "Game should be Active"
    assert game.current_color is not None, "Current color should be set"


def test_game_initialization_state_creation():
    """Verify state creation works correctly."""
    game = Uno()
    game.new_game(seed=42)
    
    state = game.create_S()
    assert state is not None
    assert len(state[0]) == 7  # H_1
    assert len(state[1]) == 7  # H_2
    assert len(state[2]) > 0  # D_g
    assert len(state[3]) == 1  # P
    assert state[4] is not None  # P_t
    assert state[5] == "Active"  # G_o


# Legal Play Rules Tests
def test_legal_play_color_match():
    """Test color matching (same color as top card)."""
    game = create_game_with_state(
        H_1=[(RED, 5), (YELLOW, 3)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    assert game.is_legal_play((RED, 5)) is True, "Red card should match red top card"
    assert game.is_legal_play((YELLOW, 3)) is False, "Yellow card should not match red top card"


def test_legal_play_value_match():
    """Test value matching (same value as top card)."""
    game = create_game_with_state(
        H_1=[(RED, 5), (YELLOW, 5)],
        H_2=[],
        top_card=(GREEN, 5),
        deck=[]
    )
    
    assert game.is_legal_play((RED, 5)) is True, "Same value should match"
    assert game.is_legal_play((YELLOW, 5)) is True, "Same value should match"


def test_legal_play_wild_always_legal():
    """Test Wild cards are always legal."""
    game = create_game_with_state(
        H_1=[(BLACK, WILD), (BLACK, WILD_DRAW_4)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    assert game.is_legal_play((BLACK, WILD)) is True, "Wild should always be legal"
    assert game.is_legal_play((BLACK, WILD_DRAW_4)) is True, "Wild Draw 4 should always be legal"


def test_legal_play_illegal_rejected():
    """Test illegal plays are rejected."""
    game = create_game_with_state(
        H_1=[(YELLOW, 3), (GREEN, 8)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    assert game.is_legal_play((YELLOW, 3)) is False, "Different color and value should be illegal"
    assert game.is_legal_play((GREEN, 8)) is False, "Different color and value should be illegal"


def test_legal_play_wild_color_handling():
    """Test current_color handling for Wild cards."""
    game = create_game_with_state(
        H_1=[(RED, 5)],
        H_2=[],
        top_card=(BLACK, WILD),
        deck=[]
    )
    game.current_color = BLUE
    
    assert game.is_legal_play((RED, 5)) is False, "Should not match when current_color is BLUE"
    
    game.current_color = RED
    assert game.is_legal_play((RED, 5)) is True, "Should match when current_color is RED"


def test_get_legal_actions():
    """Test get_legal_actions returns correct actions."""
    game = create_game_with_state(
        H_1=[(RED, 5), (YELLOW, 3)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    legal_actions = game.get_legal_actions(1)
    assert len(legal_actions) == 1, "Should have one legal play (RED 5)"
    assert legal_actions[0].is_play() is True
    assert legal_actions[0].X_1 == (RED, 5)


def test_get_legal_actions_no_legal_plays():
    """Test get_legal_actions returns draw action when no legal plays."""
    game = create_game_with_state(
        H_1=[(YELLOW, 3), (GREEN, 8)],
        H_2=[],
        top_card=(RED, 7),
        deck=[(BLUE, 1)]
    )
    
    legal_actions = game.get_legal_actions(1)
    assert len(legal_actions) == 1, "Should have one action (draw)"
    assert legal_actions[0].is_draw() is True
    assert legal_actions[0].n == 1


# Special Card Effects Tests
def test_special_card_skip():
    """Test SKIP: Player plays again (2-player rule)."""
    game = create_game_with_state(
        H_1=[(RED, SKIP)],
        H_2=[],
        top_card=(RED, 5),
        deck=[]
    )
    
    action = Action(X_1=(RED, SKIP))
    result = game.execute_action(action, 1)
    
    assert result is True
    assert game.player_plays_again is True, "Player should play again after Skip"
    assert len(game.H_1) == 0, "Card should be removed from hand"


def test_special_card_reverse():
    """Test REVERSE: Player plays again (2-player rule, acts like Skip)."""
    game = create_game_with_state(
        H_1=[(RED, REVERSE)],
        H_2=[],
        top_card=(RED, 5),
        deck=[]
    )
    
    action = Action(X_1=(RED, REVERSE))
    result = game.execute_action(action, 1)
    
    assert result is True
    assert game.player_plays_again is True, "Player should play again after Reverse (2-player rule)"


def test_special_card_draw_2():
    """Test DRAW_2: Next player draws 2 and skips turn."""
    game = create_game_with_state(
        H_1=[(RED, DRAW_2)],
        H_2=[(YELLOW, 1)],
        top_card=(RED, 5),
        deck=[(BLUE, 2), (GREEN, 3)]
    )
    
    action = Action(X_1=(RED, DRAW_2))
    result = game.execute_action(action, 1)
    
    assert result is True
    assert game.draw_pending == 2, "Next player should draw 2 cards"


def test_special_card_wild():
    """Test WILD: Choose color, current_color updated."""
    game = create_game_with_state(
        H_1=[(BLACK, WILD)],
        H_2=[],
        top_card=(RED, 5),
        deck=[]
    )
    
    action = Action(X_1=(BLACK, WILD), wild_color=BLUE)
    result = game.execute_action(action, 1)
    
    assert result is True
    assert game.current_color == BLUE, "Current color should be set to chosen color"
    assert len(game.H_1) == 0, "Card should be removed from hand"


def test_special_card_wild_draw_4():
    """Test WILD_DRAW_4: Next player draws 4 and skips turn."""
    game = create_game_with_state(
        H_1=[(BLACK, WILD_DRAW_4)],
        H_2=[(YELLOW, 1)],
        top_card=(RED, 5),
        deck=[(BLUE, 2), (GREEN, 3), (RED, 4), (YELLOW, 5)]
    )
    
    action = Action(X_1=(BLACK, WILD_DRAW_4), wild_color=GREEN)
    result = game.execute_action(action, 1)
    
    assert result is True
    assert game.current_color == GREEN, "Current color should be set"
    assert game.draw_pending == 4, "Next player should draw 4 cards"


# Draw Rules Tests
def test_draw_rules_no_legal_plays():
    """Test when no legal plays, must draw 1 card."""
    game = create_game_with_state(
        H_1=[(YELLOW, 3)],
        H_2=[],
        top_card=(RED, 7),
        deck=[(BLUE, 1)]
    )
    
    legal_actions = game.get_legal_actions(1)
    assert len(legal_actions) == 1
    assert legal_actions[0].is_draw() is True
    assert legal_actions[0].n == 1


def test_draw_action_adds_cards():
    """Test draw action properly adds cards to hand."""
    game = create_game_with_state(
        H_1=[(YELLOW, 3)],
        H_2=[],
        top_card=(RED, 7),
        deck=[(BLUE, 1), (GREEN, 2)]
    )
    
    initial_hand_size = len(game.H_1)
    action = Action(n=1)
    result = game.execute_action(action, 1)
    
    assert result is True
    assert len(game.H_1) == initial_hand_size + 1, "Hand should have one more card"
    assert len(action.Y_n) == 1, "Action should record drawn cards"


def test_draw_pending_cards():
    """Test draw pending cards (from DRAW_2/WILD_DRAW_4)."""
    game = create_game_with_state(
        H_1=[],
        H_2=[(YELLOW, 1)],
        top_card=(RED, DRAW_2),
        deck=[(BLUE, 1), (GREEN, 2)]
    )
    game.draw_pending = 2
    
    action = Action(n=1)  # Player tries to draw 1, but should draw 2
    result = game.execute_action(action, 2)
    
    assert result is True
    assert len(game.H_2) == 3, "Should have drawn 2 cards (1 original + 2 pending)"
    assert game.draw_pending == 0, "Draw pending should be cleared"


# Game End Tests
def test_game_end_player_has_zero_cards():
    """Test game ends when player has 0 cards."""
    game = create_game_with_state(
        H_1=[(RED, 5)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    action = Action(X_1=(RED, 5))
    game.execute_action(action, 1)
    
    assert len(game.H_1) == 0, "Player 1 should have 0 cards"
    assert game.G_o == "GameOver", "Game should be over"


def test_game_end_no_actions_after_game_over():
    """Test no actions allowed after game over."""
    game = create_game_with_state(
        H_1=[],
        H_2=[(YELLOW, 1)],
        top_card=(RED, 7),
        deck=[]
    )
    game.G_o = "GameOver"
    
    action = Action(X_1=(YELLOW, 1))
    result = game.execute_action(action, 2)
    
    assert result is False, "Should not allow actions after game over"


# Deck Reshuffling Tests
def test_deck_reshuffle_when_empty():
    """Test when draw pile empty, reshuffle played pile (except top card)."""
    game = create_game_with_state(
        H_1=[(YELLOW, 3)],
        H_2=[],
        top_card=(YELLOW, 3),  # Top card is last in pile
        deck=[]  # Empty deck
    )
    # Add some cards to played pile (top card is last)
    game.P = [(RED, 7), (BLUE, 1), (GREEN, 2), (YELLOW, 3)]
    game.create_S()  # Update P_t
    
    top_card_before = game.P_t
    
    action = Action(n=1)
    game.execute_action(action, 1)
    
    assert len(game.D_g) > 0, "Deck should have cards after reshuffle"
    assert len(game.P) == 1, "Played pile should only have top card"
    assert game.P[0] == top_card_before, "Top card should remain"


def test_draw_after_reshuffle():
    """Test cards can still be drawn after reshuffle."""
    game = create_game_with_state(
        H_1=[(YELLOW, 3)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]  # Empty deck
    )
    game.P = [(RED, 7), (BLUE, 1), (GREEN, 2)]
    
    initial_hand_size = len(game.H_1)
    action = Action(n=1)
    result = game.execute_action(action, 1)
    
    assert result is True
    assert len(game.H_1) == initial_hand_size + 1, "Should have drawn a card after reshuffle"


# Action Execution Tests
def test_action_execution_valid_play():
    """Test valid actions execute successfully."""
    game = create_game_with_state(
        H_1=[(RED, 5)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    initial_hand_size = len(game.H_1)
    initial_pile_size = len(game.P)
    
    action = Action(X_1=(RED, 5))
    result = game.execute_action(action, 1)
    
    assert result is True
    assert len(game.H_1) == initial_hand_size - 1, "Card should be removed from hand"
    assert len(game.P) == initial_pile_size + 1, "Card should be added to played pile"


def test_action_execution_invalid_card_not_in_hand():
    """Test invalid actions are rejected (card not in hand)."""
    game = create_game_with_state(
        H_1=[(RED, 5)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    action = Action(X_1=(YELLOW, 3))  # Not in hand
    result = game.execute_action(action, 1)
    
    assert result is False, "Should reject card not in hand"


def test_action_execution_invalid_illegal_play():
    """Test invalid actions are rejected (illegal play)."""
    game = create_game_with_state(
        H_1=[(YELLOW, 3)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    action = Action(X_1=(YELLOW, 3))  # Illegal play
    result = game.execute_action(action, 1)
    
    assert result is False, "Should reject illegal play"


def test_action_execution_player_2():
    """Test action execution for Player 2."""
    game = create_game_with_state(
        H_1=[],
        H_2=[(RED, 5)],
        top_card=(RED, 7),
        deck=[]
    )
    
    action = Action(X_1=(RED, 5))
    result = game.execute_action(action, 2)
    
    assert result is True
    assert len(game.H_2) == 0, "Card should be removed from Player 2's hand"
    assert len(game.P) == 2, "Card should be added to played pile"

