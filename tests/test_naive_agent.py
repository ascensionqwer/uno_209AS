"""Unit tests for naive agent behavior."""

import unittest
from unittest.mock import Mock
from src.uno.game import Uno
from src.uno.cards import RED, YELLOW, GREEN, BLUE, WILD, WILD_DRAW_4
from src.uno.state import Action
from src.utils.game_runner import choose_action_naive


class TestNaiveAgent(unittest.TestCase):
    """Test cases for naive agent behavior."""

    def setUp(self):
        """Set up test fixtures."""
        self.game = Mock(spec=Uno)
        self.game.H_1 = []
        self.game.H_2 = []
        self.game._is_wild = Mock(side_effect=lambda card: card[0] == "Black")

    def test_plays_first_legal_card(self):
        """Test that naive agent plays the first legal card found."""
        # Mock legal actions with multiple play options
        legal_actions = [
            Action(X_1=(RED, "5")),  # First legal card
            Action(X_1=(BLUE, "7")),  # Second legal card
            Action(X_1=(GREEN, "2")),  # Third legal card
        ]
        self.game.get_legal_actions.return_value = legal_actions

        result = choose_action_naive(self.game, 1)

        # Should return the first legal action
        self.assertEqual(result, legal_actions[0])
        self.assertTrue(result.is_play())
        self.assertEqual(result.X_1, (RED, "5"))

    def test_draws_when_no_legal_plays(self):
        """Test that naive agent draws when no legal plays available."""
        # Mock legal actions with only draw option
        legal_actions = [Action(n=1)]
        self.game.get_legal_actions.return_value = legal_actions

        result = choose_action_naive(self.game, 1)

        # Should return draw action
        self.assertEqual(result, legal_actions[0])
        self.assertFalse(result.is_play())
        self.assertEqual(result.n, 1)

    def test_wild_card_chooses_most_frequent_color(self):
        """Test that wild card chooses most frequent color in hand."""
        # Mock hand with more red cards
        self.game.H_1 = [
            (RED, "1"),
            (RED, "3"),
            (RED, "5"),  # 3 red cards
            (BLUE, "2"),
            (BLUE, "4"),  # 2 blue cards
            (GREEN, "6"),  # 1 green card
        ]

        # Mock legal actions with wild card
        def mock_get_legal_actions(player):
            action = Action(X_1=("Black", WILD))
            return [action]

        self.game.get_legal_actions = mock_get_legal_actions

        result = choose_action_naive(self.game, 1)

        # Should choose red (most frequent)
        self.assertEqual(result.wild_color, RED)

    def test_wild_card_random_on_tie(self):
        """Test that wild card chooses randomly when there's a tie."""
        # Mock hand with tie between red and blue
        self.game.H_1 = [
            (RED, "1"),
            (RED, "3"),  # 2 red cards
            (BLUE, "2"),
            (BLUE, "4"),  # 2 blue cards
        ]

        # Mock legal actions with wild card
        def mock_get_legal_actions(player):
            action = Action(X_1=("Black", WILD))
            return [action]

        self.game.get_legal_actions = mock_get_legal_actions

        # Test multiple times to ensure randomness
        colors_chosen = set()
        for _ in range(20):
            result = choose_action_naive(self.game, 1)
            colors_chosen.add(result.wild_color)

        # Should have chosen both red and blue (due to randomness)
        self.assertIn(RED, colors_chosen)
        self.assertIn(BLUE, colors_chosen)

    def test_wild_card_random_when_no_colors(self):
        """Test that wild card chooses randomly when hand has no colored cards."""
        # Mock hand with only wild cards
        self.game.H_1 = [("Black", WILD), ("Black", WILD_DRAW_4)]

        # Mock legal actions with wild card
        def mock_get_legal_actions(player):
            action = Action(X_1=("Black", WILD))
            return [action]

        self.game.get_legal_actions = mock_get_legal_actions

        # Test multiple times to ensure randomness
        colors_chosen = set()
        for i in range(20):
            result = choose_action_naive(self.game, 1)
            colors_chosen.add(result.wild_color)

        # Should have chosen from all four colors
        expected_colors = {RED, YELLOW, GREEN, BLUE}
        self.assertTrue(colors_chosen.issubset(expected_colors))
        self.assertGreater(len(colors_chosen), 1)  # Should have chosen multiple colors

    def test_wild_draw_4_chooses_most_frequent_color(self):
        """Test that wild draw 4 chooses most frequent color in hand."""
        # Mock hand with more green cards
        self.game.H_1 = [
            (GREEN, "1"),
            (GREEN, "3"),
            (GREEN, "5"),
            (GREEN, "7"),  # 4 green cards
            (YELLOW, "2"),  # 1 yellow card
        ]

        # Mock legal actions with wild draw 4
        def mock_get_legal_actions(player):
            action = Action(X_1=("Black", WILD_DRAW_4))
            return [action]

        self.game.get_legal_actions = mock_get_legal_actions

        result = choose_action_naive(self.game, 1)

        # Should choose green (most frequent)
        self.assertEqual(result.wild_color, GREEN)

    def test_wild_draw_4_random_on_tie(self):
        """Test that wild draw 4 chooses randomly when there's a tie."""
        # Mock hand with tie between red and blue
        self.game.H_1 = [
            (RED, "1"),
            (RED, "3"),  # 2 red cards
            (BLUE, "2"),
            (BLUE, "4"),  # 2 blue cards
        ]

        # Mock legal actions with wild draw 4
        def mock_get_legal_actions(player):
            action = Action(X_1=("Black", WILD_DRAW_4))
            return [action]

        self.game.get_legal_actions = mock_get_legal_actions

        # Test multiple times to ensure randomness
        colors_chosen = set()
        for _ in range(20):
            result = choose_action_naive(self.game, 1)
            colors_chosen.add(result.wild_color)

        # Should have chosen both red and blue (due to randomness)
        self.assertIn(RED, colors_chosen)
        self.assertIn(BLUE, colors_chosen)

    def test_wild_draw_4_random_when_no_colors(self):
        """Test that wild draw 4 chooses randomly when hand has no colored cards."""
        # Mock hand with only wild cards
        self.game.H_1 = [("Black", WILD), ("Black", WILD_DRAW_4)]

        # Mock legal actions with wild draw 4
        def mock_get_legal_actions(player):
            action = Action(X_1=("Black", WILD_DRAW_4))
            return [action]

        self.game.get_legal_actions = mock_get_legal_actions

        # Test multiple times to ensure randomness
        colors_chosen = set()
        for _ in range(20):
            result = choose_action_naive(self.game, 1)
            colors_chosen.add(result.wild_color)

        # Should have chosen from all four colors
        expected_colors = {RED, YELLOW, GREEN, BLUE}
        self.assertTrue(colors_chosen.issubset(expected_colors))
        self.assertGreater(len(colors_chosen), 1)  # Should have chosen multiple colors

    def test_ignores_wild_cards_for_color_counting(self):
        """Test that wild cards are ignored when counting colors."""
        # Mock hand with wild cards and some colored cards
        self.game.H_1 = [
            ("Black", WILD),
            ("Black", WILD_DRAW_4),  # Wild cards (should be ignored)
            ("Blue", "1"),
            ("Blue", "3"),  # 2 blue cards (should win)
            ("Red", "5"),  # 1 red card
        ]

        # Mock legal actions with wild card
        def mock_get_legal_actions(player):
            action = Action(X_1=("Black", WILD))
            return [action]

        self.game.get_legal_actions = mock_get_legal_actions

        result = choose_action_naive(self.game, 1)

        # Should choose blue (ignoring wild cards)
        self.assertEqual(result.wild_color, "Blue")

    def test_player_2_hand_used_for_player_2(self):
        """Test that player 2's hand is used when player is 2."""
        # Mock player 2 hand with more yellow cards
        self.game.H_2 = [
            (YELLOW, "1"),
            (YELLOW, "3"),
            (YELLOW, "5"),  # 3 yellow cards
            (GREEN, "2"),  # 1 green card
        ]

        # Mock legal actions with wild card
        def mock_get_legal_actions(player):
            action = Action(X_1=("Black", WILD))
            return [action]

        self.game.get_legal_actions = mock_get_legal_actions

        result = choose_action_naive(self.game, 2)

        # Should choose yellow (most frequent in player 2's hand)
        self.assertEqual(result.wild_color, YELLOW)

    def test_timing_is_recorded(self):
        """Test that decision timing is recorded."""
        # Mock legal actions
        legal_actions = [Action(X_1=(RED, "5"))]
        self.game.get_legal_actions.return_value = legal_actions

        # Reset timing stats before test
        from src.utils.game_runner import (
            reset_naive_decision_stats,
            get_naive_decision_stats,
        )

        reset_naive_decision_stats()

        # Make decision
        choose_action_naive(self.game, 1)

        # Check that timing was recorded
        stats = get_naive_decision_stats()
        self.assertEqual(stats["count"], 1)
        self.assertGreater(stats["avg"], 0)
        self.assertGreater(stats["max"], 0)

    def test_empty_legal_actions_returns_none(self):
        """Test that empty legal actions returns None."""
        self.game.get_legal_actions.return_value = []

        result = choose_action_naive(self.game, 1)

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
