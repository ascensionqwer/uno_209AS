import sys
import os
import unittest

# Add parent directory to path to import uno
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage_1_mini_uno.mini_uno import MiniUno
from cards import RED, BLUE

class TestMiniUno(unittest.TestCase):
    def setUp(self):
        self.game = MiniUno()
        self.game.new_game(seed=42)

    def test_deck_size(self):
        deck = self.game.build_number_deck()
        self.assertEqual(len(deck), 10, "Deck size should be 10 for Mini Uno")
        
        # Verify composition
        red_cards = [c for c in deck if c[0] == RED]
        blue_cards = [c for c in deck if c[0] == BLUE]
        self.assertEqual(len(red_cards), 5)
        self.assertEqual(len(blue_cards), 5)
        
        # Verify ranks
        red_ranks = sorted([c[1] for c in red_cards])
        self.assertEqual(red_ranks, [0, 1, 1, 2, 2])

    def test_initial_state(self):
        # Check hands
        self.assertEqual(len(self.game.H_1), 2)
        self.assertEqual(len(self.game.H_2), 2)
        
        # Check pile
        self.assertEqual(len(self.game.P), 1)
        self.assertIsNotNone(self.game.P_t)
        
        # Check remaining deck
        # 10 total - 2 (H1) - 2 (H2) - 1 (P) = 5
        self.assertEqual(len(self.game.D_g), 5)

    def test_legal_moves(self):
        # Force a specific state
        # P_t = (RED, 1)
        # H_1 = [(RED, 2), (BLUE, 1), (BLUE, 0)]
        self.game.P = [(RED, 1)]  # Must update P because update_S derives P_t from P
        self.game.H_1 = [(RED, 2), (BLUE, 1), (BLUE, 0)]
        self.game.update_S()
        
        actions = self.game.get_legal_actions(player=1)
        # Legal: (RED, 2) matches color, (BLUE, 1) matches rank
        # (BLUE, 0) is not legal
        legal_cards = [a.X_1 for a in actions if a.is_play()]
        self.assertIn((RED, 2), legal_cards)
        self.assertIn((BLUE, 1), legal_cards)
        self.assertNotIn((BLUE, 0), legal_cards)

if __name__ == '__main__':
    unittest.main()
