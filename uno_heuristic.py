from typing import Optional
from pomdp import Action
from uno import Uno

class Uno_Heuristic:
    """
    Simple heuristic player: plays first legal card, otherwise draws.

    This implements the baseline strategy:
    - Check hand for legal cards
    - Play first legal card found
    - If no legal cards, draw from position [0]
    """

    def __init__(self, player_id: int = 2):
        """
        Initialize heuristic player.

        Args:
            player_id: Player number (1 or 2)
        """
        self.player_id = player_id
        self.game = None

    def set_game(self, game: 'Uno'):
        """Set game reference."""
        self.game = game

    def choose_action(self) -> Optional[Action]:
        """
        Choose action using simple heuristic.

        Returns:
            First legal card to play, or draw action if no legal plays
        """
        if not self.game:
            raise ValueError("Game not set - call set_game() first")

        # Get legal actions
        actions = self.game.get_legal_actions(self.player_id)

        if len(actions) == 0:
            return None

        # Return first action (greedy: play first legal card or draw)
        return actions[0]