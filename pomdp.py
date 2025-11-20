# pomdp.py:
from typing import List, Tuple, Optional
from cards import Card

# State type: S = (H_1, H_2, D_g, P, P_t, G_o)
State = Tuple[List[Card], List[Card], List[Card], List[Card], Optional[Card], str]


# Action type: A = {X_1} ∪ {Y_n : n ∈ {1, 2, 4}}
class Action:
    """
    Action space for UNO game.
    A = {X_1}, {Y_n : n ∈ {1, 2, 4}}

    An action is EITHER:
    - X_1: a single card to play from hand
    - Y_n: draw n cards from deck where n ∈ {1, 2, 4}
    """

    def __init__(self, X_1: Optional[Card] = None, n: Optional[int] = None):
        """
        Initialize an action - must specify EITHER X_1 OR n, not both.

        Args:
            X_1: Card to play (COLOR, VALUE) - for PLAY action
            n: Number of cards to draw, n ∈ {1, 2, 4} - for DRAW action
        """
        # Validate that exactly one is specified
        if (X_1 is None and n is None) or (X_1 is not None and n is not None):
            raise ValueError("Must specify EITHER X_1 (play) OR n (draw), not both or neither")

        self.X_1 = X_1  # Card to play (None if draw action)
        self.n = n      # Number to draw (None if play action)
        self.Y_n = []   # Cards actually drawn (filled during execution)

        # Validation for draw action
        if n is not None and n not in [1, 2, 4]:
            raise ValueError(f"n must be in {{1, 2, 4}}, got {n}")

    def is_play(self) -> bool:
        """Returns True if this is a PLAY action (X_1)"""
        return self.X_1 is not None

    def is_draw(self) -> bool:
        """Returns True if this is a DRAW action (Y_n)"""
        return self.n is not None

    def __repr__(self):
        if self.is_play():
            return f"Action(X_1={self.X_1})"
        elif self.is_draw():
            if len(self.Y_n) > 0:
                return f"Action(Y_{self.n}={self.Y_n})"
            else:
                return f"Action(Y_{self.n}=[])"