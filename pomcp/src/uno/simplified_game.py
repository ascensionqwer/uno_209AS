"""Simplified UNO game engine with numbered cards only."""

import random
from typing import List
from .cards import Card, card_to_string
from .state import State, Action
from .simplified_cards import build_simplified_deck


class UnoSimplified:
    """
    Simplified UNO game - NUMBERS ONLY (0-max_number).
    No special cards: no wild, +2, skip, reverse.
    D = H_1 ⊔ H_2 ⊔ D_g ⊔ P : entire deck partitioned into hands, draw pile, played pile.
    """

    def __init__(
        self,
        max_number: int = 9,
        max_colors: int = 4,
        H_1: List[Card] = None,
        H_2: List[Card] = None,
        D_g: List[Card] = None,
        P: List[Card] = None,
    ):
        self.max_number = max_number
        self.max_colors = max_colors
        self.H_1 = H_1 if H_1 is not None else []
        self.H_2 = H_2 if H_2 is not None else []
        self.D_g = D_g if D_g is not None else []
        self.P = P if P is not None else []
        self.G_o = "Active"
        self.State = None
        self.P_t = None

    def build_full_deck(self) -> List[Card]:
        """Builds simplified deck with numbered cards only."""
        return build_simplified_deck(self.max_number, self.max_colors)

    def new_game(self, seed: int = None, deal: int = 7):
        """
        Initializes a new game state.
        - Shuffles simplified deck (numbered cards only).
        - Deals 'deal' cards to H_1 and H_2.
        - D_g gets the remainder.
        - P starts with first card from D_g.
        """
        rng = random.Random(seed)
        full_deck = self.build_full_deck()
        rng.shuffle(full_deck)

        # Deal to hands
        self.H_1 = full_deck[:deal]
        self.H_2 = full_deck[deal : 2 * deal]
        remaining = full_deck[2 * deal :]

        # Start played pile with first card
        if remaining:
            self.P = [remaining.pop(0)]
        else:
            # Fallback: reshuffle everything
            full_deck = self.build_full_deck()
            rng.shuffle(full_deck)
            self.H_1 = full_deck[:deal]
            self.H_2 = full_deck[deal : 2 * deal]
            remaining = full_deck[2 * deal :]
            self.P = [remaining.pop(0)]

        self.D_g = remaining
        self.G_o = "Active"

        self.create_S()

    def create_S(self) -> State:
        """
        Creates the system state S as seen by "God".
        S = (H_1, H_2, D_g, P, P_t, G_o)
        """
        self.P_t = self.P[-1] if len(self.P) > 0 else None
        self.State = (self.H_1, self.H_2, self.D_g, self.P, self.P_t, self.G_o)
        return self.State

    def update_S(self):
        """Updates the system state."""
        if len(self.H_1) == 0 or len(self.H_2) == 0:
            self.G_o = "GameOver"
        self.create_S()

    def print_S(self):
        """Print current game state."""
        if self.State is None:
            self.create_S()
        print(
            f"H_1: {[card_to_string(c) for c in self.State[0]]}, {len(self.State[0])} cards"
        )
        print(
            f"H_2: {[card_to_string(c) for c in self.State[1]]}, {len(self.State[1])} cards"
        )
        print(f"D_g: {len(self.State[2])} cards")
        print(
            f"P: {len(self.State[3])} cards, top: {card_to_string(self.State[4]) if self.State[4] else 'None'}"
        )
        print(f"G_o: {self.State[5]}")

    def is_legal_play(self, card: Card) -> bool:
        """
        Checks if a card can be legally played.
        Legal if:
        - Card color matches P_t color OR
        - Card value matches P_t value
        """
        if self.P_t is None:
            return False

        card_color, card_value = card
        P_t_color, P_t_value = self.P_t

        # Match by color
        if card_color == P_t_color:
            return True

        # Match by value
        if card_value == P_t_value:
            return True

        return False

    def get_legal_actions(self, player: int = 1) -> List[Action]:
        """
        Returns list of legal actions for specified player.
        """
        if self.State is None:
            self.create_S()
        legal_actions = []
        hand = self.H_1 if player == 1 else self.H_2

        if self.P_t is None:
            return legal_actions

        # Check which cards in hand can be played
        for card in hand:
            if self.is_legal_play(card):
                legal_actions.append(Action(X_1=card))

        # If no legal plays, must draw 1 card
        if len(legal_actions) == 0:
            legal_actions.append(Action(n=1))

        return legal_actions

    def execute_action(self, action: Action, player: int = 1) -> bool:
        """
        Executes an action and updates the game state.
        Simplified: no special card effects.
        """
        if self.G_o == "GameOver":
            return False

        hand = self.H_1 if player == 1 else self.H_2

        if action.is_play():
            card = action.X_1
            if card not in hand:
                return False

            if not self.is_legal_play(card):
                return False

            # Remove from hand and add to pile
            hand.remove(card)
            self.P.append(card)

        elif action.is_draw():
            # Draw n cards
            n = action.n

            cards_drawn = []
            for i in range(n):
                if len(self.D_g) == 0:
                    # Reshuffle played pile (keep top card)
                    if len(self.P) <= 1:
                        # Can't reshuffle if only one card
                        break
                    cards_to_shuffle = self.P[:-1]
                    self.P = [self.P[-1]]
                    random.shuffle(cards_to_shuffle)
                    self.D_g = cards_to_shuffle

                if len(self.D_g) > 0:
                    card = self.D_g.pop()
                    cards_drawn.append(card)
                    hand.append(card)

            action.Y_n = cards_drawn

        self.update_S()
        return True

    def get_O_space(self):
        """
        O = (H_1, |H_2|, |D_g|, P, P_t, G_o) - observation (as seen by "Player 1")
        """
        if self.State is None:
            self.create_S()
        self.O = (self.H_1, len(self.H_2), len(self.D_g), self.P, self.P_t, self.G_o)
        return self.O
