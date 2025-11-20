# uno.py

import random
from typing import List
from collections import Counter
from math import comb
from cards import RED, YELLOW, GREEN, BLUE, Card
from pomdp import State, Action

class Uno:
    """
    Simplified UNO initialization with only number cards (no specials).
    D = H_1 ⊔ H_2 ⊔ D_g ⊔ P : entire deck partitioned into hands, draw pile, played pile.
    Builds 76-card deck: 4 colors, (1 zero + 2 each of 1-9).
    """
    def __init__(self, H_1: List[Card] = None, H_2: List[Card] = None, D_g: List[Card] = None, P: List[Card] = None):
        self.H_1 = H_1 if H_1 is not None else [] # Player 1 hand
        self.H_2 = H_2 if H_2 is not None else [] # Player 2 hand
        self.D_g = D_g if D_g is not None else [] # Current draw deck (as list for drawing)
        self.P = P if P is not None else [] # Played cards (top card at end)
        self.G_o = "Active" # Game status: "Active" or "GameOver"
        self.State = None
        self.P_t = None

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
        self.G_o = "Active"

        self.create_S()

    def create_S(self) -> State:
        """
        Creates the system state S as seen by "God".
        S = (H_1, H_2, D_g, P, P_t, G_o)
        - H_1: Player 1 hand (list)
        - H_2: Player 2 hand (list)
        - D_g: Current deck (list)
        - P: Played cards pile (list)
        - P_t: Top card of pile (COLOR, VALUE) or None if empty
        - G_o: Game status ("Active" or "GameOver")
        """
        # Get top card P_t (last card in pile, or None if empty)
        self.P_t = self.P[-1] if len(self.P) > 0 else None

        self.State = (self.H_1, self.H_2, self.D_g, self.P, self.P_t, self.G_o)
        return self.State

    def update_S(self):
        """
        Updates the system state.
        Any parameter that is None will not be updated.
        """
        # Check game over condition: if either hand is empty
        if len(self.H_1) == 0 or len(self.H_2) == 0:
            self.G_o = "GameOver"
        self.create_S()  # Recompute P_t and State with updated G_o


    def print_S(self):
        if self.State is None:
            self.create_S()
        print(f"H_1: {self.State[0]}, {len(self.State[0])} cards")
        print(f"H_2: {self.State[1]}, {len(self.State[1])} cards")
        print(f"D_g: {self.State[2]}, {len(self.State[2])} cards")
        print(f"P: {self.State[3]}, {len(self.State[3])} cards")
        print(f"P_t: {self.State[4]}")
        print(f"G_o: {self.State[5]}")

    def is_legal_play(self, card: Card) -> bool:
        """
        Checks if a card can be legally played on top of P_t.
        Legal if: card color matches P_t color OR card value matches P_t value

        Args:
            card: Card to check (COLOR, VALUE)

        Returns:
            True if card is legal to play, False otherwise
        """
        if self.P_t is None:
            return False

        P_t_color, P_t_value = self.P_t
        card_color, card_value = card

        return card_color == P_t_color or card_value == P_t_value

    def get_legal_actions(self, player: int = 1) -> List[Action]:
        """
        Returns list of legal actions for specified player.

        Args:
            player: 1 for H_1, 2 for H_2

        Returns:
            List of legal Action objects
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

        Args:
            action: Action to execute (either X_1 play or Y_n draw)
            player: 1 for H_1, 2 for H_2

        Returns:
            True if action was successful, False otherwise
        """
        if self.G_o == "GameOver":
            print("Game is over, no actions allowed")
            return False

        # Select the correct hand
        hand = self.H_1 if player == 1 else self.H_2

        if action.is_play():
            # Play card X_1
            if action.X_1 not in hand:
                print(f"Card {action.X_1} not in player {player}'s hand")
                return False

            if not self.is_legal_play(action.X_1):
                print(f"Card {action.X_1} is not a legal play on {self.P_t}")
                return False

            # Remove from hand and add to pile
            hand.remove(action.X_1)
            self.P.append(action.X_1)
        elif action.is_draw():
            # Draw n cards
            n = action.n
            cards_drawn = []

            for i in range(n):
                if len(self.D_g) == 0:
                    # Reshuffle played pile into new deck (keep top card P_t), taking all cards except the top card (P_t)
                    cards_to_shuffle = self.P[:-1]
                    self.P = [self.P[-1]]  # Keep only top card

                    # Shuffle and make new deck
                    random.shuffle(cards_to_shuffle)
                    self.D_g = cards_to_shuffle
                    print(f"Deck empty - reshuffled {len(self.D_g)} cards from pile")

                # Draw from end of deck (top)
                card = self.D_g.pop()
                cards_drawn.append(card)
                hand.append(card)

            action.Y_n = cards_drawn

        # Update state
        self.update_S()
        return True

    def get_O_space(self):
        """
        O = (H_1, |H_2|, |D_g|, P, P_t, G_o) - observation (as seen by “Player 1”)
        """
        if self.State is None:
            self.create_S()
        self.O = (self.H_1, len(self.H_2), len(self.D_g), self.P, self.P_t, self.G_o)
        return self.O

    def T(self, s_prime: State, action: Action, player: int = 1) -> float:
        """
        Transition probability function T(s' | s, a).

        Given current state s (self), action a, and next state s_prime,
        returns the probability of transitioning to s_prime.

        Args:
            s_prime: Next state (H_1', H_2', D_g', P', P_t', G_o')
            action: Action taken (X_1 or Y_n)
            player: Which player (1 or 2)

        Returns:
            Probability in [0, 1]
        """
        if self.State is None:
            self.create_S()

        # Unpack current state from self
        H_1, H_2, D_g, P, P_t, G_o = self.State

        # Unpack next state
        H_1_prime, H_2_prime, D_g_prime, P_prime, P_t_prime, G_o_prime = s_prime

        # Select current and next hand based on player
        if player == 1:
            H_curr, H_curr_prime = H_1, H_1_prime
            H_other, H_other_prime = H_2, H_2_prime
        else:
            H_curr, H_curr_prime = H_2, H_2_prime
            H_other, H_other_prime = H_1, H_1_prime

        # Case 0: Game over - no transitions allowed
        if G_o == "GameOver":
            return 0.0

        # Common checks
        if G_o_prime != "Active" and G_o_prime != "GameOver":
            return 0.0

        # Check other hand unchanged (multiset equality)
        if Counter(H_other_prime) != Counter(H_other):
            return 0.0

        # Case 1: Play action (deterministic, prob = 1 if valid)
        if action.is_play():
            X_1 = action.X_1

            # Check if X_1 is in current hand (multiset)
            H_curr_counter = Counter(H_curr)
            if H_curr_counter[X_1] == 0:
                return 0.0

            # Check legality
            if not self.is_legal_play(X_1):
                return 0.0

            # Check hand update: H' = H \ {X_1} (multiset)
            expected_h_counter = H_curr_counter - Counter([X_1])
            if Counter(H_curr_prime) != expected_h_counter:
                return 0.0

            # Check pile update: P' = P + [X_1] (exact list, assuming order matters for played pile)
            if P_prime != P + [X_1]:
                return 0.0

            # Check top card update
            if P_t_prime != X_1:
                return 0.0

            # Check deck unchanged
            if Counter(D_g_prime) != Counter(D_g):
                return 0.0

            # Check game over condition
            expected_g_o = "GameOver" if len(H_curr_prime) == 0 else "Active"
            if G_o_prime != expected_g_o:
                return 0.0

            return 1.0

        # Case 2: Draw action
        elif action.is_draw():
            n = action.n

            # Deck size check (no reshuffle handling; assume sufficient cards)
            if len(D_g) < n:
                return 0.0

            # Compute added multiset: cards added to hand
            added_counter = Counter(H_curr_prime) - Counter(H_curr)
            total_added = sum(added_counter.values())

            # Must add exactly n cards, no removals
            if total_added != n or any(v < 0 for v in added_counter.values()):
                return 0.0

            # For Y_1: must have no legal moves
            if n == 1:
                has_legal_move = any(self.is_legal_play(card) for card in H_curr)
                if has_legal_move:
                    return 0.0

            # For Y_n (n>1): in simplified number-card UNO, no +2/+4 specials exist.
            # Assuming Y_2/Y_4 invalid unless P_t value is 2 or 4 (heuristic match to math).
            # Adjust this condition based on your exact rules; here returning 0 for n>1.
            if n > 1:
                # Example condition: if P_t and P_t[1] in [2, 4] (treating numbers 2/4 as draw triggers)
                if self.P_t is None or self.P_t[1] not in [2, 4]:
                    return 0.0

            # Validate added cards were available in deck
            dg_counter = Counter(D_g)
            for t, cnt in added_counter.items():
                if cnt > dg_counter[t]:
                    return 0.0

            # Check deck update: D_g' = D_g \ added cards (multiset)
            expected_dg_counter = dg_counter - added_counter
            if Counter(D_g_prime) != expected_dg_counter:
                return 0.0

            # Check pile and top unchanged
            if P_prime != P or P_t_prime != P_t:
                return 0.0

            # Check game over (draw can't cause win)
            if G_o_prime != "Active":
                return 0.0

            # Calculate probability
            if n == 1:
                # For Y_1: added_counter has one type with count 1
                if len(added_counter) != 1:
                    return 0.0
                drawn_type = next(iter(added_counter))
                k = dg_counter[drawn_type]
                return k / len(D_g) if len(D_g) > 0 else 0.0
            else:
                # For Y_n: multinomial probability for the specific multiset
                ways = 1
                for t, cnt in added_counter.items():
                    ways *= comb(dg_counter[t], cnt)
                total_ways = comb(len(D_g), n)
                return ways / total_ways if total_ways > 0 else 0.0

        # Invalid action
        return 0.0

# Simple example usage
if __name__ == "__main__":
    # Initialize game
    uno = Uno()
    uno.new_game(seed=1)

    # Print initial state
    uno.print_S()

    # Player 1 action
    actions = uno.get_legal_actions(1)
    if actions:
        uno.execute_action(actions[0], 1)
        uno.print_S()

    # Basic T test (play)
    print("\nT Play Test:")
    uno_t = Uno(H_1=[('R', 7), ('Y', 7)], H_2=[('B', 2)], D_g=[('G', 3)], P=[('Y', 7)])
    uno_t.create_S()
    act = Action(X_1=('Y', 7))
    s_prime = ([('R', 7)], [('B', 2)], [('G', 3)], [('Y', 7), ('Y', 7)], ('Y', 7), "Active")
    print(uno_t.T(s_prime, act, 1))  # 1.0

    # Basic T test (draw sum)
    print("\nT Draw Sum:")
    uno_d = Uno(H_1=[('R', 1)], H_2=[('B', 2)], D_g=[('R', 1), ('G', 4)], P=[('Y', 5)])
    uno_d.create_S()
    act_d = Action(n=1)
    total = 0.0
    for card in [('R', 1), ('G', 4)]:
        d_prime = uno_d.D_g[:]; d_prime.remove(card)
        s_d = (uno_d.H_1 + [card], uno_d.H_2[:], d_prime, uno_d.P[:], uno_d.P_t, "Active")
        total += uno_d.T(s_d, act_d, 1)
    print(total)  # 1.0