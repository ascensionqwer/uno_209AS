# Implement uno game engine in Python

_Exported on 11/20/2025 at 13:33:14 PST from Cursor (2.0.77)_

---

**User**

implement a fully working uno game engine within the src/uno/ folder, such that if i run main.py it will start that simulator for 2 players. for now, it should be functioning within terminal only, and assume both players irl are trusting and take turns so you don't need to hide the full game state in the terminal

@AGENTS.md

---

**Cursor**

Reviewing the codebase to understand the current state before planning the UNO game engine.

Checking for additional context:

A few clarifications:

1. **UNO rules scope**: Should this include special cards (Skip, Reverse, Draw 2, Wild, Wild Draw 4) or only number cards (0–9)?

2. **Code organization**: Should I:
   - Create a new implementation in `src/uno/` (separate from existing `uno.py`), or
   - Refactor/move the existing code into `src/uno/`?

3. **Player interaction**: Should players:
   - Enter moves interactively (wait for input each turn), or
   - Use a simple automated turn-taking system?

---

**User**

1. yes it should include those cards

2. reorganize all files such that the uno implementation is in src/uno, the root has only 1 python file main.py, and anything else is placed in src/<subfolder> depending on what it is

3. use a simple automated turn taking system

---

**Cursor**

Reviewing MATH.md to understand the full rules, then drafting the plan.

Creating the plan:

---

**User**

Implement the plan as specified, it is attached for your reference. Do NOT edit the plan file itself.

To-do's from the plan have already been created. Do not create them again. Mark them as in_progress as you work, starting with the first one. Don't stop until you have completed all the to-dos.

---

**Cursor**

Starting implementation. Reviewing existing files, then reorganizing and extending the code.

Creating the directory structure and starting file reorganization:

```python
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

```

```python
from typing import List, Tuple, Optional
from .cards import Card

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

    def __init__(self, X_1: Optional[Card] = None, n: Optional[int] = None, wild_color: Optional[str] = None):
        """
        Initialize an action - must specify EITHER X_1 OR n, not both.

        Args:
            X_1: Card to play (COLOR, VALUE) - for PLAY action
            n: Number of cards to draw, n ∈ {1, 2, 4} - for DRAW action
            wild_color: Color chosen for Wild/Wild Draw 4 (only used when X_1 is Wild)
        """
        # Validate that exactly one is specified
        if (X_1 is None and n is None) or (X_1 is not None and n is not None):
            raise ValueError("Must specify EITHER X_1 (play) OR n (draw), not both or neither")

        self.X_1 = X_1  # Card to play (None if draw action)
        self.n = n      # Number to draw (None if play action)
        self.Y_n = []   # Cards actually drawn (filled during execution)
        self.wild_color = wild_color  # Color chosen for Wild cards

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
            if self.wild_color:
                return f"Action(X_1={self.X_1}, wild_color={self.wild_color})"
            return f"Action(X_1={self.X_1})"
        elif self.is_draw():
            if len(self.Y_n) > 0:
                return f"Action(Y_{self.n}={self.Y_n})"
            else:
                return f"Action(Y_{self.n}=[])"

```

---

**User**

remember to read @AGENTS.md for agent rules to follow

---

**Cursor**

Continuing implementation, following AGENTS.md rules. Creating the full game engine:

```python
import random
from typing import List, Optional
from collections import Counter
from math import comb

from .cards import (
    Card, RED, YELLOW, GREEN, BLUE, BLACK,
    SKIP, REVERSE, DRAW_2, WILD, WILD_DRAW_4,
    build_full_deck, card_to_string
)
from .state import State, Action


class Uno:
    """
    Full UNO game engine with all card types.
    D = H_1 ⊔ H_2 ⊔ D_g ⊔ P : entire deck partitioned into hands, draw pile, played pile.
    Builds 108-card deck with all UNO cards.
    """

    def __init__(self, H_1: List[Card] = None, H_2: List[Card] = None,
                 D_g: List[Card] = None, P: List[Card] = None):
        self.H_1 = H_1 if H_1 is not None else []
        self.H_2 = H_2 if H_2 is not None else []
        self.D_g = D_g if D_g is not None else []
        self.P = P if P is not None else []
        self.G_o = "Active"
        self.State = None
        self.P_t = None
        self.current_color = None  # For Wild cards - the chosen color
        self.turn_direction = 1  # 1 = forward (P1->P2), -1 = reverse (P2->P1)
        self.skip_next = False  # Skip next player's turn
        self.draw_pending = 0  # Cards to draw before next turn

    def build_full_deck(self) -> List[Card]:
        """Builds the full 108-card UNO deck."""
        return build_full_deck()

    def new_game(self, seed: int = None, deal: int = 7):
        """
        Initializes a new game state.
        - Shuffles the full 108-card deck.
        - Deals 'deal' cards to H_1 and H_2.
        - D_g gets the remainder.
        - P starts with the first card from D_g (must be non-Wild).
        """
        rng = random.Random(seed)
        full_deck = self.build_full_deck()
        rng.shuffle(full_deck)

        # Deal to hands
        self.H_1 = full_deck[:deal]
        self.H_2 = full_deck[deal:2 * deal]
        remaining = full_deck[2 * deal:]

        # Start played pile with first non-Wild card
        while remaining and self._is_wild(remaining[0]):
            # Reshuffle wilds back in
            rng.shuffle(remaining)

        if remaining:
            self.P = [remaining.pop(0)]
            self.current_color, _ = self.P[0]
            if self.current_color is None:
                # Shouldn't happen, but handle it
                self.current_color = RED
        else:
            # Fallback: reshuffle everything
            full_deck = self.build_full_deck()
            rng.shuffle(full_deck)
            self.H_1 = full_deck[:deal]
            self.H_2 = full_deck[deal:2 * deal]
            remaining = full_deck[2 * deal:]
            self.P = [remaining.pop(0)]
            self.current_color, _ = self.P[0]
            if self.current_color is None:
                self.current_color = RED

        self.D_g = remaining
        self.G_o = "Active"
        self.turn_direction = 1
        self.skip_next = False
        self.draw_pending = 0

        self.create_S()

    def _is_wild(self, card: Card) -> bool:
        """Check if card is Wild or Wild Draw 4."""
        _, value = card
        return value in [WILD, WILD_DRAW_4]

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
        print(f"H_1: {[card_to_string(c) for c in self.State[0]]}, {len(self.State[0])} cards")
        print(f"H_2: {[card_to_string(c) for c in self.State[1]]}, {len(self.State[1])} cards")
        print(f"D_g: {len(self.State[2])} cards")
        print(f"P: {len(self.State[3])} cards, top: {card_to_string(self.State[4]) if self.State[4] else 'None'}")
        print(f"Current color: {self.current_color}")
        print(f"G_o: {self.State[5]}")

    def is_legal_play(self, card: Card) -> bool:
        """
        Checks if a card can be legally played.
        Legal if:
        - Card color matches current color OR
        - Card value matches P_t value OR
        - Card is Wild/Wild Draw 4 (always legal)
        """
        if self.P_t is None:
            return False

        card_color, card_value = card
        P_t_color, P_t_value = self.P_t

        # Wild cards are always legal
        if self._is_wild(card):
            return True

        # Match by color (use current_color for Wild cards)
        effective_color = self.current_color if P_t_color is None else P_t_color
        if card_color == effective_color:
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

    def _choose_wild_color(self, hand: List[Card]) -> str:
        """
        Choose color for Wild card based on most common color in hand.
        Defaults to RED if no colored cards.
        """
        color_counts = {RED: 0, YELLOW: 0, GREEN: 0, BLUE: 0}
        for card in hand:
            color, _ = card
            if color in color_counts:
                color_counts[color] += 1

        max_color = max(color_counts, key=color_counts.get)
        if color_counts[max_color] > 0:
            return max_color
        return RED  # Default

    def execute_action(self, action: Action, player: int = 1) -> bool:
        """
        Executes an action and updates the game state.
        Handles special card effects.
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

            # Handle special card effects
            _, value = card

            if self._is_wild(card):
                # Wild card: choose color
                if action.wild_color:
                    self.current_color = action.wild_color
                else:
                    self.current_color = self._choose_wild_color(hand)

                if value == WILD_DRAW_4:
                    # Next player draws 4 and skips turn
                    self.draw_pending = 4
                    self.skip_next = True
            elif value == SKIP:
                # Skip next player's turn
                self.skip_next = True
            elif value == REVERSE:
                # In 2-player, Reverse acts like Skip
                self.skip_next = True
            elif value == DRAW_2:
                # Next player draws 2 and skips turn
                self.draw_pending = 2
                self.skip_next = True
            else:
                # Number card: update current color
                self.current_color, _ = card
        elif action.is_draw():
            # Draw n cards
            n = action.n

            # If draw_pending > 0, must draw that many
            if self.draw_pending > 0:
                n = self.draw_pending
                self.draw_pending = 0

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

            # If drew due to draw_pending, skip turn
            if self.draw_pending == 0 and n > 1:
                self.skip_next = True

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

```

Fixing game logic and completing the implementation:

```python
import random
from typing import List, Optional
from collections import Counter
from math import comb

from .cards import (
    Card, RED, YELLOW, GREEN, BLUE, BLACK,
    SKIP, REVERSE, DRAW_2, WILD, WILD_DRAW_4,
    build_full_deck, card_to_string
)
from .state import State, Action


class Uno:
    """
    Full UNO game engine with all card types.
    D = H_1 ⊔ H_2 ⊔ D_g ⊔ P : entire deck partitioned into hands, draw pile, played pile.
    Builds 108-card deck with all UNO cards.
    """

    def __init__(self, H_1: List[Card] = None, H_2: List[Card] = None,
                 D_g: List[Card] = None, P: List[Card] = None):
        self.H_1 = H_1 if H_1 is not None else []
        self.H_2 = H_2 if H_2 is not None else []
        self.D_g = D_g if D_g is not None else []
        self.P = P if P is not None else []
        self.G_o = "Active"
        self.State = None
        self.P_t = None
        self.current_color = None  # For Wild cards - the chosen color
        self.turn_direction = 1  # 1 = forward (P1->P2), -1 = reverse (P2->P1)
        self.skip_next = False  # Skip next player's turn
        self.draw_pending = 0  # Cards to draw before next turn

    def build_full_deck(self) -> List[Card]:
        """Builds the full 108-card UNO deck."""
        return build_full_deck()

    def new_game(self, seed: int = None, deal: int = 7):
        """
        Initializes a new game state.
        - Shuffles the full 108-card deck.
        - Deals 'deal' cards to H_1 and H_2.
        - D_g gets the remainder.
        - P starts with the first card from D_g (must be non-Wild).
        """
        rng = random.Random(seed)
        full_deck = self.build_full_deck()
        rng.shuffle(full_deck)

        # Deal to hands
        self.H_1 = full_deck[:deal]
        self.H_2 = full_deck[deal:2 * deal]
        remaining = full_deck[2 * deal:]

        # Start played pile with first non-Wild card
        while remaining and self._is_wild(remaining[0]):
            # Reshuffle wilds back in
            rng.shuffle(remaining)

        if remaining:
            self.P = [remaining.pop(0)]
            self.current_color, _ = self.P[0]
            if self.current_color is None:
                # Shouldn't happen, but handle it
                self.current_color = RED
        else:
            # Fallback: reshuffle everything
            full_deck = self.build_full_deck()
            rng.shuffle(full_deck)
            self.H_1 = full_deck[:deal]
            self.H_2 = full_deck[deal:2 * deal]
            remaining = full_deck[2 * deal:]
            self.P = [remaining.pop(0)]
            self.current_color, _ = self.P[0]
            if self.current_color is None:
                self.current_color = RED

        self.D_g = remaining
        self.G_o = "Active"
        self.turn_direction = 1
        self.skip_next = False
        self.draw_pending = 0

        self.create_S()

    def _is_wild(self, card: Card) -> bool:
        """Check if card is Wild or Wild Draw 4."""
        _, value = card
        return value in [WILD, WILD_DRAW_4]

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
        print(f"H_1: {[card_to_string(c) for c in self.State[0]]}, {len(self.State[0])} cards")
        print(f"H_2: {[card_to_string(c) for c in self.State[1]]}, {len(self.State[1])} cards")
        print(f"D_g: {len(self.State[2])} cards")
        print(f"P: {len(self.State[3])} cards, top: {card_to_string(self.State[4]) if self.State[4] else 'None'}")
        print(f"Current color: {self.current_color}")
        print(f"G_o: {self.State[5]}")

    def is_legal_play(self, card: Card) -> bool:
        """
        Checks if a card can be legally played.
        Legal if:
        - Card color matches current color OR
        - Card value matches P_t value OR
        - Card is Wild/Wild Draw 4 (always legal)
        """
        if self.P_t is None:
            return False

        card_color, card_value = card
        P_t_color, P_t_value = self.P_t

        # Wild cards are always legal
        if self._is_wild(card):
            return True

        # Match by color (use current_color for Wild cards)
        effective_color = self.current_color if P_t_color is None else P_t_color
        if card_color == effective_color:
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

    def _choose_wild_color(self, hand: List[Card]) -> str:
        """
        Choose color for Wild card based on most common color in hand.
        Defaults to RED if no colored cards.
        """
        color_counts = {RED: 0, YELLOW: 0, GREEN: 0, BLUE: 0}
        for card in hand:
            color, _ = card
            if color in color_counts:
                color_counts[color] += 1

        max_color = max(color_counts, key=color_counts.get)
        if color_counts[max_color] > 0:
            return max_color
        return RED  # Default

    def execute_action(self, action: Action, player: int = 1) -> bool:
        """
        Executes an action and updates the game state.
        Handles special card effects.
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

            # Handle special card effects
            _, value = card

            if self._is_wild(card):
                # Wild card: choose color
                if action.wild_color:
                    self.current_color = action.wild_color
                else:
                    self.current_color = self._choose_wild_color(hand)

                if value == WILD_DRAW_4:
                    # Next player draws 4 and skips turn
                    self.draw_pending = 4
                    self.skip_next = True
            elif value == SKIP:
                # Skip next player's turn
                self.skip_next = True
            elif value == REVERSE:
                # In 2-player, Reverse acts like Skip
                self.skip_next = True
            elif value == DRAW_2:
                # Next player draws 2 and skips turn
                self.draw_pending = 2
                self.skip_next = True
            else:
                # Number card: update current color
                self.current_color, _ = card
        elif action.is_draw():
            # Draw n cards
            n = action.n

            # If draw_pending > 0, must draw that many
            if self.draw_pending > 0:
                n = self.draw_pending
                self.draw_pending = 0

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


```

```python
from .game import Uno
from .cards import (
    Card, RED, YELLOW, GREEN, BLUE, BLACK,
    SKIP, REVERSE, DRAW_2, WILD, WILD_DRAW_4,
    build_full_deck, card_to_string
)
from .state import State, Action

__all__ = [
    'Uno',
    'Card', 'RED', 'YELLOW', 'GREEN', 'BLUE', 'BLACK',
    'SKIP', 'REVERSE', 'DRAW_2', 'WILD', 'WILD_DRAW_4',
    'build_full_deck', 'card_to_string',
    'State', 'Action'
]

```

```python
from src.uno import Uno, Action, card_to_string


def display_game_state(game: Uno, current_player: int):
    """Display current game state."""
    print("\n" + "=" * 60)
    print(f"Player {current_player}'s Turn")
    print("=" * 60)
    game.create_S()
    state = game.State

    print(f"\nTop Card: {card_to_string(state[4]) if state[4] else 'None'}")
    if game.current_color:
        print(f"Current Color: {game.current_color}")
    print(f"Deck: {len(state[2])} cards remaining")
    print(f"\nPlayer 1 Hand ({len(state[0])} cards):")
    for i, card in enumerate(state[0], 1):
        print(f"  {i}. {card_to_string(card)}")
    print(f"\nPlayer 2 Hand ({len(state[1])} cards):")
    for i, card in enumerate(state[1], 1):
        print(f"  {i}. {card_to_string(card)}")
    print()


def choose_action_simple(game: Uno, player: int):
    """
    Simple automated action selection:
    - Play first legal card found
    - If Wild, choose most common color in hand
    - If no legal play, draw 1
    """
    legal_actions = game.get_legal_actions(player)

    if not legal_actions:
        return None

    # Prefer playing over drawing
    play_actions = [a for a in legal_actions if a.is_play()]
    if play_actions:
        action = play_actions[0]
        # If Wild card, set color
        if action.X_1 and game._is_wild(action.X_1):
            hand = game.H_1 if player == 1 else game.H_2
            action.wild_color = game._choose_wild_color(hand)
        return action

    # Must draw
    return legal_actions[0]


def main():
    """Run 2-player UNO game simulator."""
    print("Starting UNO Game Simulator (2 Players)")
    print("=" * 60)

    game = Uno()
    game.new_game(seed=None)

    current_player = 1
    turn_count = 0
    max_turns = 1000  # Safety limit

    while game.G_o == "Active" and turn_count < max_turns:
        turn_count += 1

        # Check if we need to skip this player
        if game.skip_next:
            print(f"\n>>> Player {current_player} is skipped!")
            game.skip_next = False
            current_player = 3 - current_player  # Switch: 1->2, 2->1
            continue

        # Check if player needs to draw cards
        if game.draw_pending > 0:
            print(f"\n>>> Player {current_player} must draw {game.draw_pending} cards!")
            action = Action(n=game.draw_pending)
            game.execute_action(action, current_player)
            game.draw_pending = 0
            game.skip_next = True
            display_game_state(game, current_player)
            current_player = 3 - current_player
            continue

        display_game_state(game, current_player)

        # Choose and execute action
        action = choose_action_simple(game, current_player)
        if action is None:
            print(f"Player {current_player} has no actions available!")
            break

        if action.is_play():
            card_str = card_to_string(action.X_1)
            if action.wild_color:
                print(f"Player {current_player} plays {card_str} and chooses color {action.wild_color}")
            else:
                print(f"Player {current_player} plays {card_str}")
        else:
            print(f"Player {current_player} draws {action.n} card(s)")

        success = game.execute_action(action, current_player)
        if not success:
            print(f"Action failed for player {current_player}")
            break

        # Check game over
        if game.G_o == "GameOver":
            break

        # Switch players (unless skip_next is set)
        if not game.skip_next:
            current_player = 3 - current_player
        else:
            # Skip was set by the card played, next player will be skipped
            pass

    # Game over
    print("\n" + "=" * 60)
    print("GAME OVER")
    print("=" * 60)
    game.create_S()
    state = game.State

    if len(state[0]) == 0:
        print("Player 1 wins!")
    elif len(state[1]) == 0:
        print("Player 2 wins!")
    else:
        print(f"Game ended after {turn_count} turns")
        print(f"Player 1: {len(state[0])} cards")
        print(f"Player 2: {len(state[1])} cards")


if __name__ == "__main__":
    main()

```

Fixing turn logic and cleaning up old files:

```python
import random
from typing import List, Optional
from collections import Counter
from math import comb

from .cards import (
    Card, RED, YELLOW, GREEN, BLUE, BLACK,
    SKIP, REVERSE, DRAW_2, WILD, WILD_DRAW_4,
    build_full_deck, card_to_string
)
from .state import State, Action


class Uno:
    """
    Full UNO game engine with all card types.
    D = H_1 ⊔ H_2 ⊔ D_g ⊔ P : entire deck partitioned into hands, draw pile, played pile.
    Builds 108-card deck with all UNO cards.
    """

    def __init__(self, H_1: List[Card] = None, H_2: List[Card] = None,
                 D_g: List[Card] = None, P: List[Card] = None):
        self.H_1 = H_1 if H_1 is not None else []
        self.H_2 = H_2 if H_2 is not None else []
        self.D_g = D_g if D_g is not None else []
        self.P = P if P is not None else []
        self.G_o = "Active"
        self.State = None
        self.P_t = None
        self.current_color = None  # For Wild cards - the chosen color
        self.turn_direction = 1  # 1 = forward (P1->P2), -1 = reverse (P2->P1)
        self.skip_next = False  # Skip next player's turn
        self.draw_pending = 0  # Cards to draw before next turn

    def build_full_deck(self) -> List[Card]:
        """Builds the full 108-card UNO deck."""
        return build_full_deck()

    def new_game(self, seed: int = None, deal: int = 7):
        """
        Initializes a new game state.
        - Shuffles the full 108-card deck.
        - Deals 'deal' cards to H_1 and H_2.
        - D_g gets the remainder.
        - P starts with the first card from D_g (must be non-Wild).
        """
        rng = random.Random(seed)
        full_deck = self.build_full_deck()
        rng.shuffle(full_deck)

        # Deal to hands
        self.H_1 = full_deck[:deal]
        self.H_2 = full_deck[deal:2 * deal]
        remaining = full_deck[2 * deal:]

        # Start played pile with first non-Wild card
        while remaining and self._is_wild(remaining[0]):
            # Reshuffle wilds back in
            rng.shuffle(remaining)

        if remaining:
            self.P = [remaining.pop(0)]
            self.current_color, _ = self.P[0]
            if self.current_color is None:
                # Shouldn't happen, but handle it
                self.current_color = RED
        else:
            # Fallback: reshuffle everything
            full_deck = self.build_full_deck()
            rng.shuffle(full_deck)
            self.H_1 = full_deck[:deal]
            self.H_2 = full_deck[deal:2 * deal]
            remaining = full_deck[2 * deal:]
            self.P = [remaining.pop(0)]
            self.current_color, _ = self.P[0]
            if self.current_color is None:
                self.current_color = RED

        self.D_g = remaining
        self.G_o = "Active"
        self.turn_direction = 1
        self.skip_next = False
        self.draw_pending = 0

        self.create_S()

    def _is_wild(self, card: Card) -> bool:
        """Check if card is Wild or Wild Draw 4."""
        _, value = card
        return value in [WILD, WILD_DRAW_4]

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
        print(f"H_1: {[card_to_string(c) for c in self.State[0]]}, {len(self.State[0])} cards")
        print(f"H_2: {[card_to_string(c) for c in self.State[1]]}, {len(self.State[1])} cards")
        print(f"D_g: {len(self.State[2])} cards")
        print(f"P: {len(self.State[3])} cards, top: {card_to_string(self.State[4]) if self.State[4] else 'None'}")
        print(f"Current color: {self.current_color}")
        print(f"G_o: {self.State[5]}")

    def is_legal_play(self, card: Card) -> bool:
        """
        Checks if a card can be legally played.
        Legal if:
        - Card color matches current color OR
        - Card value matches P_t value OR
        - Card is Wild/Wild Draw 4 (always legal)
        """
        if self.P_t is None:
            return False

        card_color, card_value = card
        P_t_color, P_t_value = self.P_t

        # Wild cards are always legal
        if self._is_wild(card):
            return True

        # Match by color (use current_color for Wild cards)
        effective_color = self.current_color if P_t_color is None else P_t_color
        if card_color == effective_color:
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

    def _choose_wild_color(self, hand: List[Card]) -> str:
        """
        Choose color for Wild card based on most common color in hand.
        Defaults to RED if no colored cards.
        """
        color_counts = {RED: 0, YELLOW: 0, GREEN: 0, BLUE: 0}
        for card in hand:
            color, _ = card
            if color in color_counts:
                color_counts[color] += 1

        max_color = max(color_counts, key=color_counts.get)
        if color_counts[max_color] > 0:
            return max_color
        return RED  # Default

    def execute_action(self, action: Action, player: int = 1) -> bool:
        """
        Executes an action and updates the game state.
        Handles special card effects.
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

            # Handle special card effects
            _, value = card

            if self._is_wild(card):
                # Wild card: choose color
                if action.wild_color:
                    self.current_color = action.wild_color
                else:
                    self.current_color = self._choose_wild_color(hand)

                if value == WILD_DRAW_4:
                    # Next player draws 4 and skips turn
                    self.draw_pending = 4
            elif value == SKIP:
                # Skip next player's turn
                self.skip_next = True
            elif value == REVERSE:
                # In 2-player, Reverse acts like Skip
                self.skip_next = True
            elif value == DRAW_2:
                # Next player draws 2 and skips turn
                self.draw_pending = 2
            else:
                # Number card: update current color
                self.current_color, _ = card
        elif action.is_draw():
            # Draw n cards
            n = action.n

            # If draw_pending > 0, must draw that many
            if self.draw_pending > 0:
                n = self.draw_pending
                self.draw_pending = 0

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


```

```python
from src.uno import Uno, Action, card_to_string


def display_game_state(game: Uno, current_player: int):
    """Display current game state."""
    print("\n" + "=" * 60)
    print(f"Player {current_player}'s Turn")
    print("=" * 60)
    game.create_S()
    state = game.State

    print(f"\nTop Card: {card_to_string(state[4]) if state[4] else 'None'}")
    if game.current_color:
        print(f"Current Color: {game.current_color}")
    print(f"Deck: {len(state[2])} cards remaining")
    print(f"\nPlayer 1 Hand ({len(state[0])} cards):")
    for i, card in enumerate(state[0], 1):
        print(f"  {i}. {card_to_string(card)}")
    print(f"\nPlayer 2 Hand ({len(state[1])} cards):")
    for i, card in enumerate(state[1], 1):
        print(f"  {i}. {card_to_string(card)}")
    print()


def choose_action_simple(game: Uno, player: int):
    """
    Simple automated action selection:
    - Play first legal card found
    - If Wild, choose most common color in hand
    - If no legal play, draw 1
    """
    legal_actions = game.get_legal_actions(player)

    if not legal_actions:
        return None

    # Prefer playing over drawing
    play_actions = [a for a in legal_actions if a.is_play()]
    if play_actions:
        action = play_actions[0]
        # If Wild card, set color
        if action.X_1 and game._is_wild(action.X_1):
            hand = game.H_1 if player == 1 else game.H_2
            action.wild_color = game._choose_wild_color(hand)
        return action

    # Must draw
    return legal_actions[0]


def main():
    """Run 2-player UNO game simulator."""
    print("Starting UNO Game Simulator (2 Players)")
    print("=" * 60)

    game = Uno()
    game.new_game(seed=None)

    current_player = 1
    turn_count = 0
    max_turns = 1000  # Safety limit

    while game.G_o == "Active" and turn_count < max_turns:
        turn_count += 1

        # Check if we need to skip this player
        if game.skip_next:
            print(f"\n>>> Player {current_player} is skipped!")
            game.skip_next = False
            current_player = 3 - current_player  # Switch: 1->2, 2->1
            continue

        # Check if player needs to draw cards (from Draw 2 or Wild Draw 4)
        if game.draw_pending > 0:
            print(f"\n>>> Player {current_player} must draw {game.draw_pending} cards!")
            action = Action(n=game.draw_pending)
            game.execute_action(action, current_player)
            draw_amount = game.draw_pending
            game.draw_pending = 0
            game.skip_next = True  # Skip turn after drawing
            display_game_state(game, current_player)
            current_player = 3 - current_player
            continue

        display_game_state(game, current_player)

        # Choose and execute action
        action = choose_action_simple(game, current_player)
        if action is None:
            print(f"Player {current_player} has no actions available!")
            break

        if action.is_play():
            card_str = card_to_string(action.X_1)
            if action.wild_color:
                print(f"Player {current_player} plays {card_str} and chooses color {action.wild_color}")
            else:
                print(f"Player {current_player} plays {card_str}")
        else:
            print(f"Player {current_player} draws {action.n} card(s)")

        success = game.execute_action(action, current_player)
        if not success:
            print(f"Action failed for player {current_player}")
            break

        # Check game over
        if game.G_o == "GameOver":
            break

        # Switch players (unless skip_next is set)
        if not game.skip_next:
            current_player = 3 - current_player
        else:
            # Skip was set by the card played, next player will be skipped
            pass

    # Game over
    print("\n" + "=" * 60)
    print("GAME OVER")
    print("=" * 60)
    game.create_S()
    state = game.State

    if len(state[0]) == 0:
        print("Player 1 wins!")
    elif len(state[1]) == 0:
        print("Player 2 wins!")
    else:
        print(f"Game ended after {turn_count} turns")
        print(f"Player 1: {len(state[0])} cards")
        print(f"Player 2: {len(state[1])} cards")


if __name__ == "__main__":
    main()

```

```python
from src.uno import Uno, Action, card_to_string


def display_game_state(game: Uno, current_player: int):
    """Display current game state."""
    print("\n" + "=" * 60)
    print(f"Player {current_player}'s Turn")
    print("=" * 60)
    game.create_S()
    state = game.State

    print(f"\nTop Card: {card_to_string(state[4]) if state[4] else 'None'}")
    if game.current_color:
        print(f"Current Color: {game.current_color}")
    print(f"Deck: {len(state[2])} cards remaining")
    print(f"\nPlayer 1 Hand ({len(state[0])} cards):")
    for i, card in enumerate(state[0], 1):
        print(f"  {i}. {card_to_string(card)}")
    print(f"\nPlayer 2 Hand ({len(state[1])} cards):")
    for i, card in enumerate(state[1], 1):
        print(f"  {i}. {card_to_string(card)}")
    print()


def choose_action_simple(game: Uno, player: int):
    """
    Simple automated action selection:
    - Play first legal card found
    - If Wild, choose most common color in hand
    - If no legal play, draw 1
    """
    legal_actions = game.get_legal_actions(player)

    if not legal_actions:
        return None

    # Prefer playing over drawing
    play_actions = [a for a in legal_actions if a.is_play()]
    if play_actions:
        action = play_actions[0]
        # If Wild card, set color
        if action.X_1 and game._is_wild(action.X_1):
            hand = game.H_1 if player == 1 else game.H_2
            action.wild_color = game._choose_wild_color(hand)
        return action

    # Must draw
    return legal_actions[0]


def main():
    """Run 2-player UNO game simulator."""
    print("Starting UNO Game Simulator (2 Players)")
    print("=" * 60)

    game = Uno()
    game.new_game(seed=None)

    current_player = 1
    turn_count = 0
    max_turns = 1000  # Safety limit

    while game.G_o == "Active" and turn_count < max_turns:
        turn_count += 1

        # Check if we need to skip this player
        if game.skip_next:
            print(f"\n>>> Player {current_player} is skipped!")
            game.skip_next = False
            current_player = 3 - current_player  # Switch: 1->2, 2->1
            continue

        # Check if player needs to draw cards (from Draw 2 or Wild Draw 4)
        if game.draw_pending > 0:
            print(f"\n>>> Player {current_player} must draw {game.draw_pending} cards!")
            action = Action(n=game.draw_pending)
            game.execute_action(action, current_player)
            draw_amount = game.draw_pending
            game.draw_pending = 0
            game.skip_next = True  # Skip turn after drawing
            display_game_state(game, current_player)
            current_player = 3 - current_player
            continue

        display_game_state(game, current_player)

        # Choose and execute action
        action = choose_action_simple(game, current_player)
        if action is None:
            print(f"Player {current_player} has no actions available!")
            break

        if action.is_play():
            card_str = card_to_string(action.X_1)
            if action.wild_color:
                print(f"Player {current_player} plays {card_str} and chooses color {action.wild_color}")
            else:
                print(f"Player {current_player} plays {card_str}")
        else:
            print(f"Player {current_player} draws {action.n} card(s)")

        success = game.execute_action(action, current_player)
        if not success:
            print(f"Action failed for player {current_player}")
            break

        # Check game over
        if game.G_o == "GameOver":
            break

        # Switch players
        # If skip_next is True, it will be handled at start of next iteration
        if not game.skip_next:
            current_player = 3 - current_player

    # Game over
    print("\n" + "=" * 60)
    print("GAME OVER")
    print("=" * 60)
    game.create_S()
    state = game.State

    if len(state[0]) == 0:
        print("Player 1 wins!")
    elif len(state[1]) == 0:
        print("Player 2 wins!")
    else:
        print(f"Game ended after {turn_count} turns")
        print(f"Player 1: {len(state[0])} cards")
        print(f"Player 2: {len(state[1])} cards")


if __name__ == "__main__":
    main()

```

Fixing the draw_pending and skip logic:

```python
                if value == WILD_DRAW_4:
                    # Next player draws 4 and skips turn
                    self.draw_pending = 4
            elif value == SKIP:
                # Skip next player's turn
                self.skip_next = True
            elif value == REVERSE:
                # In 2-player, Reverse acts like Skip
                self.skip_next = True
            elif value == DRAW_2:
                # Next player draws 2 and skips turn
                self.draw_pending = 2
```

```python
from src.uno import Uno, Action, card_to_string


def display_game_state(game: Uno, current_player: int):
    """Display current game state."""
    print("\n" + "=" * 60)
    print(f"Player {current_player}'s Turn")
    print("=" * 60)
    game.create_S()
    state = game.State

    print(f"\nTop Card: {card_to_string(state[4]) if state[4] else 'None'}")
    if game.current_color:
        print(f"Current Color: {game.current_color}")
    print(f"Deck: {len(state[2])} cards remaining")
    print(f"\nPlayer 1 Hand ({len(state[0])} cards):")
    for i, card in enumerate(state[0], 1):
        print(f"  {i}. {card_to_string(card)}")
    print(f"\nPlayer 2 Hand ({len(state[1])} cards):")
    for i, card in enumerate(state[1], 1):
        print(f"  {i}. {card_to_string(card)}")
    print()


def choose_action_simple(game: Uno, player: int):
    """
    Simple automated action selection:
    - Play first legal card found
    - If Wild, choose most common color in hand
    - If no legal play, draw 1
    """
    legal_actions = game.get_legal_actions(player)

    if not legal_actions:
        return None

    # Prefer playing over drawing
    play_actions = [a for a in legal_actions if a.is_play()]
    if play_actions:
        action = play_actions[0]
        # If Wild card, set color
        if action.X_1 and game._is_wild(action.X_1):
            hand = game.H_1 if player == 1 else game.H_2
            action.wild_color = game._choose_wild_color(hand)
        return action

    # Must draw
    return legal_actions[0]


def main():
    """Run 2-player UNO game simulator."""
    print("Starting UNO Game Simulator (2 Players)")
    print("=" * 60)

    game = Uno()
    game.new_game(seed=None)

    current_player = 1
    turn_count = 0
    max_turns = 1000  # Safety limit

    while game.G_o == "Active" and turn_count < max_turns:
        turn_count += 1

        # Check if we need to skip this player
        if game.skip_next:
            print(f"\n>>> Player {current_player} is skipped!")
            game.skip_next = False
            current_player = 3 - current_player  # Switch: 1->2, 2->1
            continue

        # Check if player needs to draw cards (from Draw 2 or Wild Draw 4)
        if game.draw_pending > 0:
            print(f"\n>>> Player {current_player} must draw {game.draw_pending} cards!")
            action = Action(n=game.draw_pending)
            game.execute_action(action, current_player)
            # After drawing, skip this player's turn
            game.skip_next = True
            display_game_state(game, current_player)
            current_player = 3 - current_player
            continue

        display_game_state(game, current_player)

        # Choose and execute action
        action = choose_action_simple(game, current_player)
        if action is None:
            print(f"Player {current_player} has no actions available!")
            break

        if action.is_play():
            card_str = card_to_string(action.X_1)
            if action.wild_color:
                print(f"Player {current_player} plays {card_str} and chooses color {action.wild_color}")
            else:
                print(f"Player {current_player} plays {card_str}")
        else:
            print(f"Player {current_player} draws {action.n} card(s)")

        success = game.execute_action(action, current_player)
        if not success:
            print(f"Action failed for player {current_player}")
            break

        # Check game over
        if game.G_o == "GameOver":
            break

        # Switch players
        # If skip_next is True, it will be handled at start of next iteration
        if not game.skip_next:
            current_player = 3 - current_player

    # Game over
    print("\n" + "=" * 60)
    print("GAME OVER")
    print("=" * 60)
    game.create_S()
    state = game.State

    if len(state[0]) == 0:
        print("Player 1 wins!")
    elif len(state[1]) == 0:
        print("Player 2 wins!")
    else:
        print(f"Game ended after {turn_count} turns")
        print(f"Player 1: {len(state[0])} cards")
        print(f"Player 2: {len(state[1])} cards")


if __name__ == "__main__":
    main()

```

Verifying the implementation is complete:

Verifying imports and running a quick syntax check:

Implementation complete. Summary:

## Completed Implementation

1. **File reorganization**: Moved all code to `src/uno/`:
   - `cards.py` → `src/uno/cards.py` (extended with all card types)
   - `uno.py` → `src/uno/game.py` (refactored with full UNO rules)
   - `pomdp.py` → `src/uno/state.py` (moved and updated)
   - Old root files deleted

2. **Full UNO deck**: 108 cards including:
   - Number cards (0-9) in 4 colors
   - Skip, Reverse, Draw 2 cards
   - Wild and Wild Draw 4 cards

3. **Game engine**: Handles all special card effects:
   - Skip, Reverse (acts as Skip in 2-player)
   - Draw 2 and Wild Draw 4
   - Wild color selection
   - Turn management and game over detection

4. **Terminal simulator**: `main.py` runs a 2-player game with:
   - Automated turn-taking
   - Full game state display (both hands visible)
   - Simple strategy: play first legal card, draw if none available
   - Handles all special card effects automatically

The game is ready to run with `python3 main.py`. All imports work and the code compiles without errors.

---

**User**

skip cards are implemented wrong, if a player plays a skip, then the player gets another turn immediately after (same with reverse since same logic as skip with 2 players)

---

**Cursor**

Fixing Skip/Reverse: the player who plays it gets another turn immediately (no player switch).

```python
import random
from typing import List, Optional
from collections import Counter
from math import comb

from .cards import (
    Card, RED, YELLOW, GREEN, BLUE, BLACK,
    SKIP, REVERSE, DRAW_2, WILD, WILD_DRAW_4,
    build_full_deck, card_to_string
)
from .state import State, Action


class Uno:
    """
    Full UNO game engine with all card types.
    D = H_1 ⊔ H_2 ⊔ D_g ⊔ P : entire deck partitioned into hands, draw pile, played pile.
    Builds 108-card deck with all UNO cards.
    """

    def __init__(self, H_1: List[Card] = None, H_2: List[Card] = None,
                 D_g: List[Card] = None, P: List[Card] = None):
        self.H_1 = H_1 if H_1 is not None else []
        self.H_2 = H_2 if H_2 is not None else []
        self.D_g = D_g if D_g is not None else []
        self.P = P if P is not None else []
        self.G_o = "Active"
        self.State = None
        self.P_t = None
        self.current_color = None  # For Wild cards - the chosen color
        self.turn_direction = 1  # 1 = forward (P1->P2), -1 = reverse (P2->P1)
        self.skip_next = False  # Skip next player's turn
        self.draw_pending = 0  # Cards to draw before next turn
        self.player_plays_again = False  # Current player gets another turn

    def build_full_deck(self) -> List[Card]:
        """Builds the full 108-card UNO deck."""
        return build_full_deck()

    def new_game(self, seed: int = None, deal: int = 7):
        """
        Initializes a new game state.
        - Shuffles the full 108-card deck.
        - Deals 'deal' cards to H_1 and H_2.
        - D_g gets the remainder.
        - P starts with the first card from D_g (must be non-Wild).
        """
        rng = random.Random(seed)
        full_deck = self.build_full_deck()
        rng.shuffle(full_deck)

        # Deal to hands
        self.H_1 = full_deck[:deal]
        self.H_2 = full_deck[deal:2 * deal]
        remaining = full_deck[2 * deal:]

        # Start played pile with first non-Wild card
        while remaining and self._is_wild(remaining[0]):
            # Reshuffle wilds back in
            rng.shuffle(remaining)

        if remaining:
            self.P = [remaining.pop(0)]
            self.current_color, _ = self.P[0]
            if self.current_color is None:
                # Shouldn't happen, but handle it
                self.current_color = RED
        else:
            # Fallback: reshuffle everything
            full_deck = self.build_full_deck()
            rng.shuffle(full_deck)
            self.H_1 = full_deck[:deal]
            self.H_2 = full_deck[deal:2 * deal]
            remaining = full_deck[2 * deal:]
            self.P = [remaining.pop(0)]
            self.current_color, _ = self.P[0]
            if self.current_color is None:
                self.current_color = RED

        self.D_g = remaining
        self.G_o = "Active"
        self.turn_direction = 1
        self.skip_next = False
        self.draw_pending = 0

        self.create_S()

    def _is_wild(self, card: Card) -> bool:
        """Check if card is Wild or Wild Draw 4."""
        _, value = card
        return value in [WILD, WILD_DRAW_4]

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
        print(f"H_1: {[card_to_string(c) for c in self.State[0]]}, {len(self.State[0])} cards")
        print(f"H_2: {[card_to_string(c) for c in self.State[1]]}, {len(self.State[1])} cards")
        print(f"D_g: {len(self.State[2])} cards")
        print(f"P: {len(self.State[3])} cards, top: {card_to_string(self.State[4]) if self.State[4] else 'None'}")
        print(f"Current color: {self.current_color}")
        print(f"G_o: {self.State[5]}")

    def is_legal_play(self, card: Card) -> bool:
        """
        Checks if a card can be legally played.
        Legal if:
        - Card color matches current color OR
        - Card value matches P_t value OR
        - Card is Wild/Wild Draw 4 (always legal)
        """
        if self.P_t is None:
            return False

        card_color, card_value = card
        P_t_color, P_t_value = self.P_t

        # Wild cards are always legal
        if self._is_wild(card):
            return True

        # Match by color (use current_color for Wild cards)
        effective_color = self.current_color if P_t_color is None else P_t_color
        if card_color == effective_color:
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

    def _choose_wild_color(self, hand: List[Card]) -> str:
        """
        Choose color for Wild card based on most common color in hand.
        Defaults to RED if no colored cards.
        """
        color_counts = {RED: 0, YELLOW: 0, GREEN: 0, BLUE: 0}
        for card in hand:
            color, _ = card
            if color in color_counts:
                color_counts[color] += 1

        max_color = max(color_counts, key=color_counts.get)
        if color_counts[max_color] > 0:
            return max_color
        return RED  # Default

    def execute_action(self, action: Action, player: int = 1) -> bool:
        """
        Executes an action and updates the game state.
        Handles special card effects.
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

            # Handle special card effects
            _, value = card

            if self._is_wild(card):
                # Wild card: choose color
                if action.wild_color:
                    self.current_color = action.wild_color
                else:
                    self.current_color = self._choose_wild_color(hand)

                if value == WILD_DRAW_4:
                    # Next player draws 4 and skips turn
                    self.draw_pending = 4
            elif value == SKIP:
                # Skip next player's turn
                self.skip_next = True
            elif value == REVERSE:
                # In 2-player, Reverse acts like Skip
                self.skip_next = True
            elif value == DRAW_2:
                # Next player draws 2 and skips turn
                self.draw_pending = 2
            else:
                # Number card: update current color
                self.current_color, _ = card
        elif action.is_draw():
            # Draw n cards
            n = action.n

            # If draw_pending > 0, must draw that many
            if self.draw_pending > 0:
                n = self.draw_pending
                self.draw_pending = 0

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


```

```python
import random
from typing import List, Optional
from collections import Counter
from math import comb

from .cards import (
    Card, RED, YELLOW, GREEN, BLUE, BLACK,
    SKIP, REVERSE, DRAW_2, WILD, WILD_DRAW_4,
    build_full_deck, card_to_string
)
from .state import State, Action


class Uno:
    """
    Full UNO game engine with all card types.
    D = H_1 ⊔ H_2 ⊔ D_g ⊔ P : entire deck partitioned into hands, draw pile, played pile.
    Builds 108-card deck with all UNO cards.
    """

    def __init__(self, H_1: List[Card] = None, H_2: List[Card] = None,
                 D_g: List[Card] = None, P: List[Card] = None):
        self.H_1 = H_1 if H_1 is not None else []
        self.H_2 = H_2 if H_2 is not None else []
        self.D_g = D_g if D_g is not None else []
        self.P = P if P is not None else []
        self.G_o = "Active"
        self.State = None
        self.P_t = None
        self.current_color = None  # For Wild cards - the chosen color
        self.turn_direction = 1  # 1 = forward (P1->P2), -1 = reverse (P2->P1)
        self.skip_next = False  # Skip next player's turn
        self.draw_pending = 0  # Cards to draw before next turn
        self.player_plays_again = False  # Current player gets another turn

    def build_full_deck(self) -> List[Card]:
        """Builds the full 108-card UNO deck."""
        return build_full_deck()

    def new_game(self, seed: int = None, deal: int = 7):
        """
        Initializes a new game state.
        - Shuffles the full 108-card deck.
        - Deals 'deal' cards to H_1 and H_2.
        - D_g gets the remainder.
        - P starts with the first card from D_g (must be non-Wild).
        """
        rng = random.Random(seed)
        full_deck = self.build_full_deck()
        rng.shuffle(full_deck)

        # Deal to hands
        self.H_1 = full_deck[:deal]
        self.H_2 = full_deck[deal:2 * deal]
        remaining = full_deck[2 * deal:]

        # Start played pile with first non-Wild card
        while remaining and self._is_wild(remaining[0]):
            # Reshuffle wilds back in
            rng.shuffle(remaining)

        if remaining:
            self.P = [remaining.pop(0)]
            self.current_color, _ = self.P[0]
            if self.current_color is None:
                # Shouldn't happen, but handle it
                self.current_color = RED
        else:
            # Fallback: reshuffle everything
            full_deck = self.build_full_deck()
            rng.shuffle(full_deck)
            self.H_1 = full_deck[:deal]
            self.H_2 = full_deck[deal:2 * deal]
            remaining = full_deck[2 * deal:]
            self.P = [remaining.pop(0)]
            self.current_color, _ = self.P[0]
            if self.current_color is None:
                self.current_color = RED

        self.D_g = remaining
        self.G_o = "Active"
        self.turn_direction = 1
        self.skip_next = False
        self.draw_pending = 0
        self.player_plays_again = False

        self.create_S()

    def _is_wild(self, card: Card) -> bool:
        """Check if card is Wild or Wild Draw 4."""
        _, value = card
        return value in [WILD, WILD_DRAW_4]

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
        print(f"H_1: {[card_to_string(c) for c in self.State[0]]}, {len(self.State[0])} cards")
        print(f"H_2: {[card_to_string(c) for c in self.State[1]]}, {len(self.State[1])} cards")
        print(f"D_g: {len(self.State[2])} cards")
        print(f"P: {len(self.State[3])} cards, top: {card_to_string(self.State[4]) if self.State[4] else 'None'}")
        print(f"Current color: {self.current_color}")
        print(f"G_o: {self.State[5]}")

    def is_legal_play(self, card: Card) -> bool:
        """
        Checks if a card can be legally played.
        Legal if:
        - Card color matches current color OR
        - Card value matches P_t value OR
        - Card is Wild/Wild Draw 4 (always legal)
        """
        if self.P_t is None:
            return False

        card_color, card_value = card
        P_t_color, P_t_value = self.P_t

        # Wild cards are always legal
        if self._is_wild(card):
            return True

        # Match by color (use current_color for Wild cards)
        effective_color = self.current_color if P_t_color is None else P_t_color
        if card_color == effective_color:
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

    def _choose_wild_color(self, hand: List[Card]) -> str:
        """
        Choose color for Wild card based on most common color in hand.
        Defaults to RED if no colored cards.
        """
        color_counts = {RED: 0, YELLOW: 0, GREEN: 0, BLUE: 0}
        for card in hand:
            color, _ = card
            if color in color_counts:
                color_counts[color] += 1

        max_color = max(color_counts, key=color_counts.get)
        if color_counts[max_color] > 0:
            return max_color
        return RED  # Default

    def execute_action(self, action: Action, player: int = 1) -> bool:
        """
        Executes an action and updates the game state.
        Handles special card effects.
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

            # Handle special card effects
            _, value = card

            if self._is_wild(card):
                # Wild card: choose color
                if action.wild_color:
                    self.current_color = action.wild_color
                else:
                    self.current_color = self._choose_wild_color(hand)

                if value == WILD_DRAW_4:
                    # Next player draws 4 and skips turn
                    self.draw_pending = 4
            elif value == SKIP:
                # Skip next player's turn
                self.skip_next = True
            elif value == REVERSE:
                # In 2-player, Reverse acts like Skip
                self.skip_next = True
            elif value == DRAW_2:
                # Next player draws 2 and skips turn
                self.draw_pending = 2
            else:
                # Number card: update current color
                self.current_color, _ = card
        elif action.is_draw():
            # Draw n cards
            n = action.n

            # If draw_pending > 0, must draw that many
            if self.draw_pending > 0:
                n = self.draw_pending
                self.draw_pending = 0

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


```

```python
import random
from typing import List, Optional
from collections import Counter
from math import comb

from .cards import (
    Card, RED, YELLOW, GREEN, BLUE, BLACK,
    SKIP, REVERSE, DRAW_2, WILD, WILD_DRAW_4,
    build_full_deck, card_to_string
)
from .state import State, Action


class Uno:
    """
    Full UNO game engine with all card types.
    D = H_1 ⊔ H_2 ⊔ D_g ⊔ P : entire deck partitioned into hands, draw pile, played pile.
    Builds 108-card deck with all UNO cards.
    """

    def __init__(self, H_1: List[Card] = None, H_2: List[Card] = None,
                 D_g: List[Card] = None, P: List[Card] = None):
        self.H_1 = H_1 if H_1 is not None else []
        self.H_2 = H_2 if H_2 is not None else []
        self.D_g = D_g if D_g is not None else []
        self.P = P if P is not None else []
        self.G_o = "Active"
        self.State = None
        self.P_t = None
        self.current_color = None  # For Wild cards - the chosen color
        self.turn_direction = 1  # 1 = forward (P1->P2), -1 = reverse (P2->P1)
        self.skip_next = False  # Skip next player's turn
        self.draw_pending = 0  # Cards to draw before next turn
        self.player_plays_again = False  # Current player gets another turn

    def build_full_deck(self) -> List[Card]:
        """Builds the full 108-card UNO deck."""
        return build_full_deck()

    def new_game(self, seed: int = None, deal: int = 7):
        """
        Initializes a new game state.
        - Shuffles the full 108-card deck.
        - Deals 'deal' cards to H_1 and H_2.
        - D_g gets the remainder.
        - P starts with the first card from D_g (must be non-Wild).
        """
        rng = random.Random(seed)
        full_deck = self.build_full_deck()
        rng.shuffle(full_deck)

        # Deal to hands
        self.H_1 = full_deck[:deal]
        self.H_2 = full_deck[deal:2 * deal]
        remaining = full_deck[2 * deal:]

        # Start played pile with first non-Wild card
        while remaining and self._is_wild(remaining[0]):
            # Reshuffle wilds back in
            rng.shuffle(remaining)

        if remaining:
            self.P = [remaining.pop(0)]
            self.current_color, _ = self.P[0]
            if self.current_color is None:
                # Shouldn't happen, but handle it
                self.current_color = RED
        else:
            # Fallback: reshuffle everything
            full_deck = self.build_full_deck()
            rng.shuffle(full_deck)
            self.H_1 = full_deck[:deal]
            self.H_2 = full_deck[deal:2 * deal]
            remaining = full_deck[2 * deal:]
            self.P = [remaining.pop(0)]
            self.current_color, _ = self.P[0]
            if self.current_color is None:
                self.current_color = RED

        self.D_g = remaining
        self.G_o = "Active"
        self.turn_direction = 1
        self.skip_next = False
        self.draw_pending = 0
        self.player_plays_again = False

        self.create_S()

    def _is_wild(self, card: Card) -> bool:
        """Check if card is Wild or Wild Draw 4."""
        _, value = card
        return value in [WILD, WILD_DRAW_4]

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
        print(f"H_1: {[card_to_string(c) for c in self.State[0]]}, {len(self.State[0])} cards")
        print(f"H_2: {[card_to_string(c) for c in self.State[1]]}, {len(self.State[1])} cards")
        print(f"D_g: {len(self.State[2])} cards")
        print(f"P: {len(self.State[3])} cards, top: {card_to_string(self.State[4]) if self.State[4] else 'None'}")
        print(f"Current color: {self.current_color}")
        print(f"G_o: {self.State[5]}")

    def is_legal_play(self, card: Card) -> bool:
        """
        Checks if a card can be legally played.
        Legal if:
        - Card color matches current color OR
        - Card value matches P_t value OR
        - Card is Wild/Wild Draw 4 (always legal)
        """
        if self.P_t is None:
            return False

        card_color, card_value = card
        P_t_color, P_t_value = self.P_t

        # Wild cards are always legal
        if self._is_wild(card):
            return True

        # Match by color (use current_color for Wild cards)
        effective_color = self.current_color if P_t_color is None else P_t_color
        if card_color == effective_color:
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

    def _choose_wild_color(self, hand: List[Card]) -> str:
        """
        Choose color for Wild card based on most common color in hand.
        Defaults to RED if no colored cards.
        """
        color_counts = {RED: 0, YELLOW: 0, GREEN: 0, BLUE: 0}
        for card in hand:
            color, _ = card
            if color in color_counts:
                color_counts[color] += 1

        max_color = max(color_counts, key=color_counts.get)
        if color_counts[max_color] > 0:
            return max_color
        return RED  # Default

    def execute_action(self, action: Action, player: int = 1) -> bool:
        """
        Executes an action and updates the game state.
        Handles special card effects.
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

            # Handle special card effects
            _, value = card

            if self._is_wild(card):
                # Wild card: choose color
                if action.wild_color:
                    self.current_color = action.wild_color
                else:
                    self.current_color = self._choose_wild_color(hand)

                if value == WILD_DRAW_4:
                    # Next player draws 4 and skips turn
                    self.draw_pending = 4
            elif value == SKIP:
                # Player who played Skip gets another turn
                self.player_plays_again = True
            elif value == REVERSE:
                # In 2-player, Reverse acts like Skip - player gets another turn
                self.player_plays_again = True
            elif value == DRAW_2:
                # Next player draws 2 and skips turn
                self.draw_pending = 2
            else:
                # Number card: update current color
                self.current_color, _ = card
        elif action.is_draw():
            # Draw n cards
            n = action.n

            # If draw_pending > 0, must draw that many
            if self.draw_pending > 0:
                n = self.draw_pending
                self.draw_pending = 0

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


```

```python
from src.uno import Uno, Action, card_to_string


def display_game_state(game: Uno, current_player: int):
    """Display current game state."""
    print("\n" + "=" * 60)
    print(f"Player {current_player}'s Turn")
    print("=" * 60)
    game.create_S()
    state = game.State

    print(f"\nTop Card: {card_to_string(state[4]) if state[4] else 'None'}")
    if game.current_color:
        print(f"Current Color: {game.current_color}")
    print(f"Deck: {len(state[2])} cards remaining")
    print(f"\nPlayer 1 Hand ({len(state[0])} cards):")
    for i, card in enumerate(state[0], 1):
        print(f"  {i}. {card_to_string(card)}")
    print(f"\nPlayer 2 Hand ({len(state[1])} cards):")
    for i, card in enumerate(state[1], 1):
        print(f"  {i}. {card_to_string(card)}")
    print()


def choose_action_simple(game: Uno, player: int):
    """
    Simple automated action selection:
    - Play first legal card found
    - If Wild, choose most common color in hand
    - If no legal play, draw 1
    """
    legal_actions = game.get_legal_actions(player)

    if not legal_actions:
        return None

    # Prefer playing over drawing
    play_actions = [a for a in legal_actions if a.is_play()]
    if play_actions:
        action = play_actions[0]
        # If Wild card, set color
        if action.X_1 and game._is_wild(action.X_1):
            hand = game.H_1 if player == 1 else game.H_2
            action.wild_color = game._choose_wild_color(hand)
        return action

    # Must draw
    return legal_actions[0]


def main():
    """Run 2-player UNO game simulator."""
    print("Starting UNO Game Simulator (2 Players)")
    print("=" * 60)

    game = Uno()
    game.new_game(seed=None)

    current_player = 1
    turn_count = 0
    max_turns = 1000  # Safety limit

    while game.G_o == "Active" and turn_count < max_turns:
        turn_count += 1

        # Check if we need to skip this player
        if game.skip_next:
            print(f"\n>>> Player {current_player} is skipped!")
            game.skip_next = False
            current_player = 3 - current_player  # Switch: 1->2, 2->1
            continue

        # Check if player needs to draw cards (from Draw 2 or Wild Draw 4)
        if game.draw_pending > 0:
            print(f"\n>>> Player {current_player} must draw {game.draw_pending} cards!")
            action = Action(n=game.draw_pending)
            game.execute_action(action, current_player)
            # After drawing, skip this player's turn
            game.skip_next = True
            display_game_state(game, current_player)
            current_player = 3 - current_player
            continue

        display_game_state(game, current_player)

        # Choose and execute action
        action = choose_action_simple(game, current_player)
        if action is None:
            print(f"Player {current_player} has no actions available!")
            break

        if action.is_play():
            card_str = card_to_string(action.X_1)
            if action.wild_color:
                print(f"Player {current_player} plays {card_str} and chooses color {action.wild_color}")
            else:
                print(f"Player {current_player} plays {card_str}")
        else:
            print(f"Player {current_player} draws {action.n} card(s)")

        success = game.execute_action(action, current_player)
        if not success:
            print(f"Action failed for player {current_player}")
            break

        # Check game over
        if game.G_o == "GameOver":
            break

        # Switch players (unless current player plays again)
        if not game.player_plays_again and not game.skip_next:
            current_player = 3 - current_player
        elif game.player_plays_again:
            # Reset flag - player will continue their turn
            game.player_plays_again = False

    # Game over
    print("\n" + "=" * 60)
    print("GAME OVER")
    print("=" * 60)
    game.create_S()
    state = game.State

    if len(state[0]) == 0:
        print("Player 1 wins!")
    elif len(state[1]) == 0:
        print("Player 2 wins!")
    else:
        print(f"Game ended after {turn_count} turns")
        print(f"Player 1: {len(state[0])} cards")
        print(f"Player 2: {len(state[1])} cards")


if __name__ == "__main__":
    main()

```

Fixed Skip and Reverse: when a player plays Skip or Reverse, they get another turn immediately instead of skipping the next player. Updated the game logic to use a `player_plays_again` flag that prevents switching players after these cards.

---

**User**

how are the moves decided currently?

---

**Cursor**

Checking how moves are decided:

Moves are decided by `choose_action_simple` in `main.py`:

```25:48:main.py
def choose_action_simple(game: Uno, player: int):
    """
    Simple automated action selection:
    - Play first legal card found
    - If Wild, choose most common color in hand
    - If no legal play, draw 1
    """
    legal_actions = game.get_legal_actions(player)

    if not legal_actions:
        return None

    # Prefer playing over drawing
    play_actions = [a for a in legal_actions if a.is_play()]
    if play_actions:
        action = play_actions[0]
        # If Wild card, set color
        if action.X_1 and game._is_wild(action.X_1):
            hand = game.H_1 if player == 1 else game.H_2
            action.wild_color = game._choose_wild_color(hand)
        return action

    # Must draw
    return legal_actions[0]
```

Strategy:

1. Play the first legal card found in the hand (first match from `get_legal_actions`).
2. For Wild/Wild Draw 4, choose the most common color in the player's hand.
3. If no legal play, draw 1 card.

No prioritization (e.g., special cards, matching colors, or hand size). It just plays the first legal card or draws.

```python
def choose_action_simple(game: Uno, player: int):
    """
    Simple automated action selection:
    - Play first legal card found
    - If Wild, choose most common color in hand
    - If no legal play, draw 1
    """
    legal_actions = game.get_legal_actions(player)

    if not legal_actions:
        return None

    # Prefer playing over drawing
    play_actions = [a for a in legal_actions if a.is_play()]
    if play_actions:
        action = play_actions[0]
        # If Wild card, set color
        if action.X_1 and game._is_wild(action.X_1):
            hand = game.H_1 if player == 1 else game.H_2
            action.wild_color = game._choose_wild_color(hand)
        return action

    # Must draw
    return legal_actions[0]
```
