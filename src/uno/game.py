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

