import random
from typing import List
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
        self.State = (self.H_1, self.H_2, self.D_g, self.P, self.P_t, self.G_o)
        
        # Check game over condition: if either hand is empty
        if len(self.H_1) == 0 or len(self.H_2) == 0:
            self.G_o = "GameOver"
            
    def print_S(self):
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


# Example usage
if __name__ == "__main__":
    uno = Uno()
    uno.new_game(seed=1)

    # Create state
    uno.create_S()
    uno.print_S()
    
    # Get legal actions for player 1
    print(f"\nLegal actions for Player 1:")
    legal_actions = uno.get_legal_actions(player=1)
    for i, action in enumerate(legal_actions):
        print(f"  {i}: {action}")
    
    # Execute first legal action
    if len(legal_actions) > 0:
        print(f"\nExecuting action: {legal_actions[0]}")
        uno.execute_action(legal_actions[0], player=1)
       
        # Show updated state
        uno.update_S()
        uno.print_S()
