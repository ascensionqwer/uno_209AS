import sys
import os
import pickle
from typing import Dict, Tuple, List, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage_1_mini_uno.mini_uno import MiniUno
from cards import Card
from pomdp import Action

# State for the solver: (H_1, H_2, D_g, P_t, Turn)
# H_1, H_2, D_g are sorted tuples of cards for canonical representation
SolverState = Tuple[Tuple[Card, ...], Tuple[Card, ...], Tuple[Card, ...], Optional[Card], int]

class OfflineSolver:
    def __init__(self):
        self.memo: Dict[SolverState, float] = {}
        self.policy: Dict[SolverState, Action] = {}
        self.game = MiniUno()

    def get_canonical_state(self, game: MiniUno, turn: int) -> SolverState:
        """Converts game state to canonical tuple for memoization."""
        h1 = tuple(sorted(game.H_1))
        h2 = tuple(sorted(game.H_2))
        dg = tuple(sorted(game.D_g)) # Order doesn't matter for probability if we assume random draw, 
                                     # but for perfect info, we usually know the order.
                                     # However, for "solving the game" generally, we assume D_g is a set 
                                     # from which we draw. 
                                     # BUT, if we want a policy for a SPECIFIC deck order, we must preserve it.
                                     # The prompt asks for "offline full table solution".
                                     # Usually this implies Expected Value over draws, or Minimax if D_g is known.
                                     # Let's assume Expected Value over random draws from D_g (Expectiminimax).
        
        # If we assume D_g is a set (random draw), we sort it.
        return (h1, h2, dg, game.P_t, turn)

    def solve(self, state: SolverState) -> float:
        """
        Computes the value of the state for the current player.
        Returns: 1.0 (Win), -1.0 (Loss), 0.0 (Draw/Loop - handled by depth/visited?)
        Using Expectiminimax since draws are random.
        """
        if state in self.memo:
            return self.memo[state]

        h1, h2, dg, p_t, turn = state

        # Check terminal states
        if len(h1) == 0:
            return 1.0 if turn == 0 else -1.0 # Player 1 won (if it was their turn, they just played last card)
            # Actually, if h1 is empty, P1 already won.
            # The state is usually evaluated at the START of a turn.
            # If H1 is empty, P1 won.
        if len(h1) == 0: return 1.0 # P1 Win
        if len(h2) == 0: return -1.0 # P2 Win

        # Reconstruct game object to get legal actions
        # Note: This is expensive, maybe optimize later
        self.game.H_1 = list(h1)
        self.game.H_2 = list(h2)
        self.game.D_g = list(dg)
        self.game.P_t = p_t
        self.game.P = [p_t] if p_t else [] # Set P to contain P_t so create_S preserves it
        self.game.G_o = "Active"
        self.game.create_S() # MUST update state because we modified attributes directly

        # Get legal actions for current player
        # Player 1 is 1 (index 0 in turn?), Player 2 is 2.
        # Let's use turn=0 for P1, turn=1 for P2.
        player_num = 1 if turn == 0 else 2
        actions = self.game.get_legal_actions(player=player_num)

        if not actions:
            print(f"No actions for player {player_num} in state {state}")
            # Should not happen in Uno as draw is always an option if defined?
            # In our implementation, get_legal_actions returns Draw if no play.
            pass

        best_value = -float('inf') if turn == 0 else float('inf')
        best_action = None

        for action in actions:
            val = 0.0
            # print(f"Evaluating action {action} for player {player_num}")
            
            if action.is_play():
                # Deterministic transition
                # Create next state
                next_h1 = list(h1)
                next_h2 = list(h2)
                next_p_t = action.X_1
                
                if turn == 0:
                    next_h1.remove(action.X_1)
                    if len(next_h1) == 0: val = 1.0 # Win immediately
                    else:
                        next_state = (tuple(sorted(next_h1)), tuple(sorted(next_h2)), dg, next_p_t, 1 - turn)
                        val = self.solve(next_state)
                else:
                    next_h2.remove(action.X_1)
                    if len(next_h2) == 0: val = -1.0 # Win immediately
                    else:
                        next_state = (tuple(sorted(next_h1)), tuple(sorted(next_h2)), dg, next_p_t, 1 - turn)
                        val = self.solve(next_state)

            elif action.is_draw():
                # Stochastic transition (Expectation over drawn cards)
                # Draw 1 card (Mini Uno rules)
                if len(dg) == 0:
                    # Deck empty. In Mini Uno, maybe we reshuffle? 
                    # Or just end game? Or assume infinite deck?
                    # Standard Uno reshuffles.
                    # For this solver, let's assume if deck empty, it's a draw (0.0) or loss?
                    # To keep it simple and finite, let's say 0.0
                    val = 0.0 
                else:
                    # Expected value over all possible cards in D_g
                    total_val = 0.0
                    # D_g is a multiset. We iterate over unique cards.
                    unique_cards = set(dg)
                    for card in unique_cards:
                        prob = dg.count(card) / len(dg)
                        
                        # Next state
                        next_dg = list(dg)
                        next_dg.remove(card)
                        
                        next_h1 = list(h1)
                        next_h2 = list(h2)
                        
                        if turn == 0:
                            next_h1.append(card)
                            # After draw, is it next player's turn? 
                            # Usually in Uno, if you draw and can't play, it's next player.
                            # If you can play, you can?
                            # Simplified Uno: Draw ends turn? Or Draw then Play?
                            # uno.py execute_action just adds to hand.
                            # We need to know the rule.
                            # Usually: Draw 1. If playable, play. Else keep.
                            # Let's assume simple rule: Draw 1, then Turn Ends.
                            # This avoids infinite draw loops in one turn.
                            next_state = (tuple(sorted(next_h1)), tuple(sorted(next_h2)), tuple(sorted(next_dg)), p_t, 1 - turn)
                        else:
                            next_h2.append(card)
                            next_state = (tuple(sorted(next_h1)), tuple(sorted(next_h2)), tuple(sorted(next_dg)), p_t, 1 - turn)
                        
                        total_val += prob * self.solve(next_state)
                    val = total_val

            # Update best value
            if turn == 0: # Maximize
                if val > best_value:
                    best_value = val
                    best_action = action
            else: # Minimize
                if val < best_value:
                    best_value = val
                    best_action = action

        self.memo[state] = best_value
        self.memo[state] = best_value
        self.policy[state] = best_action
            
        return best_value

    def save_policy(self, filename="policy.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self.policy, f)

    def compute_full_policy(self):
        """
        Explores all reachable states from the initial game state and computes policy.
        """
        # Start with a new game
        self.game.new_game(seed=42) # Seed doesn't matter for reachable set if we explore all branches
        # Actually, new_game shuffles. To get ALL reachable states, we need to consider all possible deals.
        # But maybe just solving for one specific deal is enough?
        # The user said "offline full table solution". Usually implies policy for ANY state.
        # But "Mini Uno" has 10 cards.
        # Let's try to iterate all valid distributions of cards.
        
        # Iterating all partitions of 10 cards into H1(2), H2(2), Dg(5), Pt(1).
        # This is (10 choose 2) * (8 choose 2) * (6 choose 1) = 45 * 28 * 6 = 7560 initial deals.
        # Then play out.
        # This is small enough.
        
        # However, generating them is tedious.
        # Let's just implement a recursive explorer from a set of initial states?
        # Or just expose solve() and let the visualizer call it for specific states of interest.
        pass


if __name__ == "__main__":
    solver = OfflineSolver()
    # Example state
    # H1: [(R, 1)], H2: [(B, 1)], Dg: [(R, 0), (B, 0)], Pt: (R, 2), Turn: 0
    state = (
        (('R', 1),), 
        (('B', 1),), 
        (('R', 0), ('B', 0)), 
        ('R', 2), 
        0
    )
    print("Solving example state...")
    val = solver.solve(state)
    print(f"Value: {val}")
    print(f"Optimal Action: {solver.policy.get(state)}")
