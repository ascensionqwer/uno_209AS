import sys
import os
from typing import List, Tuple, Dict, Optional
from collections import Counter
import itertools

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uno import Uno
from cards import Card
from pomdp import Action
from stage_1_mini_uno.offline_solver import OfflineSolver
from stage_1_mini_uno.flexible_uno import FlexibleUno

class ExactBeliefSolver:
    """
    Exhaustive Limited Information Solver.
    Enumerates ALL consistent hidden states and computes Expected Value using Oracle.
    """
    def __init__(self, oracle: OfflineSolver):
        self.oracle = oracle
        self.game = None
        self.belief_states = [] # List of (State, Probability)

    def init_belief(self, game: Uno):
        """
        Initialize belief by enumerating all consistent states.
        O = (H_1, |H_2|, |D_g|, P, P_t, G_o)
        """
        self.game = game
        observation = game.get_O_space()
        H_1, len_H2, len_Dg, P, P_t, G_o = observation
        
        # 1. Determine Unknown Cards L = D_total \ (H_1 + P)
        # We need the full deck composition.
        # If game is FlexibleUno, we can get it.
        if isinstance(game, FlexibleUno):
            full_deck = game.build_number_deck()
        else:
            # Assume MiniUno or Standard Uno based on class?
            # Or just use the game's method if available
            full_deck = game.build_number_deck()
            
        # Count knowns
        known_cards = list(H_1) + list(P)
        known_counter = Counter(known_cards)
        full_counter = Counter(full_deck)
        
        L = []
        for card, count in full_counter.items():
            unknown_count = count - known_counter.get(card, 0)
            if unknown_count < 0:
                raise ValueError(f"Inconsistent state: More {card} observed than in deck")
            L.extend([card] * unknown_count)
            
        # 2. Enumerate all partitions of L into H_2 and D_g
        # H_2 needs len_H2 cards.
        # D_g needs len_Dg cards.
        # Check consistency
        if len(L) != len_H2 + len_Dg:
             raise ValueError(f"L size {len(L)} != |H2| {len_H2} + |Dg| {len_Dg}")
             
        self.belief_states = []
        
        # We need unique combinations for H_2.
        # L is a multiset.
        # itertools.combinations treats elements by position, so duplicates in L generate duplicates.
        # We need unique combinations of the *values*.
        
        # Optimization: Use set of sorted tuples to dedup H_2
        unique_h2_hands = set()
        
        # Generate all combinations of indices to pick for H_2
        for indices in itertools.combinations(range(len(L)), len_H2):
            h2_hand = []
            dg_pile = []
            
            # Split L based on indices
            for i, card in enumerate(L):
                if i in indices:
                    h2_hand.append(card)
                else:
                    dg_pile.append(card)
            
            # Sort to canonicalize
            h2_tuple = tuple(sorted(h2_hand))
            
            if h2_tuple in unique_h2_hands:
                continue
            unique_h2_hands.add(h2_tuple)
            
            # Construct State
            # State = (H_1, H_2, D_g, P, P_t, G_o)
            # Note: D_g order matters for drawing?
            # OfflineSolver assumes D_g is a sorted tuple (bag) for Expectiminimax.
            # So we don't need to enumerate D_g permutations.
            
            dg_tuple = tuple(sorted(dg_pile))
            
            # Turn? We assume it's our turn (Player 1) when choosing action.
            # But OfflineSolver state includes turn.
            # If we are Player 1, turn=0.
            # If we are Player 2, turn=1.
            # This solver is generic, but usually used for the active player.
            # Let's assume we are solving for the current player.
            # But OfflineSolver `solve` takes `turn` as param in state.
            # Let's assume we are Player 1 (turn 0) for now, or pass it.
            
            # Wait, `init_belief` is called. `choose_action` is called later.
            # We store the components to reconstruct state.
            
            # Probability? Uniform over unique hands?
            # Actually, the probability depends on the multiplicity.
            # But if we assume uniform prior over physical cards, 
            # then (A, A') in hand is same probability as (A, B) if counts match?
            # No.
            # Let's use the `indices` approach which implicitly handles probability by multiplicity if we didn't dedup.
            # BUT we deduped.
            # Correct approach: Iterate all `combinations` of indices, count how many map to each unique `h2_tuple`.
            # Then Prob(h2_tuple) = count / total_combinations.
            
            # Let's redo without early dedup to get weights, then aggregate.
            pass
            
        # Re-approach for correct probability:
        # 1. Generate all index combinations (nCr).
        # 2. Map to H2, Dg.
        # 3. Aggregate by canonical state.
        
        state_counts = Counter()
        total_combinations = 0
        
        for indices in itertools.combinations(range(len(L)), len_H2):
            h2_hand = []
            dg_pile = []
            for i, card in enumerate(L):
                if i in indices:
                    h2_hand.append(card)
                else:
                    dg_pile.append(card)
            
            # Canonical state key
            # (H1, H2, Dg, Pt, Turn)
            # We assume Turn=0 (Player 1) for now as we are choosing action for P1.
            # If we are P2, we'd use Turn=1.
            # Let's store the parts and decide turn later? No, state needs turn.
            # Let's assume this solver is always instantiated for the AI player.
            # If AI is P2, turn=1.
            # Let's pass `player_id` to init or __init__.
            
            state_key = (tuple(sorted(H_1)), tuple(sorted(h2_hand)), tuple(sorted(dg_pile)), P_t)
            state_counts[state_key] += 1
            total_combinations += 1
            
        # Store states with probabilities
        self.belief_states = []
        for key, count in state_counts.items():
            prob = count / total_combinations
            # key is (H1, H2, Dg, Pt)
            self.belief_states.append((key, prob))
            
    def choose_action(self, player_id: int = 1) -> Tuple[Optional[Action], float]:
        """
        Choose optimal action maximizing expected value.
        Args:
            player_id: 1 for Player 1 (Turn 0), 2 for Player 2 (Turn 1).
        Returns:
            (Best Action, Expected Value)
        """
        # Turn index for OfflineSolver: 0 for P1, 1 for P2.
        turn = 0 if player_id == 1 else 1
        
        # Get legal actions
        # Legal actions depend only on H_1 (or H_2) and P_t.
        # Since H_1 and P_t are known (observed), legal actions are the same for all belief states
        # IF we are Player 1.
        # If we are Player 2, H_2 is unknown to P1, but known to P2.
        # Wait, who is this solver for?
        # "Verify the Online Solver (Particle Filter) against... Exact Belief Solver"
        # The Online Solver is an AI playing the game. It knows its own hand.
        # So H_curr is known. H_other is unknown.
        
        # If we are P1: H1 known, H2 unknown.
        # If we are P2: H2 known, H1 unknown.
        
        # My `init_belief` assumed H1 is known and H2 is unknown (Standard O space).
        # If this solver is acting as P2, we need to swap perspective?
        # Or just feed it the observation from P2's perspective?
        # `game.get_O_space()` returns P1's observation.
        # If we want P2's observation, we should swap H1/H2 in the input to `init_belief`?
        # Or just handle it here.
        
        # Let's assume `init_belief` was called with the CORRECT observation for the player.
        # i.e. H_known is in the first slot of O, H_unknown in second slot (as size).
        # So `self.belief_states` contains (H_known, H_unknown, Dg, Pt).
        
        # So `turn` should always be "my turn". 
        # But OfflineSolver uses absolute turns (0=P1, 1=P2).
        # If we are P2, we are maximizing P2's value (which is minimizing P1's value in zero-sum?).
        # OfflineSolver returns +1 for P1 win, -1 for P2 win.
        # So if we are P2, we want to Minimize the value.
        
        # Let's verify `OfflineSolver.solve`.
        # It returns value from P1 perspective?
        # "Returns: 1.0 (Win), -1.0 (Loss)" -> Yes, P1 win is +1.
        # If turn=1 (P2), it minimizes.
        
        # So:
        # If player_id=1: Maximize E[V].
        # If player_id=2: Minimize E[V].
        
        # 1. Identify Legal Actions
        # We need a game instance to check legal actions.
        # Any state from belief will do for checking *my* legal actions (since my hand is constant).
        if not self.belief_states:
            return None, 0.0
            
        sample_state_key, _ = self.belief_states[0]
        h_known, h_unknown, dg, pt = sample_state_key
        
        # Reconstruct temp game
        temp_game = Uno()
        # If player_id=1, H1=h_known.
        # If player_id=2, H2=h_known? 
        # Actually, `init_belief` parsed O = (H_1, len_H2...).
        # So H_1 is ALWAYS the first element of the key.
        # If we are P2, and we used `get_O_space` (P1 perspective), we are confused.
        
        # Let's assume we are testing P1 vs P2 where P2 is the AI.
        # The AI (P2) observes (H2, len(H1)...).
        # But `Uno.get_O_space` is hardcoded for P1.
        # We need `get_O_space(player)`?
        # Or just manually construct O.
        
        # Let's stick to P1 for simplicity in testing?
        # Or support both.
        # Let's assume the input `game` to `init_belief` is the ground truth.
        # And we want to solve for `player_id`.
        
        # If player_id=1:
        # Known: H1. Unknown: H2, Dg.
        # O = (H1, len(H2), ...)
        
        # If player_id=2:
        # Known: H2. Unknown: H1, Dg.
        # O = (H2, len(H1), ...) -> We need to construct this.
        
        pass 
        # I will implement `choose_action` to take `player_id` and handle the perspective correctly.
        # But `init_belief` needs to know which player it is to parse O correctly?
        # Actually, `init_belief` takes `game`. It has access to everything (Oracle).
        # But it SHOULD restrict itself to what the player sees.
        
        # Let's modify `init_belief` to take `player_id`.
        
    def init_belief_for_player(self, game: Uno, player_id: int):
        self.game = game
        full_deck = game.build_number_deck()
        
        if player_id == 1:
            H_known = game.H_1
            len_unknown = len(game.H_2)
            H_unknown_real = game.H_2 # For debugging/verification? No, shouldn't use.
        else:
            H_known = game.H_2
            len_unknown = len(game.H_1)
            
        len_Dg = len(game.D_g)
        P = game.P
        P_t = game.P_t
        
        # L = D \ (H_known + P)
        known_cards = list(H_known) + list(P)
        known_counter = Counter(known_cards)
        full_counter = Counter(full_deck)
        
        L = []
        for card, count in full_counter.items():
            unknown_count = count - known_counter.get(card, 0)
            L.extend([card] * unknown_count)
            
        # Enumerate partitions of L into H_unknown and Dg
        state_counts = Counter()
        total_combinations = 0
        
        for indices in itertools.combinations(range(len(L)), len_unknown):
            h_unknown_list = []
            dg_pile = []
            for i, card in enumerate(L):
                if i in indices:
                    h_unknown_list.append(card)
                else:
                    dg_pile.append(card)
            
            # Construct canonical state for OfflineSolver (H1, H2, Dg, Pt, Turn)
            if player_id == 1:
                h1 = tuple(sorted(H_known))
                h2 = tuple(sorted(h_unknown_list))
            else:
                h1 = tuple(sorted(h_unknown_list))
                h2 = tuple(sorted(H_known))
                
            dg = tuple(sorted(dg_pile))
            
            # Turn is set when solving.
            state_key = (h1, h2, dg, P_t)
            state_counts[state_key] += 1
            total_combinations += 1
            
        self.belief_states = []
        for key, count in state_counts.items():
            prob = count / total_combinations
            self.belief_states.append((key, prob))
            
    def solve(self, player_id: int) -> Tuple[Optional[Action], float]:
        """
        Computes optimal action.
        """
        if not self.belief_states:
            return None, 0.0
            
        # 1. Get Legal Actions (from any consistent state)
        # Since H_known is constant, legal actions are constant.
        sample_key, _ = self.belief_states[0]
        h1, h2, dg, pt = sample_key
        
        # Reconstruct temp game to get actions
        temp_game = Uno(H_1=list(h1), H_2=list(h2), D_g=list(dg), P=[pt] if pt else [])
        temp_game.create_S()
        actions = temp_game.get_legal_actions(player=player_id)
        
        if not actions:
            return None, 0.0
            
        # 2. Evaluate each action
        best_value = -float('inf') if player_id == 1 else float('inf')
        best_action = None
        
        # Turn for solver: 0 if P1, 1 if P2
        turn = 0 if player_id == 1 else 1
        
        for action in actions:
            expected_value = 0.0
            
            for state_key, prob in self.belief_states:
                h1, h2, dg, pt = state_key
                # Construct solver state
                solver_state = (h1, h2, dg, pt, turn)
                
                # We need Q-value: Value of taking action `a` in state `s`.
                # OfflineSolver.solve(s) gives V(s).
                # We need to simulate `a` to get s', then V(s').
                
                # Simulate action
                # Note: Deterministic play, Stochastic draw.
                
                # To reuse OfflineSolver logic, we can't just call solve(s).
                # solve(s) returns the value of the BEST action.
                # We want the value of THIS action `a`.
                
                # We can simulate the transition and then call solve(s').
                
                val_s_a = 0.0
                
                if action.is_play():
                    # Deterministic
                    next_h1 = list(h1)
                    next_h2 = list(h2)
                    next_pt = action.X_1
                    
                    if turn == 0:
                        next_h1.remove(action.X_1)
                        if len(next_h1) == 0: val_s_a = 1.0
                        else:
                            next_state = (tuple(sorted(next_h1)), tuple(sorted(next_h2)), dg, next_pt, 1 - turn)
                            val_s_a = self.oracle.solve(next_state)
                    else:
                        next_h2.remove(action.X_1)
                        if len(next_h2) == 0: val_s_a = -1.0
                        else:
                            next_state = (tuple(sorted(next_h1)), tuple(sorted(next_h2)), dg, next_pt, 1 - turn)
                            val_s_a = self.oracle.solve(next_state)
                            
                elif action.is_draw():
                    # Stochastic (Expectation over draws)
                    if len(dg) == 0:
                        val_s_a = 0.0 # Draw/Empty
                    else:
                        unique_cards = set(dg)
                        draw_ev = 0.0
                        for card in unique_cards:
                            card_prob = dg.count(card) / len(dg)
                            
                            next_dg = list(dg)
                            next_dg.remove(card)
                            next_h1 = list(h1)
                            next_h2 = list(h2)
                            
                            if turn == 0:
                                next_h1.append(card)
                                next_state = (tuple(sorted(next_h1)), tuple(sorted(next_h2)), tuple(sorted(next_dg)), pt, 1 - turn)
                            else:
                                next_h2.append(card)
                                next_state = (tuple(sorted(next_h1)), tuple(sorted(next_h2)), tuple(sorted(next_dg)), pt, 1 - turn)
                                
                            draw_ev += card_prob * self.oracle.solve(next_state)
                        val_s_a = draw_ev
                        
                expected_value += prob * val_s_a
                
            # Update best
            if player_id == 1: # Maximize
                if expected_value > best_value:
                    best_value = expected_value
                    best_action = action
            else: # Minimize
                if expected_value < best_value:
                    best_value = expected_value
                    best_action = action
                    
        return best_action, best_value

if __name__ == "__main__":
    # Test
    from stage_1_mini_uno.mini_uno import MiniUno
    game = MiniUno()
    game.new_game(seed=42)
    
    oracle = OfflineSolver()
    solver = ExactBeliefSolver(oracle)
    
    print("Initializing belief for P1...")
    solver.init_belief_for_player(game, 1)
    print(f"Number of belief states: {len(solver.belief_states)}")
    
    action, val = solver.solve(1)
    print(f"Optimal Action: {action}, Expected Value: {val}")
