import sys
import os
import csv
import itertools
from collections import Counter
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage_1_mini_uno.offline_solver import OfflineSolver
from stage_1_mini_uno.mini_uno import MiniUno
from uno_ai import Uno_AI
from cards import Card

def generate_tables():
    print("Generating tables for Mini Uno (10 cards)...")
    
    # Ensure output directory exists
    output_dir = "stage_1_mini_uno/generated_tables"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Generate Oracle Table (MDP)
    # We reuse the logic from generate_full_policy.py but formatted for this request.
    
    solver = OfflineSolver()
    full_deck = MiniUno().build_number_deck()
    deck_cards = sorted(full_deck)
    
    # Generate partitions
    def get_partitions(cards, sizes):
        if not sizes:
            yield []
            return
        size = sizes[0]
        for combo in set(itertools.combinations(cards, size)):
            remaining = list(cards)
            for c in combo:
                remaining.remove(c)
            for rest in get_partitions(tuple(remaining), sizes[1:]):
                yield [combo] + rest

    print("Partitioning cards...")
    partitions = list(get_partitions(tuple(deck_cards), [2, 2, 1, 5]))
    print(f"Found {len(partitions)} unique initial partitions.")
    
    print("Solving Oracle...")
    # Solve all reachable states
    for i, p in enumerate(partitions):
        h1, h2, pile, dg = p
        p_t = pile[0]
        state = (tuple(sorted(h1)), tuple(sorted(h2)), tuple(sorted(dg)), p_t, 0)
        solver.solve(state)
        
    print(f"Total states in Oracle Policy: {len(solver.policy)}")
    
    # Save Oracle Table
    oracle_file = os.path.join(output_dir, "oracle_table_10.csv")
    print(f"Saving Oracle Table to {oracle_file}...")
    
    # We need to save the states in a way we can reload them for POMDP generation.
    # We'll store them in a list first.
    all_states = []
    
    with open(oracle_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["H1", "H2", "Dg", "Pt", "Turn", "Value", "Action"])
        
        for state, action in solver.policy.items():
            h1, h2, dg, pt, turn = state
            
            # User requested to treat opponent turns as black box environment transitions.
            # We only list states where the Agent (P1) makes a decision.
            if turn != 0:
                continue
                
            val = solver.memo.get(state, 0.0)
            writer.writerow([str(h1), str(h2), str(dg), str(pt), turn, val, str(action)])
            all_states.append((state, action, val))
            
    # 2. Generate POMDP Table
    # For each state in Oracle table (where Turn=0 i.e. Player 1), run Uno_AI.
    # We only care about P1's decisions.
    
    pomdp_file = os.path.join(output_dir, "pomdp_table_10.csv")
    print(f"Generating POMDP Table to {pomdp_file}...")
    
    # Initialize AI
    ai = Uno_AI(player_id=1, num_samples=100, lookahead=2)
    
    with open(pomdp_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["H1", "H2", "Dg", "Pt", "Turn", "Oracle Action", "POMDP Action", "Match"])
        
        count = 0
        total_p1 = sum(1 for s in all_states if s[0][4] == 0)
        
        for state, oracle_action, oracle_val in all_states:
            h1, h2, dg, pt, turn = state
            
            if turn != 0: # Skip P2 states
                continue
                
            # Construct Game Observation
            # O = (H_1, |H_2|, |D_g|, P, P_t, G_o)
            # P is not in state directly. State has P_t.
            # But we know P contains P_t.
            # And P might contain other cards?
            # In OfflineSolver state, P is abstracted away except P_t.
            # But Uno_AI needs P to compute L = D \ (H1 + P).
            # If we assume P = [P_t] (minimal history), then L includes all other cards.
            # Is this correct?
            # If the game has progressed, P has more cards.
            # But OfflineSolver state doesn't track P history.
            # This means OfflineSolver policy is valid for ANY P history ending in P_t.
            # However, Uno_AI belief depends on P history (known cards).
            # If we assume P=[P_t], we assume minimal knowledge (maximum uncertainty).
            # This is the "hardest" case for POMDP.
            # Or should we try to infer P? Impossible.
            # So we assume P = [P_t].
            
            # Construct temp game for AI
            temp_game = MiniUno(H_1=list(h1), H_2=list(h2), D_g=list(dg), P=[pt])
            temp_game.create_S()
            
            # Init AI
            ai.init_belief(temp_game)
            
            # Choose Action
            pomdp_action = ai.choose_action()
            
            # Compare
            match = (str(oracle_action) == str(pomdp_action))
            
            writer.writerow([str(h1), str(h2), str(dg), str(pt), turn, str(oracle_action), str(pomdp_action), match])
            
            count += 1
            if count % 100 == 0:
                print(f"Processed {count}/{total_p1} POMDP states...")
                
    print("Done.")

if __name__ == "__main__":
    generate_tables()
