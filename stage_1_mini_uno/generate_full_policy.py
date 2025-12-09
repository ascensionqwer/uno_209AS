import sys
import os
import itertools
from collections import Counter
import csv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage_1_mini_uno.offline_solver import OfflineSolver
from stage_1_mini_uno.mini_uno import MiniUno
from cards import Card

def generate_full_policy():
    print("Generating full policy for Mini Uno...")
    solver = OfflineSolver()
    
    # 1. Generate all possible initial deals
    full_deck = MiniUno().build_number_deck()
    deck_cards = sorted(full_deck)
    
    def get_partitions(cards, sizes):
        if not sizes:
            yield []
            return
            
        size = sizes[0]
        for combo in set(itertools.combinations(cards, size)):
            # combo is a tuple of cards
            # Remaining cards
            remaining = list(cards)
            for c in combo:
                remaining.remove(c)
            
            for rest in get_partitions(tuple(remaining), sizes[1:]):
                yield [combo] + rest

    # Sizes: H1(2), H2(2), P(1), Dg(5)
    print("Partitioning cards...")
    partitions = list(get_partitions(tuple(deck_cards), [2, 2, 1, 5]))
    print(f"Found {len(partitions)} unique initial partitions.")
    
    # Now solve each
    print("Solving...")
    for i, p in enumerate(partitions):
        h1, h2, pile, dg = p
        p_t = pile[0]
        
        # Canonical state: (h1, h2, dg, p_t, turn)
        # Turn 0 (Player 1 starts)
        state = (tuple(sorted(h1)), tuple(sorted(h2)), tuple(sorted(dg)), p_t, 0)
        solver.solve(state)
        
        if i % 100 == 0:
            print(f"Processed {i}/{len(partitions)}...")
            
    print(f"Policy computed. Total states in policy: {len(solver.policy)}")
    
    # Save to CSV
    print("Saving to CSV...")
    
    output_file = "stage_1_mini_uno/full_offline_policy.csv"
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["H1", "H2", "Dg", "Pt", "Turn", "Value", "Action"])
        
        for state, action in solver.policy.items():
            h1, h2, dg, pt, turn = state
            val = solver.memo.get(state, 0.0)
            writer.writerow([str(h1), str(h2), str(dg), str(pt), turn, val, str(action)])
            
    print(f"Done. Saved to {output_file}")

if __name__ == "__main__":
    generate_full_policy()
