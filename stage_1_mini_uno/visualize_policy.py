import sys
import os

from typing import List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage_1_mini_uno.offline_solver import OfflineSolver
from stage_1_mini_uno.mini_uno import MiniUno
from cards import Card, RED, BLUE

def visualize_policy():
    solver = OfflineSolver()
    
    # Define a set of interesting states to visualize
    # We can't show all 10^5 states.
    # Let's show states where P1 has a choice.
    
    states_to_show = []
    
    # Scenario 1: Choice between matching color or matching number
    # H1: [(R, 1), (B, 2)]
    # Pt: (R, 2)
    # Legal: (R, 1) [Color], (B, 2) [Number]
    s1 = (
        (('R', 1), ('B', 2)), # H1
        (('B', 1), ('R', 0)), # H2
        (('B', 0), ('R', 1), ('B', 2), ('R', 2), ('B', 1)), # Dg (dummy)
        ('R', 2), # Pt
        0 # Turn P1
    )
    states_to_show.append(s1)
    
    # Scenario 2: Choice between two same colors
    # H1: [(R, 0), (R, 1)]
    # Pt: (R, 2)
    # Legal: Both
    s2 = (
        (('R', 0), ('R', 1)),
        (('B', 1), ('B', 2)),
        tuple(),
        ('R', 2),
        0
    )
    states_to_show.append(s2)
    
    # Scenario 3: Forced Draw
    # H1: [(B, 0)]
    # Pt: (R, 1)
    # Legal: None -> Draw
    s3 = (
        (('B', 0),),
        (('R', 2),),
        (('R', 0),),
        ('R', 1),
        0
    )
    states_to_show.append(s3)

    results = []
    for state in states_to_show:
        val = solver.solve(state)
        action = solver.policy.get(state)
        
        # Format for table
        h1_str = str(state[0])
        h2_str = str(state[1])
        pt_str = str(state[3])
        results.append({
            "H1": h1_str,
            "H2": h2_str,
            "Pt": pt_str,
            "Value": val,
            "Optimal Action": str(action)
        })
        
    # Print table manually
    print(f"{'H1':<30} | {'H2':<30} | {'Pt':<10} | {'Value':<6} | {'Optimal Action'}")
    print("-" * 100)
    for row in results:
        print(f"{row['H1']:<30} | {row['H2']:<30} | {row['Pt']:<10} | {row['Value']:<6.1f} | {row['Optimal Action']}")
    
    # Save to CSV manually
    with open("stage_1_mini_uno/policy_table.csv", "w") as f:
        f.write("H1,H2,Pt,Value,Optimal Action\n")
        for row in results:
            # Escape commas in string representations if necessary, but tuple repr usually has commas
            # Simple CSV might break with tuple reprs. Let's just use pipe or something if needed, 
            # or quote them.
            h1 = f'"{row["H1"]}"'
            h2 = f'"{row["H2"]}"'
            pt = f'"{row["Pt"]}"'
            action = f'"{row["Optimal Action"]}"'
            f.write(f"{h1},{h2},{pt},{row['Value']},{action}\n")

if __name__ == "__main__":
    visualize_policy()
