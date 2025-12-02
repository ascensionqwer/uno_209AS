import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage_1_mini_uno.offline_solver import OfflineSolver
from stage_1_mini_uno.online_solver_adapter import MiniUnoAI
from stage_1_mini_uno.mini_uno import MiniUno
from cards import Card

def compare_solvers():
    print("Initializing solvers...")
    offline_solver = OfflineSolver()
    
    # Online solver needs to be initialized per state usually, or reset
    # We'll create a new one per scenario or re-init belief
    online_solver = MiniUnoAI(player_id=1, num_samples=100, lookahead=2)
    
    scenarios = []
    
    # Scenario 1: Choice between matching color or matching number
    # H1: [(R, 1), (B, 2)]
    # Pt: (R, 2)
    s1 = (
        (('R', 1), ('B', 2)), # H1
        (('B', 1), ('R', 0)), # H2
        (('B', 0), ('R', 1), ('B', 2), ('R', 2), ('B', 1)), # Dg (dummy)
        ('R', 2), # Pt
        0 # Turn P1
    )
    scenarios.append(("Scenario 1", s1))
    
    # Scenario 2: Choice between two same colors
    # H1: [(R, 0), (R, 1)]
    # Pt: (R, 2)
    s2 = (
        (('R', 0), ('R', 1)),
        (('B', 1), ('B', 2)),
        tuple(),
        ('R', 2),
        0
    )
    scenarios.append(("Scenario 2", s2))
    
    # Scenario 3: Forced Draw
    # H1: [(B, 0)]
    # Pt: (R, 1)
    s3 = (
        (('B', 0),),
        (('R', 2),),
        (('R', 0),),
        ('R', 1),
        0
    )
    scenarios.append(("Scenario 3", s3))

    # Scenario 4: Immediate Win
    # H1: [(B, 1)]
    # Pt: (B, 0)
    s4 = (
        (('B', 1),),
        (('R', 1), ('R', 2)),
        (('R', 0), ('B', 0), ('B', 2)),
        ('B', 0),
        0
    )
    scenarios.append(("Scenario 4", s4))

    # Scenario 5: Prevent Opponent Win
    # H1: [(R, 1), (B, 2)]
    # H2: [(B, 0)] (Opponent has 1 card)
    # Pt: (R, 2)
    # Moves: (R, 1) -> P2 plays (B, 0) on (R, 1)? No. P2 needs Red or 1.
    # If P1 plays (R, 1), P_t becomes (R, 1). P2 has (B, 0). P2 draws. P1 safe.
    # If P1 plays (B, 2), P_t becomes (B, 2). P2 has (B, 0). P2 plays (B, 0) and WINS.
    # Correct move: (R, 1).
    s5 = (
        (('R', 1), ('B', 2)),
        (('B', 0),),
        (('R', 0), ('B', 1), ('R', 2)),
        ('R', 2),
        0
    )
    scenarios.append(("Scenario 5", s5))

    # Scenario 6: Color Change Strategy
    # H1: [(R, 1), (B, 1)]
    # H2: [(R, 0), (R, 2)] (Opponent has Reds)
    # Pt: (B, 2)
    # Moves: (B, 1) -> P_t=(B, 1). P2 has no Blue, no 1. P2 draws.
    # Moves: (R, 1) -> Invalid on (B, 2)? No, (B, 1) is valid on (B, 2).
    # Wait, (R, 1) is NOT valid on (B, 2) unless number matches.
    # Let's construct: Pt=(B, 1). H1=[(R, 1), (B, 2)].
    # P1 plays (R, 1) -> Pt=(R, 1). P2 has Reds. P2 plays.
    # P1 plays (B, 2) -> Pt=(B, 2). P2 has no Blue, no 2. P2 draws.
    # Best move: (B, 2) to force draw.
    s6 = (
        (('R', 1), ('B', 2)),
        (('R', 0), ('R', 2)),
        (('B', 0), ('B', 1)),
        ('B', 1),
        0
    )
    scenarios.append(("Scenario 6", s6))

    results = []
    
    print(f"{'Scenario':<12} | {'Offline Action':<25} | {'Online Action':<25} | {'Match':<5} | {'Offline Val':<6}")
    print("-" * 90)
    
    for name, state in scenarios:
        # Offline Solution
        off_val = offline_solver.solve(state)
        off_action = offline_solver.policy.get(state)
        
        # Online Solution
        # We need to construct a Game object that matches the state
        # But wait, Online Solver only sees OBSERVATION.
        # It doesn't know H2.
        # We must give it the observation derived from the state.
        
        h1, h2, dg, pt, turn = state
        
        # Create game instance
        game = MiniUno(H_1=list(h1), H_2=list(h2), D_g=list(dg), P=[pt] if pt else [])
        game.create_S()
        
        # Init belief
        online_solver.init_belief(game)
        
        # Choose action
        # Note: Online solver uses sampling. It might vary.
        # We might want to run it multiple times?
        start_time = time.time()
        on_action = online_solver.choose_action()
        elapsed = time.time() - start_time
        
        # Compare
        # Action equality check
        # Action objects might be different instances.
        # We check string repr or attributes.
        match = False
        if off_action is None and on_action is None:
            match = True
        elif off_action and on_action:
            if off_action.is_play() and on_action.is_play():
                match = (off_action.X_1 == on_action.X_1)
            elif off_action.is_draw() and on_action.is_draw():
                match = (off_action.n == on_action.n)
        
        match_str = "YES" if match else "NO"
        
        print(f"{name:<12} | {str(off_action):<25} | {str(on_action):<25} | {match_str:<5} | {off_val:<6.1f}")
        
        results.append({
            "Scenario": name,
            "Offline Action": str(off_action),
            "Online Action": str(on_action),
            "Match": match,
            "Offline Value": off_val
        })

    # Save results
    with open("stage_1_mini_uno/comparison_results.csv", "w") as f:
        f.write("Scenario,Offline Action,Online Action,Match,Offline Value\n")
        for row in results:
            f.write(f"{row['Scenario']},{row['Offline Action']},{row['Online Action']},{row['Match']},{row['Offline Value']}\n")

if __name__ == "__main__":
    compare_solvers()
