"""
Q-MDP POMDP Solver Demonstration for UNO

This script demonstrates the complete pipeline:
1. OFFLINE: Generate state transition table and compute Q-values
2. ONLINE: Use belief state + particles to select optimal actions

Reward Structure: Sparse rewards (+1 win, -1 loss, 0 otherwise)
Discounting: gamma = 0.95 (configurable)
POMDP: Uses belief state and particle sampling for action selection
"""

from cards import RED, YELLOW, GREEN, BLUE
from uno import Uno
from belief import Belief
from policy_generator import PolicyGenerator
from pomdp import Action
import random

def demonstrate_qmdp_solver():
    print("=" * 80)
    print("Q-MDP POMDP SOLVER FOR UNO")
    print("=" * 80)
    
    # ========================================================================
    # PHASE 1: OFFLINE - BUILD TRANSITION TABLE AND COMPUTE Q-VALUES
    # ========================================================================
    print("\n[PHASE 1: OFFLINE POLICY GENERATION]")
    print("-" * 80)
    
    pg = PolicyGenerator()
    
    # Define a small initial state to explore
    # This will build a reachable state space from this starting point
    initial_state = pg._to_canonical(
        h1=[(RED, 1), (BLUE, 2)],
        h2=[(GREEN, 1), (YELLOW, 2)],
        p_t=(RED, 3),
        g_o="Active",
        turn=1
    )
    
    print(f"Building state transition table from initial state...")
    print(f"  Initial State: {initial_state}")
    
    # Build the MDP transition graph
    pg.build_transition_table(initial_state, max_depth=100)
    
    print(f"\n✓ Transition table built:")
    print(f"  - States: {len(pg.states)}")
    print(f"  - Transition entries: {len(pg.transitions)}")
    
    # Compute Q-values using Value Iteration
    print(f"\nComputing Q-values via Value Iteration (gamma=0.95)...")
    pg.value_iteration(gamma=0.95, theta=1e-4)
    
    print(f"✓ Q-values computed")
    
    # Show some sample Q-values
    print(f"\nSample Q-values from the table:")
    for i, state in enumerate(pg.states[:3]):
        if i in pg.Q_table:
            print(f"  State {i}: {state}")
            for action_str, q_val in list(pg.Q_table[i].items())[:2]:
                print(f"    {action_str}: Q={q_val:.4f}")
    
    # ========================================================================
    # PHASE 2: ONLINE - USE BELIEF STATE FOR ACTION SELECTION
    # ========================================================================
    print("\n\n[PHASE 2: ONLINE ACTION SELECTION WITH BELIEF STATE]")
    print("-" * 80)
    
    # Create a game scenario
    uno = Uno()
    uno.new_game(seed=42, deal=4)
    uno.create_S()
    
    print(f"\nGame Setup:")
    print(f"  Player 1 Hand: {uno.H_1}")
    print(f"  Player 2 Hand Size: {len(uno.H_2)} cards (unknown to Player 1)")
    print(f"  Top Card (P_t): {uno.P_t}")
    
    # Create belief state (Player 1's perspective)
    observation = uno.get_O_space()
    belief = Belief(observation)
    
    print(f"\nBelief State:")
    print(f"  {belief}")
    
    # Sample particles from belief
    print(f"\nSampling {10} particles from belief:")
    particles = belief.sample_states(n_samples=10)
    for i, particle in enumerate(particles[:3]):
        print(f"  Particle {i+1}: H_2={particle[1]}")
    
    # Get legal actions using game logic
    legal_actions = uno.get_legal_actions(player=1)
    print(f"\nLegal actions from game rules:")
    for action in legal_actions:
        print(f"  - {action}")
    
    # Use Q-MDP to select best action
    print(f"\nUsing Q-MDP to select action...")
    print(f"  (Aggregating Q-values across belief particles)")
    
    best_action = pg.get_best_action(belief, num_particles=100)
    
    if best_action:
        print(f"\n✓ Q-MDP Selected Action: {best_action}")
    else:
        print(f"\n⚠ No action found in Q-table (state space may not cover this scenario)")
        print(f"  Falling back to first legal action: {legal_actions[0]}")
        best_action = legal_actions[0]
    
    # Execute the action
    print(f"\nExecuting action...")
    success = uno.execute_action(best_action, player=1)
    
    if success:
        print(f"✓ Action executed successfully")
        print(f"  New top card: {uno.P_t}")
        print(f"  Player 1 hand size: {len(uno.H_1)}")
    
    # ========================================================================
    # DEMONSTRATION: REWARD STRUCTURE
    # ========================================================================
    print("\n\n[REWARD STRUCTURE DEMONSTRATION]")
    print("-" * 80)
    print("The Q-MDP solver uses SPARSE REWARDS:")
    print("  - Win (Player 1 empties hand): +1.0")
    print("  - Loss (Player 2 empties hand): -1.0")
    print("  - All other transitions: 0.0")
    print("\nValue Iteration uses DISCOUNTED RETURNS:")
    print("  V(s) = max_a Σ P(s'|s,a) [R(s,a,s') + γ·V(s')]")
    print("  where γ = 0.95 (discount factor)")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n\n[SUMMARY]")
    print("=" * 80)
    print("✓ Built MDP transition table (state space exploration)")
    print("✓ Computed Q(s,a) values using Value Iteration with sparse rewards")
    print("✓ Demonstrated POMDP action selection using belief particles")
    print("✓ Q-MDP aggregates Q-values: a* = argmax_a Σ_particles Q(s,a)")
    print("\nThis implements a practical POMDP solver for UNO with:")
    print("  - Limited information (belief state)")
    print("  - Sparse rewards (win/loss only)")
    print("  - Discounted returns (gamma = 0.95)")
    print("=" * 80)

if __name__ == "__main__":
    demonstrate_qmdp_solver()
