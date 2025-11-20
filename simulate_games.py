"""
Game Simulation: Q-MDP Policy vs Random Opponent

This script simulates full UNO games to verify the policy performance.
Compares win rate of Q-MDP agent vs random baseline.
"""

from cards import RED, YELLOW, GREEN, BLUE
from uno import Uno
from belief import Belief
from policy_generator import PolicyGenerator
from pomdp import Action
import random

def simulate_game_with_policy(pg, seed=None, verbose=False):
    """
    Simulates a game where Player 1 uses Q-MDP policy and Player 2 plays randomly.
    Returns: 1 if Player 1 wins, -1 if Player 2 wins, 0 if draw/error
    """
    uno = Uno()
    uno.new_game(seed=seed, deal=3)  # Smaller hands for tractable state space
    uno.create_S()
    
    # Initialize belief for Player 1
    observation = uno.get_O_space()
    belief = Belief(observation)
    
    max_turns = 100  # Prevent infinite loops
    turn_count = 0
    
    while uno.G_o == "Active" and turn_count < max_turns:
        turn_count += 1
        current_player = (turn_count % 2) + 1  # Alternates 1, 2, 1, 2...
        
        if verbose:
            print(f"\n--- Turn {turn_count} (Player {current_player}) ---")
            print(f"Top card: {uno.P_t}")
        
        legal_actions = uno.get_legal_actions(player=current_player)
        
        if not legal_actions:
            if verbose:
                print("No legal actions available")
            break
        
        # Select action based on player
        if current_player == 1:
            # Use Q-MDP policy
            action = pg.get_best_action(belief, num_particles=50)
            
            # Fallback to random if policy doesn't cover this state
            if action is None or action not in legal_actions:
                action = random.choice(legal_actions)
                if verbose:
                    print(f"Policy fallback to random: {action}")
        else:
            # Player 2: Random
            action = random.choice(legal_actions)
        
        if verbose:
            print(f"Action: {action}")
        
        # Execute action
        success = uno.execute_action(action, player=current_player)
        
        if not success:
            if verbose:
                print("Action failed")
            break
        
        # Update belief if we're Player 1 and opponent just acted
        if current_player == 2:
            new_observation = uno.get_O_space()
            belief.update(action, new_observation)
        
        # Check win condition
        if current_player == 1 and len(uno.H_1) == 0:
            if verbose:
                print("\n✓ Player 1 WINS!")
            return 1
        elif current_player == 2 and len(uno.H_2) == 0:
            if verbose:
                print("\n✗ Player 2 WINS!")
            return -1
    
    if verbose:
        print(f"\nGame ended after {turn_count} turns")
    return 0

def build_policy_for_small_game():
    """Build a policy table for small UNO games (3 cards each)"""
    print("Building Q-MDP policy with 10M state target...")
    print("(This will take several minutes...)")
    import time
    start_time = time.time()
    
    pg = PolicyGenerator()
    
    # Strategy: Generate initial states from ACTUAL game setups
    # This ensures our Q-table covers the states we'll encounter in simulation
    from uno import Uno
    
    print(f"Generating initial states from real game scenarios...")
    initial_states = []
    
    # Generate states from MANY more game seeds for better coverage
    for seed in range(500):  # Increased from 50 to 500
        uno = Uno()
        uno.new_game(seed=seed, deal=3)
        uno.create_S()
        
        # Convert full state to our canonical MDP state
        h1 = tuple(sorted(uno.H_1))
        h2 = tuple(sorted(uno.H_2))
        p_t = uno.P_t
        g_o = uno.G_o
        
        canonical = pg._to_canonical(h1, h2, p_t, g_o, 1)
        if canonical not in initial_states:
            initial_states.append(canonical)
    
    print(f"Generated {len(initial_states)} unique initial states from game setups")
    
    # Build transition table from these states
    print(f"Building transition table (targeting ~10M states)...")
    last_report_time = time.time()
    
    for i, state in enumerate(initial_states):
        current_time = time.time()
        # Report every 5 seconds instead of every 10 states
        if current_time - last_report_time > 5:
            elapsed = current_time - start_time
            print(f"  Progress: {i}/{len(initial_states)} seeds, {len(pg.states):,} total states, {elapsed:.1f}s elapsed")
            last_report_time = current_time
        
        pg.build_transition_table(state, max_depth=5000)  # Increased depth
        
        # Early stop if we've hit the limit
        if len(pg.states) >= 10000000:
            print(f"\n  Reached 10M state limit with {len(pg.states):,} states")
            break
    
    build_time = time.time() - start_time
    print(f"\n✓ Built table with {len(pg.states):,} states in {build_time:.1f}s")
    
    print("Running Value Iteration (this may take a minute)...")
    vi_start = time.time()
    pg.value_iteration(gamma=0.95, theta=1e-4)
    vi_time = time.time() - vi_start
    print(f"✓ Policy ready ({vi_time:.1f}s)")
    
    # Count non-terminal states with Q-values
    q_states = len([s for s in pg.Q_table if pg.Q_table[s]])
    print(f"  Q-table covers {q_states:,} non-terminal states")
    
    total_time = time.time() - start_time
    print(f"  Total time: {total_time:.1f}s")
    
    return pg

def run_simulation(num_games=20):
    """Run simulation of multiple games"""
    print("=" * 80)
    print("GAME SIMULATION: Q-MDP POLICY vs RANDOM OPPONENT")
    print("=" * 80)
    
    # Build policy
    pg = build_policy_for_small_game()
    
    print(f"\nSimulating {num_games} games...")
    print("-" * 80)
    
    results = []
    for i in range(num_games):
        result = simulate_game_with_policy(pg, seed=i, verbose=(i == 0))
        results.append(result)
        
        if i == 0:
            print("\n(Remaining games will run silently...)")
    
    # Statistics
    wins = results.count(1)
    losses = results.count(-1)
    draws = results.count(0)
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Total Games: {num_games}")
    print(f"  Q-MDP Agent Wins (Player 1): {wins} ({100*wins/num_games:.1f}%)")
    print(f"  Random Agent Wins (Player 2): {losses} ({100*losses/num_games:.1f}%)")
    print(f"  Draws/Timeouts: {draws} ({100*draws/num_games:.1f}%)")
    print("=" * 80)
    
    if wins > losses:
        print("✓ Q-MDP policy shows advantage over random play")
    elif wins == losses:
        print("≈ Q-MDP policy performs similarly to random play")
    else:
        print("⚠ Q-MDP policy underperforms (may need larger state coverage)")

if __name__ == "__main__":
    run_simulation(num_games=100)
