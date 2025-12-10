import sys
import os
import random
import multiprocessing

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage_1_mini_uno.mini_uno import MiniUno
from stage_1_mini_uno.online_solver_adapter import MiniUnoAI
from stage_1_mini_uno.naive_mdp_solver import NaiveMiniUnoAI
from pomdp import Action

def run_game(solver_type, seed, log_file=None):
    """
    Runs a single game.
    Player 1: 'POMDP' or 'Naive'
    Player 2: Random
    """
    game = MiniUno()
    game.new_game(seed=seed)
    
    # Initialize solver for Player 1
    if solver_type == 'POMDP':
        solver = MiniUnoAI(player_id=1, num_samples=50, lookahead=2)
    else:
        solver = NaiveMiniUnoAI(player_id=1, num_samples=50, lookahead=2)
        
    solver.init_belief(game)
    
    # Play until game over or max turns
    max_turns = 200
    turns = 0
    
    if log_file:
        log_file.write(f"\n=== Game Seed {seed} ({solver_type}) ===\n")
        log_file.write(f"Initial State: H1={game.H_1}, H2={game.H_2}, Pt={game.P_t}\n")
    
    while game.G_o == 'Active' and turns < max_turns:
        turns += 1
        
        if turns % 2 != 0: # Player 1 Turn
            # Update belief with opponent's last action (if any)
            action = solver.choose_action()
            if action is None:
                if log_file: log_file.write(f"Turn {turns} (P1): No legal actions (Draw?)\n")
                break
                
            if log_file: log_file.write(f"Turn {turns} (P1): {action}\n")
            success = game.execute_action(action)
            if not success:
                if log_file: log_file.write(f"Turn {turns} (P1): Invalid Action! Retrying...\n")
                turns -= 1 # Retry turn
                continue
            
            # If P1 played, did they win?
            if len(game.H_1) == 0:
                if log_file: log_file.write(f"Winner: P1 ({solver_type})\n")
                return 1 # P1 Win
                
        else: # Player 2 Turn (Random)
            actions = game.get_legal_actions(player=2)
            if not actions:
                break # Should not happen
            
            action = random.choice(actions)
            if log_file: log_file.write(f"Turn {turns} (P2): {action}\n")
            success = game.execute_action(action, player=2)
            if not success:
                if log_file: log_file.write(f"Turn {turns} (P2): Invalid Action! Retrying...\n")
                turns -= 1 # Retry turn
                continue
            
            # Update solver's belief about P2's action
            solver.update_belief(action)
            
            if len(game.H_2) == 0:
                if log_file: log_file.write("Winner: P2 (Random)\n")
                return -1 # P2 Win (P1 Loss)

    if log_file: 
        log_file.write(f"Result: Loop Exited. G_o={game.G_o}, Turns={turns}, Max={max_turns}\n")
        log_file.write(f"H1: {len(game.H_1)}, H2: {len(game.H_2)}\n")
    return 0 # Draw/Max turns

def evaluate(num_games=10):
    # Open log file
    with open("stage_1_mini_uno/game_logs.txt", "w") as log_file:
        print(f"Evaluating POMDP vs Random (N={num_games})...")
        pomdp_wins = 0
        pomdp_losses = 0
        draws = 0
        
        for i in range(num_games):
            if i % 10 == 0: print(f"Game {i}/{num_games}")
            res = run_game('POMDP', seed=i, log_file=log_file)
            if res == 1: pomdp_wins += 1
            elif res == -1: pomdp_losses += 1
            else: draws += 1
            
        print(f"POMDP Results: Wins={pomdp_wins}, Losses={pomdp_losses}, Draws={draws}")
        if num_games > 0:
            print(f"Win Rate: {pomdp_wins/num_games:.2%}")
        
        print("\n" + "="*50 + "\n")
        
        print(f"Evaluating Naive vs Random (N={num_games})...")
        naive_wins = 0
        naive_losses = 0
        naive_draws = 0
        
        for i in range(num_games):
            if i % 10 == 0: print(f"Game {i}/{num_games}")
            res = run_game('Naive', seed=i, log_file=log_file)
            if res == 1: naive_wins += 1
            elif res == -1: naive_losses += 1
            else: naive_draws += 1
            
        print(f"Naive Results: Wins={naive_wins}, Losses={naive_losses}, Draws={naive_draws}")
        if num_games > 0:
            print(f"Win Rate: {naive_wins/num_games:.2%}")
    
    print("Logs saved to stage_1_mini_uno/game_logs.txt")

if __name__ == "__main__":
    evaluate(num_games=100)
