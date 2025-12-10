import sys
import os
import random
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cards import RED, BLUE, GREEN, YELLOW

from stage_1_mini_uno.flexible_uno import FlexibleUno
from stage_1_mini_uno.online_solver_adapter import MiniUnoAI
from stage_1_mini_uno.naive_mdp_solver import NaiveMiniUnoAI

def run_game(game_config, solver_type, seed, log_file=None):
    """
    Runs a single game with specific configuration.
    """
    # Initialize game with config
    game = FlexibleUno(
        colors=game_config['colors'],
        ranks=game_config['ranks'],
        copies=game_config['copies']
    )
    game.new_game(seed=seed, deal=game_config['deal'])
    
    # Initialize solver for Player 1
    if solver_type == 'POMDP':
        solver = MiniUnoAI(player_id=1, num_samples=50, lookahead=2)
    else:
        solver = NaiveMiniUnoAI(player_id=1, num_samples=50, lookahead=2)
        
    solver.init_belief(game)
    
    max_turns = 200
    turns = 0
    
    while game.G_o == 'Active' and turns < max_turns:
        turns += 1
        
        if turns % 2 != 0: # Player 1 Turn
            action = solver.choose_action()
            if action is None:
                break
                
            success = game.execute_action(action)
            if not success:
                turns -= 1 # Retry
                continue
            
            if len(game.H_1) == 0:
                return 1 # P1 Win
                
        else: # Player 2 Turn (Random)
            actions = game.get_legal_actions(player=2)
            if not actions:
                break
            
            action = random.choice(actions)
            success = game.execute_action(action, player=2)
            if not success:
                turns -= 1 # Retry
                continue
            
            solver.update_belief(action)
            
            if len(game.H_2) == 0:
                return -1 # P2 Win
                
    return 0 # Draw

def run_experiment(config_name, config, num_games=50):
    print(f"\nRunning Experiment: {config_name}")
    print(f"Config: {config}")
    
    results = {}
    
    for solver_type in ['POMDP', 'Naive']:
        print(f"  Testing {solver_type}...")
        wins = 0
        losses = 0
        draws = 0
        
        for i in range(num_games):
            res = run_game(config, solver_type, seed=i)
            if res == 1: wins += 1
            elif res == -1: losses += 1
            else: draws += 1
            
        win_rate = wins / num_games if num_games > 0 else 0
        results[solver_type] = {'Wins': wins, 'Losses': losses, 'Draws': draws, 'WinRate': win_rate}
        print(f"    {solver_type}: {wins}W - {losses}L - {draws}D ({win_rate:.1%})")
        
    return results

if __name__ == "__main__":
    experiments = {
        'Micro': {
            'colors': [RED, BLUE],
            'ranks': [0, 1],
            'copies': 1,
            'deal': 1 # Very small hands for very small deck (4 cards total)
        },
        'Mini': {
            'colors': [RED, BLUE],
            'ranks': [0, 1, 2],
            'copies': 2,
            'deal': 2 # Standard Mini Uno deal (12 cards total)
        },
        'Small': {
            'colors': [RED, BLUE, GREEN],
            'ranks': [0, 1, 2, 3],
            'copies': 2,
            'deal': 3 # Slightly larger hands (24 cards total)
        },
        'Medium': {
            'colors': [RED, BLUE, GREEN, YELLOW],
            'ranks': [0, 1, 2, 3, 4],
            'copies': 2,
            'deal': 4 # Larger hands (40 cards total)
        }
    }
    
    with open("stage_1_mini_uno/experiment_results.txt", "w") as f:
        for name, config in experiments.items():
            f.write(f"=== {name} ===\n")
            results = run_experiment(name, config, num_games=50)
            f.write(f"POMDP: {results['POMDP']['WinRate']:.1%}\n")
            f.write(f"Naive: {results['Naive']['WinRate']:.1%}\n")
            f.write("\n")
            
    print("\nExperiments completed. Results saved to stage_1_mini_uno/experiment_results.txt")
