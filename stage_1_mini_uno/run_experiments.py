import sys
import os
import random
import time
from typing import List, Tuple, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage_1_mini_uno.mini_uno import MiniUno
from stage_1_mini_uno.offline_solver import OfflineSolver
from pomdp import Action

# --- Player Classes ---

class Player:
    def get_action(self, game: MiniUno, player_num: int) -> Optional[Action]:
        raise NotImplementedError

class RandomPlayer(Player):
    def get_action(self, game: MiniUno, player_num: int) -> Optional[Action]:
        actions = game.get_legal_actions(player_num)
        if not actions:
            return None # Should not happen
        return random.choice(actions)

class OptimalPlayer(Player):
    def __init__(self, solver: OfflineSolver):
        self.solver = solver

    def get_action(self, game: MiniUno, player_num: int) -> Optional[Action]:
        # Optimal player needs canonical state
        # Turn: 0 for P1, 1 for P2
        turn = player_num - 1
        state = self.solver.get_canonical_state(game, turn)
        
        # Ensure solved
        self.solver.solve(state)
        
        return self.solver.policy.get(state)

# --- Simulation Logic ---

def simulate_game(game_id: int, p1: Player, p2: Player, start_turn: int) -> Tuple[int, int]:
    """
    Returns (winner, turns)
    winner: 1 or 2
    """
    game = MiniUno()
    game.new_game(seed=game_id + int(time.time()))
    
    turn = start_turn
    max_turns = 100
    current_turn = 0
    
    while game.G_o == "Active" and current_turn < max_turns:
        player_num = turn + 1
        
        if player_num == 1:
            action = p1.get_action(game, 1)
        else:
            action = p2.get_action(game, 2)
            
        if action is None:
            print(f"Error: No action for player {player_num}")
            break
            
        success = game.execute_action(action, player=player_num)
        if not success:
            print(f"Error: Invalid action {action} by player {player_num}")
            break
            
        if len(game.H_1) == 0:
            return 1, current_turn + 1
        if len(game.H_2) == 0:
            return 2, current_turn + 1
            
        turn = 1 - turn
        current_turn += 1
        
    return 0, current_turn

def run_experiment(name: str, p1: Player, p2: Player, num_games: int = 5000):
    print(f"\n=== Experiment: {name} ===")
    print(f"Games: {num_games} ({num_games//2} P1 start, {num_games//2} P2 start)")
    
    p1_wins = 0
    p2_wins = 0
    p1_start_wins = 0
    p1_start_games = 0
    p2_start_wins = 0
    p2_start_games = 0
    draws = 0
    total_turns = 0
    
    start_time = time.time()
    
    for i in range(num_games):
        start_turn = 0 if i < (num_games // 2) else 1
        winner, turns = simulate_game(i, p1, p2, start_turn)
        
        total_turns += turns
        if winner == 1:
            p1_wins += 1
            if start_turn == 0: p1_start_wins += 1
        elif winner == 2:
            p2_wins += 1
            if start_turn == 1: p2_start_wins += 1
        else:
            draws += 1
            
        if start_turn == 0: p1_start_games += 1
        else: p2_start_games += 1
            
    elapsed = time.time() - start_time
    avg_turns = total_turns / num_games
    
    print(f"Time: {elapsed:.2f}s")
    print(f"Overall P1 Wins: {p1_wins} ({p1_wins/num_games*100:.1f}%)")
    print(f"Overall P2 Wins: {p2_wins} ({p2_wins/num_games*100:.1f}%)")
    print(f"Draws: {draws}")
    print(f"Avg Turns: {avg_turns:.2f}")
    
    if p1_start_games > 0:
        print(f"  When P1 Starts ({p1_start_games} games): P1 Wins {p1_start_wins} ({p1_start_wins/p1_start_games*100:.1f}%)")
    if p2_start_games > 0:
        # P2 Wins when P2 starts
        print(f"  When P2 Starts ({p2_start_games} games): P2 Wins {p2_start_wins} ({p2_start_wins/p2_start_games*100:.1f}%)")
    
    return {
        "name": name,
        "p1_wins": p1_wins,
        "p2_wins": p2_wins,
        "draws": draws,
        "avg_turns": avg_turns
    }

def main():
    solver = OfflineSolver()
    
    optimal_p = OptimalPlayer(solver)
    random_p = RandomPlayer()
    
    results = []
    
    # 1. Optimal (P1) vs Random (P2)
    res1 = run_experiment("Optimal (P1) vs Random (P2)", optimal_p, random_p)
    results.append(res1)
    
    # 2. Random (P1) vs Random (P2)
    res2 = run_experiment("Random (P1) vs Random (P2)", random_p, random_p)
    results.append(res2)
    
    # 3. Optimal (P1) vs Optimal (P2)
    res3 = run_experiment("Optimal (P1) vs Optimal (P2)", optimal_p, optimal_p)
    results.append(res3)
    
    # Save Summary CSV
    with open("stage_1_mini_uno/experiment_results_5000.csv", "w") as f:
        f.write("Experiment,P1_Wins,P2_Wins,Draws,Avg_Turns\n")
        for r in results:
            f.write(f"{r['name']},{r['p1_wins']},{r['p2_wins']},{r['draws']},{r['avg_turns']:.2f}\n")

if __name__ == "__main__":
    main()
