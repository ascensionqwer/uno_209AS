import sys
import os
import time
import csv
import random

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage_1_mini_uno.flexible_uno import FlexibleUno, generate_deck
from stage_1_mini_uno.offline_solver import OfflineSolver
from stage_1_mini_uno.exact_belief_solver import ExactBeliefSolver
from uno_ai import Uno_AI
from uno import Uno

class TestingPipeline:
    def __init__(self):
        self.oracle = OfflineSolver()
        self.exact_solver = ExactBeliefSolver(self.oracle)
        # Particle solver will be instantiated per game/config
        
    def run_experiment(self, deck_sizes=[10, 11, 12], num_games_per_size=10):
        results = []
        
        for size in deck_sizes:
            print(f"\nRunning Experiment for Deck Size: {size}")
            deck = generate_deck(size)
            
            # Re-initialize Oracle with FlexibleUno for this deck size?
            # OfflineSolver uses MiniUno() by default.
            # We need to inject the correct game logic into OfflineSolver if it uses it for get_legal_actions.
            # Yes, OfflineSolver.game is MiniUno().
            # We should update it to FlexibleUno(deck).
            self.oracle.game = FlexibleUno(custom_deck=deck)
            
            for i in range(num_games_per_size):
                print(f"  Game {i+1}/{num_games_per_size}...")
                
                # Generate random game state
                game = FlexibleUno(custom_deck=deck)
                game.new_game(seed=random.randint(0, 10000))
                
                # Fast forward a bit to get interesting mid-game states
                # (Start of game is also interesting)
                turns = random.randint(0, 3)
                for _ in range(turns):
                    if game.G_o != "Active": break
                    actions = game.get_legal_actions()
                    if actions:
                        game.execute_action(random.choice(actions))
                
                if game.G_o != "Active": continue
                
                # We test Player 1's decision
                player_id = 1
                
                # 1. Oracle (Perfect Info)
                # Note: Oracle value is for the specific hidden state.
                # It doesn't give "Expected Value" over belief, it gives "True Value".
                # But we can use it to see what the "God" move is.
                state_oracle = self.oracle.get_canonical_state(game, turn=0)
                val_oracle = self.oracle.solve(state_oracle)
                action_oracle = self.oracle.policy.get(state_oracle)
                
                # 2. Exact Belief (Exhaustive Limited Info)
                start_time = time.time()
                self.exact_solver.init_belief_for_player(game, player_id)
                action_exact, val_exact = self.exact_solver.solve(player_id)
                time_exact = time.time() - start_time
                
                # 3. Particle Solver (Online)
                # Use Uno_AI from uno_ai.py
                particle_solver = Uno_AI(player_id=player_id, num_samples=100, lookahead=2)
                
                start_time = time.time()
                particle_solver.init_belief(game) 
                action_particle = particle_solver.choose_action()
                time_particle = time.time() - start_time
                
                # Compare
                match_exact_particle = (str(action_exact) == str(action_particle))
                
                results.append({
                    "Deck Size": size,
                    "Game ID": i,
                    "Oracle Action": str(action_oracle),
                    "Oracle Value": val_oracle,
                    "Exact Action": str(action_exact),
                    "Exact Value": val_exact,
                    "Exact Time": time_exact,
                    "Particle Action": str(action_particle),
                    "Particle Time": time_particle,
                    "Match (Exact vs Particle)": match_exact_particle
                })
                
        # Save results
        output_file = "stage_1_mini_uno/pipeline_results.csv"
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            
        print(f"Pipeline completed. Results saved to {output_file}.")

if __name__ == "__main__":
    pipeline = TestingPipeline()
    pipeline.run_experiment(deck_sizes=[10, 11, 12], num_games_per_size=10)
