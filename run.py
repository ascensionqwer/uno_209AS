from simulate_games import *

if __name__ == "__main__":
    # Import required classes from note.txt
    # from cards import Card, RED, YELLOW, GREEN, BLUE
    # from pomdp import State, Action
    # from uno import Uno
    # from belief import Belief

    total_game = 100
    print("Uno AI vs Heuristic Player Test\n")

    # Create players
    ai_player = Uno_AI(player_id=1, num_samples=500, lookahead=1)
    heuristic_player = Uno_Heuristic(player_id=2)

    # Create game controller
    controller = GameController(ai_player, heuristic_player, verbose=True)

    # Play single game
    print("Playing single game with full output:\n")
    winner_list = []
    for i in range(total_game):
      winner, turns = controller.play_game(seed=i+1e2)
      winner_list.append(winner)

    print(f"Total Games: {len(winner_list)}")
    print(f"Number of times '1' is the winner: {winner_list.count(1)}; {winner_list.count(1)/total_game * 100}%")
    print(f"Number of times '2' is the winner: {winner_list.count(2)}; {winner_list.count(2)/total_game * 100}%")

    """
    # Run experiments
    print("\n\nRunning experiment with multiple games:\n")
    results = controller.run_experiments(num_games=5, verbose_games=False)
    """