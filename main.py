from game_controller import GameController
from uno_heuristic import Uno_Heuristic
from uno_ai import Uno_AI
from plot_result import plot_uno_results


if __name__ == "__main__":
    # Heuristic vs Heuristic
    total_game = 1000
    print("Uno AI vs Heuristic Player Test\n")

    # Create players
    heuristic_player_1 = Uno_Heuristic(player_id=1)
    heuristic_player_2 = Uno_Heuristic(player_id=2)

    # Create game controller
    controller = GameController(heuristic_player_1, heuristic_player_2, verbose=False)

    # Play single game
    print("Playing single game with full output:\n")
    winner_list = []
    for i in range(total_game):
      winner, turns = controller.play_game(seed=i+67)
      winner_list.append(winner)

    print(f"Total Games: {len(winner_list)}")
    print(f"Number of times '1' is the winner: {winner_list.count(1)}; {winner_list.count(1)/total_game * 100}%")
    print(f"Number of times '2' is the winner: {winner_list.count(2)}; {winner_list.count(2)/total_game * 100}%")
    plot_uno_results(winner_list)
    
    
    # AI vs AI
    total_game = 1000
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
      winner, turns = controller.play_game(seed=i+67)
      winner_list.append(winner)

    print(f"Total Games: {len(winner_list)}")
    print(f"Number of times '1' is the winner: {winner_list.count(1)}; {winner_list.count(1)/total_game * 100}%")
    print(f"Number of times '2' is the winner: {winner_list.count(2)}; {winner_list.count(2)/total_game * 100}%")
    plot_uno_results(winner_list)