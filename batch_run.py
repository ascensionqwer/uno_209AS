"""Batch simulation runner for UNO game.

Runs multiple simulations and outputs win statistics.
"""

from src.utils.config_loader import load_config
from src.utils.game_runner import run_single_game


def main():
    """Run batch simulations and output statistics."""
    config = load_config()
    batch_config = config.get("batch_run", {})
    num_simulations = batch_config.get("num_simulations", 10000)

    print("=" * 80)
    print("UNO BATCH SIMULATION RUNNER")
    print("=" * 80)
    print(f"Running {num_simulations} simulations...")
    print("Player 1: ParticlePolicy")
    print("Player 2: Simple Policy")
    print("=" * 80)
    print()

    player1_wins = 0
    player2_wins = 0
    no_winner = 0
    total_turns = 0
    min_turns = float("inf")
    max_turns = 0
    games_with_issues = []

    # Run simulations
    for i in range(num_simulations):
        print("\n" + "=" * 80)
        print(f"GAME {i + 1} / {num_simulations}")
        print("=" * 80)
        print("UNO Simulation: ParticlePolicy (Player 1) vs Simple Policy (Player 2)")
        print("=" * 60)

        try:
            # Run single game with full verbose logging (same as main.py)
            # Only show config on first game
            turn_count, winner = run_single_game(seed=i, show_config=(i == 0))

            if winner == 1:
                player1_wins += 1
            elif winner == 2:
                player2_wins += 1
            else:
                no_winner += 1
                games_with_issues.append((i + 1, turn_count, "no_winner"))

            total_turns += turn_count
            min_turns = min(min_turns, turn_count)
            max_turns = max(max_turns, turn_count)

        except Exception as e:
            print(f"\nERROR in simulation {i + 1}: {e}")
            import traceback

            traceback.print_exc()
            no_winner += 1
            games_with_issues.append((i + 1, 0, f"error: {str(e)}"))

    # Final aggregated statistics
    print("\n" + "=" * 80)
    print("BATCH SIMULATION RESULTS")
    print("=" * 80)
    print(f"Total simulations: {num_simulations}")
    print()
    print("Win Statistics:")
    print(
        f"  Player 1 (ParticlePolicy) wins: {player1_wins} ({100.0 * player1_wins / num_simulations:.2f}%)"
    )
    print(
        f"  Player 2 (Simple Policy) wins: {player2_wins} ({100.0 * player2_wins / num_simulations:.2f}%)"
    )
    if no_winner > 0:
        print(
            f"  No winner (safety limit/errors): {no_winner} ({100.0 * no_winner / num_simulations:.2f}%)"
        )
    print()
    print("Turn Statistics:")
    print(f"  Average turns per game: {total_turns / num_simulations:.2f}")
    print(f"  Minimum turns: {min_turns}")
    print(f"  Maximum turns: {max_turns}")
    if games_with_issues:
        print()
        print(f"Games with issues: {len(games_with_issues)}")
        if len(games_with_issues) <= 10:
            for game_id, turns, issue in games_with_issues:
                print(f"  Game {game_id}: {issue} (turns: {turns})")
        else:
            print("  First 10 issues:")
            for game_id, turns, issue in games_with_issues[:10]:
                print(f"    Game {game_id}: {issue} (turns: {turns})")
    print("=" * 80)


if __name__ == "__main__":
    main()
