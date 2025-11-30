from src.utils.game_runner import run_naive_vs_naive_game


def main():
    """Run 2-player UNO game simulator.

    Player 1 uses Naive policy, Player 2 uses Naive policy.
    """
    print("=" * 60)
    print("UNO Simulation: Naive (Player 1) vs Naive (Player 2)")
    print("=" * 60)

    run_naive_vs_naive_game(seed=None, show_output=True)


if __name__ == "__main__":
    main()
