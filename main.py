from src.utils.game_runner import run_single_game


def main():
    """Run 2-player UNO game simulator.

    Player 1 uses ParticlePolicy, Player 2 uses simple policy.
    """
    print("=" * 60)
    print("UNO Simulation: ParticlePolicy (Player 1) vs Simple Policy (Player 2)")
    print("=" * 60)

    run_single_game(seed=None, show_config=True)


if __name__ == "__main__":
    main()
