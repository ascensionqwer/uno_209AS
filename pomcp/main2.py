from src.utils.game_runner import run_matchup_game
from src.utils.matchup_types import PlayerType, Matchup


def main():
    """Run 2-player UNO game simulator.

    Runs two games: naive vs particle (naive first), then naive vs particle (particle first).
    """
    print("=" * 60)
    print("UNO Simulation: Naive vs Particle Policy (2 games)")
    print("=" * 60)

    # Game 1: Naive (Player 1) vs Particle (Player 2)
    print("\n" + "=" * 60)
    print("GAME 1: Naive (Player 1) vs Particle Policy (Player 2)")
    print("=" * 60)

    matchup1 = Matchup(PlayerType.NAIVE, PlayerType.PARTICLE_POLICY)
    turn_count1, winner1, _ = run_matchup_game(matchup1, seed=None, show_output=True)

    print(f"\nGame 1 Result: {turn_count1} turns, Winner: Player {winner1}")

    # Game 2: Particle (Player 1) vs Naive (Player 2)
    print("\n" + "=" * 60)
    print("GAME 2: Particle Policy (Player 1) vs Naive (Player 2)")
    print("=" * 60)

    matchup2 = Matchup(PlayerType.PARTICLE_POLICY, PlayerType.NAIVE)
    turn_count2, winner2, _ = run_matchup_game(matchup2, seed=None, show_output=True)

    print(f"\nGame 2 Result: {turn_count2} turns, Winner: Player {winner2}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Game 1 (Naive first): {turn_count1} turns, Winner: Player {winner1}")
    print(f"Game 2 (Particle first): {turn_count2} turns, Winner: Player {winner2}")


if __name__ == "__main__":
    main()
