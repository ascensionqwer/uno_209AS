from src.utils.game_runner import run_matchup_game
from src.utils.matchup_types import PlayerType, Matchup


def main():
    """Run 1 particle vs particle UNO game."""
    print("=" * 60)
    print("UNO Simulation: Particle Policy vs Particle Policy (1 game)")
    print("=" * 60)

    matchup = Matchup(PlayerType.PARTICLE_POLICY, PlayerType.PARTICLE_POLICY)
    turn_count, winner, stats = run_matchup_game(matchup, seed=42, show_output=True)

    print(f"\nResult: {turn_count} turns, Winner: Player {winner}")
    if stats.get("cache_stats"):
        print(f"Player 1 cache size: {stats['cache_stats'].get('player1', 'N/A')}")
        print(f"Player 2 cache size: {stats['cache_stats'].get('player2', 'N/A')}")

    # Check if game ended due to infinite loop detection
    if turn_count >= 100:  # Arbitrary threshold for suspiciously long games
        print("WARNING: Game was unusually long - possible infinite loop detected!")


if __name__ == "__main__":
    main()
