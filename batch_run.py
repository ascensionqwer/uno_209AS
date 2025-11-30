"""Batch simulation runner for UNO game.

Runs multiple simulations and outputs win statistics.
"""

from src.utils.config_loader import load_config
from src.utils.game_runner import run_matchup_game
from src.utils.matchup_types import PlayerType, Matchup
from src.utils.simulation_logger import SimulationLogger, SimulationResult


def run_matchup_batch(matchup: Matchup, num_simulations: int, config: dict):
    """Run batch simulations for a specific matchup."""
    print(f"\n{'=' * 80}")
    print(f"RUNNING {matchup.player1_type.value.upper()} VS {matchup.player2_type.value.upper()}")
    print(f"{'=' * 80}")
    print(f"Running {num_simulations} simulations...")
    print(f"Player 1: {matchup.player1_type.value}")
    print(f"Player 2: {matchup.player2_type.value}")
    print(f"{'=' * 80}")

    player1_wins = 0
    player2_wins = 0
    no_winner = 0
    total_turns = 0
    min_turns = float("inf")
    max_turns = 0
    games_with_issues = []
    all_decision_times = {1: [], 2: []}
    cache_stats = {"player1": [], "player2": []}

    # Calculate starting player distribution
    player1_starts = (num_simulations + 1) // 2  # First player gets extra if odd

    # Run simulations
    for i in range(num_simulations):
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{num_simulations} games completed")

        try:
            # Determine starting player and create matchup accordingly
            if i < player1_starts:
                # Player 1 starts
                current_matchup = matchup
            else:
                # Player 2 starts - swap players
                current_matchup = Matchup(matchup.player2_type, matchup.player1_type)
            
            turn_count, winner, stats = run_matchup_game(current_matchup, seed=i, show_output=False)

            # Map winner back to original player positions
            if i < player1_starts:
                # Original orientation
                if winner == 1:
                    player1_wins += 1
                elif winner == 2:
                    player2_wins += 1
                else:
                    no_winner += 1
                    games_with_issues.append((i + 1, turn_count, "no_winner"))
            else:
                # Swapped orientation - map back
                if winner == 1:
                    player2_wins += 1  # Player 2 was position 1 in swapped game
                elif winner == 2:
                    player1_wins += 1  # Player 1 was position 2 in swapped game
                else:
                    no_winner += 1
                    games_with_issues.append((i + 1, turn_count, "no_winner"))

            total_turns += turn_count
            min_turns = min(min_turns, turn_count)
            max_turns = max(max_turns, turn_count)

            # Collect decision times
            for player in [1, 2]:
                all_decision_times[player].extend(stats["decision_times"].get(player, []))

            # Collect cache stats
            if "cache_stats" in stats:
                if "player1" in stats["cache_stats"]:
                    cache_stats["player1"].append(stats["cache_stats"]["player1"])
                if "player2" in stats["cache_stats"]:
                    cache_stats["player2"].append(stats["cache_stats"]["player2"])

        except Exception as e:
            print(f"\nERROR in simulation {i + 1}: {e}")
            no_winner += 1
            games_with_issues.append((i + 1, 0, f"error: {str(e)}"))

    # Calculate averages
    avg_decision_times = {}
    for player in [1, 2]:
        if all_decision_times[player]:
            avg_decision_times[f"player{player}"] = sum(all_decision_times[player]) / len(all_decision_times[player])
        else:
            avg_decision_times[f"player{player}"] = 0

    avg_cache_sizes = {}
    for player in ["player1", "player2"]:
        if cache_stats[player]:
            avg_cache_sizes[player] = sum(cache_stats[player]) / len(cache_stats[player])
        else:
            avg_cache_sizes[player] = 0

    # Print results
    print(f"\n{'=' * 80}")
    print(f"{matchup.player1_type.value.upper()} VS {matchup.player2_type.value.upper()} RESULTS")
    print(f"{'=' * 80}")
    print(f"Total simulations: {num_simulations}")
    print()
    print("Win Statistics:")
    print(f"  Player 1 ({matchup.player1_type.value}) wins: {player1_wins} ({100.0 * player1_wins / num_simulations:.2f}%)")
    print(f"  Player 2 ({matchup.player2_type.value}) wins: {player2_wins} ({100.0 * player2_wins / num_simulations:.2f}%)")
    if no_winner > 0:
        print(f"  No winner (safety limit/errors): {no_winner} ({100.0 * no_winner / num_simulations:.2f}%)")
    print()
    print("Turn Statistics:")
    print(f"  Average turns per game: {total_turns / num_simulations:.2f}")
    print(f"  Minimum turns: {min_turns}")
    print(f"  Maximum turns: {max_turns}")
    print()
    print("Decision Time Statistics:")
    for player in [1, 2]:
        player_type = matchup.player1_type.value if player == 1 else matchup.player2_type.value
        print(f"  Player {player} ({player_type}): {avg_decision_times[f'player{player}']:.6f}s avg")
    print()
    print("Cache Statistics:")
    for player in ["player1", "player2"]:
        player_num = 1 if player == "player1" else 2
        player_type = matchup.player1_type.value if player_num == 1 else matchup.player2_type.value
        if player_type == "particle_policy":
            print(f"  Player {player_num} ({player_type}): {avg_cache_sizes[player]:.1f} avg cache size")
        else:
            print(f"  Player {player_num} ({player_type}): N/A (no cache)")
    
    if games_with_issues:
        print()
        print(f"Games with issues: {len(games_with_issues)}")

    # Return results for JSON logging
    return {
        "player_wins": {"player1": player1_wins, "player2": player2_wins},
        "win_rates": {"player1": player1_wins / num_simulations, "player2": player2_wins / num_simulations},
        "avg_decision_times": avg_decision_times,
        "cache_stats": avg_cache_sizes if any(avg_cache_sizes.values()) else None
    }


def main():
    """Run batch simulations for all matchups."""
    config = load_config()
    batch_config = config.get("batch_run", {})
    num_simulations = batch_config.get("num_simulations", 100)

    print("=" * 80)
    print("UNO BATCH SIMULATION RUNNER - ALL MATCHUPS")
    print("=" * 80)

    # Define all matchups
    matchups = [
        Matchup(PlayerType.NAIVE, PlayerType.NAIVE),
        Matchup(PlayerType.NAIVE, PlayerType.PARTICLE_POLICY),
        Matchup(PlayerType.PARTICLE_POLICY, PlayerType.PARTICLE_POLICY),
    ]

    logger = SimulationLogger()
    all_results = {}

    # Run each matchup
    for matchup in matchups:
        result = run_matchup_batch(matchup, num_simulations, config)
        all_results[str(matchup)] = result

    # Log all results to JSON
    simulation_result = SimulationResult(
        timestamp=logger.get_timestamp_filename().split('/')[-1].replace('.json', ''),
        config=config,
        matchup="all_matchups",
        total_games=num_simulations * len(matchups),
        player_wins=all_results,
        win_rates={matchup: result["win_rates"] for matchup, result in all_results.items()},
        avg_decision_times={matchup: result["avg_decision_times"] for matchup, result in all_results.items()},
        cache_stats={matchup: result["cache_stats"] for matchup, result in all_results.items()}
    )
    
    filename = logger.log_results(simulation_result)
    print(f"\n{'=' * 80}")
    print(f"RESULTS LOGGED TO: {filename}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
