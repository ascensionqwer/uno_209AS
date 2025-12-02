"""Batch simulation runner for simplified UNO game (numbered cards only).

Runs baseline simulations with simplified game logic:
- No special cards (Wild, Draw 2, Skip, Reverse)
- Only numbered cards (0-max_number) in max_colors colors
- Configurable deck parameters via config.jsonc
"""

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple
from src.utils.config_loader import load_config
from src.utils.matchup_types import PlayerType, Matchup
from src.utils.simulation_logger import SimulationLogger, SimulationResult
from src.utils.game_runner import run_simplified_matchup_game


def run_single_simulation_black(args: Tuple[int, Matchup, int, int, int]) -> dict:
    """
    Run a single simulation with simplified game.

    Args:
        args: Tuple of (simulation_id, matchup, player1_starts, max_number, max_colors)

    Returns:
        Dict with simulation result
    """
    simulation_id, matchup, player1_starts, max_number, max_colors = args

    # Determine starting player and actual matchup
    if simulation_id < player1_starts:
        # Player 1 starts
        actual_matchup = matchup
        starting_player = 1
    else:
        # Player 2 starts (swap matchup)
        actual_matchup = Matchup(matchup.player2_type, matchup.player1_type)
        starting_player = 2

    # Run game
    seed = simulation_id + int(time.time() * 1000)
    turn_count, winner, stats = run_simplified_matchup_game(
        actual_matchup, max_number, max_colors, seed=seed, show_output=False
    )

    # Map winner back to original player positions if swapped
    if starting_player == 2:
        if winner == 1:
            winner = 2
        elif winner == 2:
            winner = 1

    return {
        "simulation_id": simulation_id,
        "winner": winner,
        "turn_count": turn_count,
        "stats": stats,
    }


def run_matchup_batch_black(
    matchup: Matchup,
    num_simulations: int,
    max_number: int,
    max_colors: int,
    max_workers: int = 100,
):
    """Run batch simulations for a specific matchup using simplified game."""
    print(f"\n{'=' * 80}")
    print(
        f"RUNNING {matchup.player1_type.value.upper()} VS {matchup.player2_type.value.upper()} - {num_simulations} GAMES"
    )
    print(f"{'=' * 80}")
    print(f"Player 1: {matchup.player1_type.value}")
    print(f"Player 2: {matchup.player2_type.value}")
    print(f"Game: Simplified UNO (max_number={max_number}, max_colors={max_colors})")
    if max_workers:
        print(f"Using {max_workers} worker processes")
    print(f"{'=' * 80}")

    # Calculate starting player distribution
    player1_starts = (num_simulations + 1) // 2  # First player gets extra if odd

    # Prepare arguments for each simulation
    simulation_args = [
        (i, matchup, player1_starts, max_number, max_colors)
        for i in range(num_simulations)
    ]

    # Run simulations in parallel
    results = []
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_sim = {
            executor.submit(run_single_simulation_black, args): args[0]
            for args in simulation_args
        }

        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_sim):
            sim_id = future_to_sim[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1

                # Progress reporting every 10 games
                if completed % 10 == 0 or completed == num_simulations:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / completed
                    remaining = (num_simulations - completed) * avg_time
                    print(
                        f"Progress: {completed}/{num_simulations} games "
                        f"({100 * completed // num_simulations}%) - "
                        f"Elapsed: {elapsed:.1f}s, Est. remaining: {remaining:.1f}s"
                    )
            except Exception as exc:
                print(f"Simulation {sim_id} generated an exception: {exc}")

    # Aggregate results
    player1_wins = sum(1 for r in results if r["winner"] == 1)
    player2_wins = sum(1 for r in results if r["winner"] == 2)
    no_winner = sum(1 for r in results if r["winner"] == 0)

    total_turns = sum(r["turn_count"] for r in results)
    avg_turns = total_turns / len(results) if results else 0
    min_turns = min(r["turn_count"] for r in results) if results else 0
    max_turns = max(r["turn_count"] for r in results) if results else 0

    # Aggregate decision times
    decision_times_p1 = []
    decision_times_p2 = []
    for r in results:
        if "stats" in r and "decision_times" in r["stats"]:
            decision_times_p1.extend(r["stats"]["decision_times"].get(1, []))
            decision_times_p2.extend(r["stats"]["decision_times"].get(2, []))

    avg_decision_time_p1 = (
        sum(decision_times_p1) / len(decision_times_p1) if decision_times_p1 else 0
    )
    avg_decision_time_p2 = (
        sum(decision_times_p2) / len(decision_times_p2) if decision_times_p2 else 0
    )

    # Aggregate cache stats
    cache_sizes_p1 = []
    cache_sizes_p2 = []
    for r in results:
        if "stats" in r and "cache_stats" in r["stats"]:
            if "player1" in r["stats"]["cache_stats"]:
                cache_sizes_p1.append(r["stats"]["cache_stats"]["player1"])
            if "player2" in r["stats"]["cache_stats"]:
                cache_sizes_p2.append(r["stats"]["cache_stats"]["player2"])

    avg_cache_size_p1 = (
        sum(cache_sizes_p1) / len(cache_sizes_p1) if cache_sizes_p1 else 0
    )
    avg_cache_size_p2 = (
        sum(cache_sizes_p2) / len(cache_sizes_p2) if cache_sizes_p2 else 0
    )

    elapsed_time = time.time() - start_time

    # Print results
    print(f"\n{'=' * 80}")
    print("RESULTS")
    print(f"{'=' * 80}")
    print(f"Total games: {num_simulations}")
    print(
        f"Player 1 wins: {player1_wins} ({100 * player1_wins / num_simulations:.1f}%)"
    )
    print(
        f"Player 2 wins: {player2_wins} ({100 * player2_wins / num_simulations:.1f}%)"
    )
    print(f"No winner: {no_winner} ({100 * no_winner / num_simulations:.1f}%)")
    print("\nTurn Statistics:")
    print(f"  Average turns: {avg_turns:.1f}")
    print(f"  Min turns: {min_turns}")
    print(f"  Max turns: {max_turns}")
    print("\nDecision Time Statistics:")
    print(f"  Player 1 avg: {avg_decision_time_p1:.4f}s")
    print(f"  Player 2 avg: {avg_decision_time_p2:.4f}s")
    if cache_sizes_p1 or cache_sizes_p2:
        print("\nCache Statistics:")
        if cache_sizes_p1:
            print(f"  Player 1 avg cache size: {avg_cache_size_p1:.1f}")
        if cache_sizes_p2:
            print(f"  Player 2 avg cache size: {avg_cache_size_p2:.1f}")
    print(f"\nTotal time: {elapsed_time:.1f}s")
    print(f"{'=' * 80}")

    return {
        "player_wins": {
            "player1": player1_wins,
            "player2": player2_wins,
            "no_winner": no_winner,
        },
        "win_rates": {
            "player1": player1_wins / num_simulations if num_simulations > 0 else 0,
            "player2": player2_wins / num_simulations if num_simulations > 0 else 0,
            "no_winner": no_winner / num_simulations if num_simulations > 0 else 0,
        },
        "turn_stats": {
            "total": total_turns,
            "average": avg_turns,
            "min": min_turns,
            "max": max_turns,
        },
        "avg_decision_times": {
            "player1": avg_decision_time_p1,
            "player2": avg_decision_time_p2,
        },
        "cache_stats": {
            "player1": avg_cache_size_p1 if cache_sizes_p1 else None,
            "player2": avg_cache_size_p2 if cache_sizes_p2 else None,
        },
    }


def run_baseline_black_simulations():
    """Run baseline simulations for all matchups with simplified game."""
    config = load_config()
    batch_config = config.get("batch_black", {})

    max_number = batch_config.get("max_number", 9)
    max_colors = batch_config.get("max_colors", 4)
    num_simulations = batch_config.get("num_simulations", 300)

    print("=" * 80)
    print("UNO SIMPLIFIED (BLACK) BASELINE SIMULATION RUNNER")
    print("=" * 80)
    print("Game: Simplified UNO (numbered cards only)")
    print(f"  max_number: {max_number} (cards 0-{max_number})")
    print(f"  max_colors: {max_colors}")
    deck_size = max_colors * (1 + 2 * max_number)
    print(f"  deck size: {deck_size} cards")
    print(f"Running {num_simulations} simulations per matchup")
    print("=" * 80)

    # Define all matchups
    matchups = [
        Matchup(PlayerType.NAIVE, PlayerType.NAIVE),
        Matchup(PlayerType.NAIVE, PlayerType.PARTICLE_POLICY),
        Matchup(PlayerType.PARTICLE_POLICY, PlayerType.PARTICLE_POLICY),
    ]

    logger = SimulationLogger()
    all_results = {}
    total_games = 0

    # Run each matchup with simplified game
    for matchup in matchups:
        print(
            f"\n--- {matchup.player1_type.value.upper()} vs {matchup.player2_type.value.upper()} ---"
        )
        result = run_matchup_batch_black(
            matchup, num_simulations, max_number, max_colors
        )
        all_results[str(matchup)] = result
        total_games += num_simulations

    # Log all results to JSON
    simulation_result = SimulationResult(
        timestamp=logger.get_timestamp_filename().split("/")[-1].replace(".json", ""),
        config=config,
        matchup=f"simplified_baseline_max{max_number}_colors{max_colors}",
        total_games=total_games,
        player_wins={k: v["player_wins"] for k, v in all_results.items()},
        win_rates={k: v["win_rates"] for k, v in all_results.items()},
        avg_decision_times={k: v["avg_decision_times"] for k, v in all_results.items()},
        cache_stats={k: v["cache_stats"] for k, v in all_results.items()},
        parameter_variants={
            "simplified_game": {
                "max_number": max_number,
                "max_colors": max_colors,
                "deck_size": deck_size,
            }
        },
    )

    filename = logger.log_results(simulation_result)
    print(f"\n{'=' * 80}")
    print(f"RESULTS LOGGED TO: {filename}")
    print(f"TOTAL GAMES PLAYED: {total_games}")
    print(f"{'=' * 80}")

    return all_results


if __name__ == "__main__":
    run_baseline_black_simulations()
