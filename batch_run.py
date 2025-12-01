"""Batch simulation runner for UNO game.

Runs multiple simulations and outputs win statistics.
"""

import concurrent.futures
from typing import Tuple, Dict, Any
from src.utils.config_loader import load_config
from src.utils.game_runner import run_matchup_game, run_matchup_game_with_configs
from src.utils.matchup_types import PlayerType, Matchup
from src.utils.simulation_logger import SimulationLogger, SimulationResult
from src.utils.config_variator import ConfigVariator


def run_single_simulation(
    args: Tuple[int, Matchup, int],
) -> Tuple[int, int, int, Dict[str, Any]]:
    """Run a single simulation for multithreaded execution.

    Args:
        args: Tuple of (simulation_id, matchup, player1_starts)

    Returns:
        Tuple of (simulation_id, winner, turn_count, stats)
    """
    simulation_id, matchup, player1_starts = args

    try:
        # Determine starting player and create matchup accordingly
        if simulation_id < player1_starts:
            # Player 1 starts
            current_matchup = matchup
        else:
            # Player 2 starts - swap players
            current_matchup = Matchup(matchup.player2_type, matchup.player1_type)

        turn_count, winner, stats = run_matchup_game(
            current_matchup, seed=simulation_id, show_output=False
        )

        return simulation_id, winner, turn_count, stats

    except Exception as e:
        # Return error result
        return simulation_id, 0, 0, {"error": str(e)}


def run_matchup_batch(
    matchup: Matchup, num_simulations: int, config: dict, max_workers: int = 100
):
    """Run batch simulations for a specific matchup using multithreading."""
    print(f"\n{'=' * 80}")
    print(
        f"RUNNING {matchup.player1_type.value.upper()} VS {matchup.player2_type.value.upper()} - {num_simulations} GAMES"
    )
    print(f"{'=' * 80}")
    print(f"Player 1: {matchup.player1_type.value}")
    print(f"Player 2: {matchup.player2_type.value}")
    if max_workers:
        print(f"Using {max_workers} worker processes")
    print(f"{'=' * 80}")

    # Calculate starting player distribution
    player1_starts = (num_simulations + 1) // 2  # First player gets extra if odd

    # Prepare arguments for each simulation
    simulation_args = [(i, matchup, player1_starts) for i in range(num_simulations)]

    # Run simulations in parallel
    results = []
    completed_count = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all simulations
        future_to_sim = {
            executor.submit(run_single_simulation, args): args[0]
            for args in simulation_args
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_sim):
            sim_id = future_to_sim[future]
            try:
                result = future.result()
                results.append(result)
                completed_count += 1

                # Progress reporting with context
                if completed_count % 10 == 0 or completed_count == num_simulations:
                    print(
                        f"Progress: {completed_count}/{num_simulations} games completed ({matchup.player1_type.value} vs {matchup.player2_type.value})"
                    )

            except Exception as e:
                print(f"\nERROR in simulation {sim_id}: {e}")
                results.append((sim_id, 0, 0, {"error": str(e)}))

    # Process results
    player1_wins = 0
    player2_wins = 0
    no_winner = 0
    total_turns = 0
    min_turns = float("inf")
    max_turns = 0
    games_with_issues = []
    all_decision_times = {1: [], 2: []}
    cache_stats = {"player1": [], "player2": []}

    for simulation_id, winner, turn_count, stats in results:
        # Map winner back to original player positions
        if simulation_id < player1_starts:
            # Original orientation
            if winner == 1:
                player1_wins += 1
            elif winner == 2:
                player2_wins += 1
            else:
                no_winner += 1
                games_with_issues.append((simulation_id + 1, turn_count, "no_winner"))
        else:
            # Swapped orientation - map back
            if winner == 1:
                player2_wins += 1  # Player 2 was position 1 in swapped game
            elif winner == 2:
                player1_wins += 1  # Player 1 was position 2 in swapped game
            else:
                no_winner += 1
                games_with_issues.append((simulation_id + 1, turn_count, "no_winner"))

        total_turns += turn_count
        min_turns = min(min_turns, turn_count)
        max_turns = max(max_turns, turn_count)

        # Collect decision times
        for player in [1, 2]:
            all_decision_times[player].extend(
                stats.get("decision_times", {}).get(player, [])
            )

        # Collect cache stats
        if "cache_stats" in stats:
            if "player1" in stats["cache_stats"]:
                cache_stats["player1"].append(stats["cache_stats"]["player1"])
            if "player2" in stats["cache_stats"]:
                cache_stats["player2"].append(stats["cache_stats"]["player2"])

    # Calculate averages
    avg_decision_times = {}
    for player in [1, 2]:
        if all_decision_times[player]:
            avg_decision_times[f"player{player}"] = sum(
                all_decision_times[player]
            ) / len(all_decision_times[player])
        else:
            avg_decision_times[f"player{player}"] = 0

    avg_cache_sizes = {}
    for player in ["player1", "player2"]:
        if cache_stats[player]:
            avg_cache_sizes[player] = sum(cache_stats[player]) / len(
                cache_stats[player]
            )
        else:
            avg_cache_sizes[player] = 0

    # Print results
    print(f"\n{'=' * 80}")
    print(
        f"{matchup.player1_type.value.upper()} VS {matchup.player2_type.value.upper()} RESULTS"
    )
    print(f"{'=' * 80}")
    print(f"Total simulations: {num_simulations}")
    print()
    print("Win Statistics:")
    print(
        f"  Player 1 ({matchup.player1_type.value}) wins: {player1_wins} ({100.0 * player1_wins / num_simulations:.2f}%)"
    )
    print(
        f"  Player 2 ({matchup.player2_type.value}) wins: {player2_wins} ({100.0 * player2_wins / num_simulations:.2f}%)"
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
    print()
    print("Decision Time Statistics:")
    for player in [1, 2]:
        player_type = (
            matchup.player1_type.value if player == 1 else matchup.player2_type.value
        )
        print(
            f"  Player {player} ({player_type}): {avg_decision_times[f'player{player}']:.6f}s avg"
        )
    print()
    print("Cache Statistics:")
    for player in ["player1", "player2"]:
        player_num = 1 if player == "player1" else 2
        player_type = (
            matchup.player1_type.value
            if player_num == 1
            else matchup.player2_type.value
        )
        if player_type == "particle_policy":
            print(
                f"  Player {player_num} ({player_type}): {avg_cache_sizes[player]:.1f} avg cache size"
            )
        else:
            print(f"  Player {player_num} ({player_type}): N/A (no cache)")

    if games_with_issues:
        print()
        print(f"Games with issues: {len(games_with_issues)}")

    # Return results for JSON logging
    return {
        "player_wins": {"player1": player1_wins, "player2": player2_wins},
        "win_rates": {
            "player1": player1_wins / num_simulations,
            "player2": player2_wins / num_simulations,
        },
        "avg_decision_times": avg_decision_times,
        "cache_stats": avg_cache_sizes if any(avg_cache_sizes.values()) else None,
    }


def run_comprehensive_analysis():
    """Run comprehensive analysis with 100+ games per configuration."""
    config = load_config()
    batch_config = config.get("batch_run", {})
    num_simulations = batch_config.get("num_simulations", 100)

    print("=" * 80)
    print("UNO COMPREHENSIVE WIN RATE ANALYSIS")
    print("=" * 80)
    print(f"Running {num_simulations} simulations per configuration...")

    variator = ConfigVariator("config.jsonc")
    num_variants = batch_config.get("num_variants", 5)
    variants = variator.generate_variants(0.25)[:num_variants]

    logger = SimulationLogger()
    all_results = {}
    total_games = 0

    # Test each variant with comprehensive simulations
    for i, (variant_config, variant_info) in enumerate(variants):
        print(f"\n{'=' * 80}")
        print(f"COMPREHENSIVE TESTING VARIANT {i + 1}/{len(variants)}")
        print(f"{'=' * 80}")

        variant_desc = variant_info[0]
        print(f"Parameter: {variant_desc.parameter_path}")
        print(
            f"Change: {variant_desc.variation_type} {variant_desc.percentage_change * 100:.0f}%"
        )
        print(f"Original: {variant_desc.original_value}")
        print(f"Variant: {variant_desc.variant_value}")

        # Run Particle vs Naive with comprehensive simulations
        matchup = Matchup(PlayerType.PARTICLE_POLICY, PlayerType.NAIVE)
        print(
            f"\n--- COMPREHENSIVE VARIANT {i + 1}/{len(variants)} - Particle Policy vs Naive ---"
        )
        result = run_matchup_batch(matchup, num_simulations, variant_config)
        total_games += num_simulations

        variant_key = f"comprehensive_variant_{i + 1}_{variant_desc.parameter_path.replace('.', '_')}_{variant_desc.variation_type}"
        all_results[variant_key] = {
            "result": result,
            "variant_info": {
                "parameter_path": variant_desc.parameter_path,
                "original_value": variant_desc.original_value,
                "variant_value": variant_desc.variant_value,
                "variation_type": variant_desc.variation_type,
                "percentage_change": variant_desc.percentage_change,
            },
        }

        print(f"Win Rate: {result['win_rates']['player1']:.2%}")
        print(f"Avg Decision Time: {result['avg_decision_times']['player1']:.6f}s")

    # Log comprehensive results
    simulation_result = SimulationResult(
        timestamp=logger.get_timestamp_filename().split("/")[-1].replace(".json", ""),
        config=config,
        matchup="comprehensive_win_rate_analysis",
        total_games=total_games,
        player_wins={k: v["result"]["player_wins"] for k, v in all_results.items()},
        win_rates={k: v["result"]["win_rates"] for k, v in all_results.items()},
        avg_decision_times={
            k: v["result"]["avg_decision_times"] for k, v in all_results.items()
        },
        cache_stats={k: v["result"]["cache_stats"] for k, v in all_results.items()},
        parameter_variants={k: v["variant_info"] for k, v in all_results.items()},
    )

    filename = logger.log_results(simulation_result)
    print(f"\n{'=' * 80}")
    print(f"COMPREHENSIVE RESULTS LOGGED TO: {filename}")
    print(f"TOTAL GAMES PLAYED: {total_games}")
    print(f"{'=' * 80}")

    return all_results


def run_parameter_sensitivity_analysis():
    """Run parameter sensitivity analysis with config variants."""
    config = load_config()
    batch_config = config.get("batch_run", {})
    num_simulations = batch_config.get("num_simulations", 10)

    print("=" * 80)
    print("UNO PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 80)

    variator = ConfigVariator("config.jsonc")
    num_variants = batch_config.get("num_variants", 5)
    variants = variator.generate_variants(0.25)[:num_variants]

    logger = SimulationLogger()
    all_results = {}
    total_games = 0

    # Test each variant across three matchup types
    for i, (variant_config, variant_info) in enumerate(variants):
        print(f"\n{'=' * 80}")
        print(f"TESTING VARIANT {i + 1}/{len(variants)}")
        print(f"{'=' * 80}")

        variant_desc = variant_info[0]
        print(f"Parameter: {variant_desc.parameter_path}")
        print(
            f"Change: {variant_desc.variation_type} {variant_desc.percentage_change * 100:.0f}%"
        )
        print(f"Original: {variant_desc.original_value}")
        print(f"Variant: {variant_desc.variant_value}")

        # Test three matchup types
        matchups = [
            (
                "particle_vs_naive",
                Matchup(PlayerType.PARTICLE_POLICY, PlayerType.NAIVE),
                variant_config,
            ),
            (
                "particle_vs_particle_same",
                Matchup(PlayerType.PARTICLE_POLICY, PlayerType.PARTICLE_POLICY),
                variant_config,
            ),
            (
                "particle_vs_particle_mixed",
                Matchup(PlayerType.PARTICLE_POLICY, PlayerType.PARTICLE_POLICY),
                "mixed",
            ),
        ]

        variant_results = {}

        for matchup_name, matchup, config_to_use in matchups:
            print(
                f"\n--- VARIANT {i + 1}/{len(variants)} - {matchup_name.upper().replace('_', ' ')} ---"
            )

            if config_to_use == "mixed":
                # For mixed config, we need to modify game_runner to handle different configs per player
                # For now, use variant config for player 1 and base config for player 2
                result = run_mixed_config_matchup(
                    matchup, num_simulations, variant_config, config
                )
            else:
                result = run_matchup_batch(matchup, num_simulations, config_to_use)

            variant_results[matchup_name] = result
            total_games += num_simulations

        variant_key = f"variant_{i + 1}_{variant_desc.parameter_path.replace('.', '_')}_{variant_desc.variation_type}"
        all_results[variant_key] = {
            "results": variant_results,
            "variant_info": {
                "parameter_path": variant_desc.parameter_path,
                "original_value": variant_desc.original_value,
                "variant_value": variant_desc.variant_value,
                "variation_type": variant_desc.variation_type,
                "percentage_change": variant_desc.percentage_change,
            },
        }

    # Log all results to JSON
    simulation_result = SimulationResult(
        timestamp=logger.get_timestamp_filename().split("/")[-1].replace(".json", ""),
        config=config,
        matchup="parameter_sensitivity_analysis",
        total_games=total_games,
        player_wins={
            k: v["results"]["particle_vs_naive"]["player_wins"]
            for k, v in all_results.items()
        },
        win_rates={
            k: f"particle_vs_naive: {v['results']['particle_vs_naive']['win_rates']}"
            for k, v in all_results.items()
        },
        avg_decision_times={
            k: f"particle_vs_naive: {v['results']['particle_vs_naive']['avg_decision_times']}"
            for k, v in all_results.items()
        },
        cache_stats={
            k: f"particle_vs_naive: {v['results']['particle_vs_naive']['cache_stats']}"
            for k, v in all_results.items()
        },
        parameter_variants={k: v["variant_info"] for k, v in all_results.items()},
    )

    filename = logger.log_results(simulation_result)
    print(f"\n{'=' * 80}")
    print(f"PARAMETER SENSITIVITY RESULTS LOGGED TO: {filename}")
    print(f"TOTAL GAMES PLAYED: {total_games}")
    print(f"{'=' * 80}")

    return all_results


def run_mixed_config_matchup(
    matchup: Matchup, num_simulations: int, config1: dict, config2: dict
):
    """Run matchup with different configs for each player."""
    print(f"\n{'=' * 80}")
    print(
        f"RUNNING MIXED CONFIG {matchup.player1_type.value.upper()} VS {matchup.player2_type.value.upper()} - {num_simulations} GAMES"
    )
    print(f"{'=' * 80}")
    print(f"Player 1: {matchup.player1_type.value} (Variant Config)")
    print(f"Player 2: {matchup.player2_type.value} (Base Config)")
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
    player1_starts = (num_simulations + 1) // 2

    # Run simulations
    for i in range(num_simulations):
        if (i + 1) % 10 == 0:
            print(
                f"Progress: {i + 1}/{num_simulations} games completed (Mixed Config - {matchup.player1_type.value} vs {matchup.player2_type.value})"
            )

        try:
            # Determine starting player and create matchup accordingly
            if i < player1_starts:
                current_matchup = matchup
                configs = (config1, config2)  # Player 1 gets variant config
            else:
                current_matchup = Matchup(matchup.player2_type, matchup.player1_type)
                configs = (
                    config2,
                    config1,
                )  # Player 2 gets variant config (swapped position)

            turn_count, winner, stats = run_matchup_game_with_configs(
                current_matchup, configs, seed=i, show_output=False
            )

            # Map winner back to original player positions
            if i < player1_starts:
                if winner == 1:
                    player1_wins += 1
                elif winner == 2:
                    player2_wins += 1
                else:
                    no_winner += 1
                    games_with_issues.append((i + 1, turn_count, "no_winner"))
            else:
                if winner == 1:
                    player2_wins += 1
                elif winner == 2:
                    player1_wins += 1
                else:
                    no_winner += 1
                    games_with_issues.append((i + 1, turn_count, "no_winner"))

            total_turns += turn_count
            min_turns = min(min_turns, turn_count)
            max_turns = max(max_turns, turn_count)

            # Collect decision times
            for player in [1, 2]:
                all_decision_times[player].extend(
                    stats["decision_times"].get(player, [])
                )

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
            avg_decision_times[f"player{player}"] = sum(
                all_decision_times[player]
            ) / len(all_decision_times[player])
        else:
            avg_decision_times[f"player{player}"] = 0

    avg_cache_sizes = {}
    for player in ["player1", "player2"]:
        if cache_stats[player]:
            avg_cache_sizes[player] = sum(cache_stats[player]) / len(
                cache_stats[player]
            )
        else:
            avg_cache_sizes[player] = 0

    # Print results
    print(f"\n{'=' * 80}")
    print(f"MIXED CONFIG RESULTS")
    print(f"{'=' * 80}")
    print(f"Total simulations: {num_simulations}")
    print()
    print("Win Statistics:")
    print(
        f"  Player 1 (Variant): {player1_wins} ({100.0 * player1_wins / num_simulations:.2f}%)"
    )
    print(
        f"  Player 2 (Base): {player2_wins} ({100.0 * player2_wins / num_simulations:.2f}%)"
    )
    if no_winner > 0:
        print(
            f"  No winner (safety limit/errors): {no_winner} ({100.0 * no_winner / num_simulations:.2f}%)"
        )

    return {
        "player_wins": {"player1": player1_wins, "player2": player2_wins},
        "win_rates": {
            "player1": player1_wins / num_simulations,
            "player2": player2_wins / num_simulations,
        },
        "avg_decision_times": avg_decision_times,
        "cache_stats": avg_cache_sizes if any(avg_cache_sizes.values()) else None,
    }


def main():
    """Run batch simulations for all matchups with variants."""
    config = load_config()
    batch_config = config.get("batch_run", {})
    num_simulations = batch_config.get("num_simulations", 100)
    num_variants = batch_config.get("num_variants", 5)

    # Generate variants
    variator = ConfigVariator("config.jsonc")
    variants = variator.generate_variants(0.25)[:num_variants]

    # Define all matchups
    matchups = [
        Matchup(PlayerType.NAIVE, PlayerType.NAIVE),
        Matchup(PlayerType.NAIVE, PlayerType.PARTICLE_POLICY),
        Matchup(PlayerType.PARTICLE_POLICY, PlayerType.PARTICLE_POLICY),
    ]

    print("=" * 80)
    print("UNO BATCH SIMULATION RUNNER - ALL MATCHUPS WITH VARIANTS")
    print("=" * 80)
    print(f"Configuration: {num_simulations} simulations per variant")
    print(f"Testing {num_variants} variants per parameter")
    print(f"Total variants to test: {len(variants)}")
    print(f"Matchups per variant: {len(matchups)}")
    print(f"Estimated total games: {len(variants) * len(matchups) * num_simulations}")
    print("=" * 80)

    logger = SimulationLogger()
    all_results = {}
    total_games = 0

    # Test each variant with all matchups
    for i, (variant_config, variant_info) in enumerate(variants):
        print(f"\n{'=' * 80}")
        print(f"TESTING VARIANT {i + 1}/{len(variants)}")
        print(f"{'=' * 80}")

        variant_desc = variant_info[0]
        print(f"Parameter: {variant_desc.parameter_path}")
        print(
            f"Change: {variant_desc.variation_type} {variant_desc.percentage_change * 100:.0f}%"
        )
        print(f"Original: {variant_desc.original_value}")
        print(f"Variant: {variant_desc.variant_value}")

        variant_results = {}

        # Run each matchup with this variant
        for matchup in matchups:
            print(
                f"\n--- VARIANT {i + 1}/{len(variants)} - {matchup.player1_type.value} vs {matchup.player2_type.value} ---"
            )
            result = run_matchup_batch(matchup, num_simulations, variant_config)
            variant_results[str(matchup)] = result
            total_games += num_simulations

        variant_key = f"variant_{i + 1}_{variant_desc.parameter_path.replace('.', '_')}_{variant_desc.variation_type}"
        all_results[variant_key] = {
            "results": variant_results,
            "variant_info": {
                "parameter_path": variant_desc.parameter_path,
                "original_value": variant_desc.original_value,
                "variant_value": variant_desc.variant_value,
                "variation_type": variant_desc.variation_type,
                "percentage_change": variant_desc.percentage_change,
            },
        }

    # Log all results to JSON
    simulation_result = SimulationResult(
        timestamp=logger.get_timestamp_filename().split("/")[-1].replace(".json", ""),
        config=config,
        matchup="all_matchups_with_variants",
        total_games=total_games,
        player_wins={k: v["results"] for k, v in all_results.items()},
        win_rates={
            k: {m: r["win_rates"] for m, r in v["results"].items()}
            for k, v in all_results.items()
        },
        avg_decision_times={
            k: {m: r["avg_decision_times"] for m, r in v["results"].items()}
            for k, v in all_results.items()
        },
        cache_stats={
            k: {m: r["cache_stats"] for m, r in v["results"].items()}
            for k, v in all_results.items()
        },
        parameter_variants={k: v["variant_info"] for k, v in all_results.items()},
    )

    filename = logger.log_results(simulation_result)
    print(f"\n{'=' * 80}")
    print(f"RESULTS LOGGED TO: {filename}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--sensitivity":
            run_parameter_sensitivity_analysis()
        elif sys.argv[1] == "--comprehensive":
            run_comprehensive_analysis()
        else:
            print("Usage: python batch_run.py [--sensitivity|--comprehensive]")
    else:
        main()
