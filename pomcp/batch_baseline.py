"""Baseline simulation runner for UNO game.

Runs baseline simulations (0% change, default config) for all matchups.
"""

from src.utils.config_loader import load_config
from src.utils.matchup_types import PlayerType, Matchup
from src.utils.simulation_logger import SimulationLogger, SimulationResult
from src.utils.config_variator import ConfigVariator
from batch_run import run_matchup_batch


def run_baseline_simulations():
    """Run baseline simulations for all matchups with default config."""
    config = load_config()
    batch_config = config.get("batch_run", {})
    # Only use num_simulations from batch_run settings (num_variants doesn't apply to baseline)
    num_simulations = batch_config.get("num_simulations", 100)

    print("=" * 80)
    print("UNO BASELINE SIMULATION RUNNER")
    print("=" * 80)
    print(
        f"Running {num_simulations} simulations per matchup with baseline (0% change) config"
    )
    print("=" * 80)

    # Generate baseline variant
    variator = ConfigVariator("config.jsonc")
    baseline_config, baseline_variants = variator.generate_baseline()

    # Define all matchups
    matchups = [
        Matchup(PlayerType.NAIVE, PlayerType.NAIVE),
        Matchup(PlayerType.NAIVE, PlayerType.PARTICLE_POLICY),
        Matchup(PlayerType.PARTICLE_POLICY, PlayerType.PARTICLE_POLICY),
    ]

    logger = SimulationLogger()
    all_results = {}
    total_games = 0

    print("\nBaseline Configuration:")
    for variant in baseline_variants:
        print(f"  {variant.parameter_path}: {variant.original_value} (baseline)")

    # Run each matchup with baseline config
    for matchup in matchups:
        print(
            f"\n--- BASELINE - {matchup.player1_type.value.upper()} vs {matchup.player2_type.value.upper()} ---"
        )
        result = run_matchup_batch(matchup, num_simulations, baseline_config)
        all_results[str(matchup)] = result
        total_games += num_simulations

    # Create variant info for baseline
    variant_info = {}
    for variant in baseline_variants:
        variant_info[variant.parameter_path] = {
            "parameter_path": variant.parameter_path,
            "original_value": variant.original_value,
            "variant_value": variant.variant_value,
            "variation_type": variant.variation_type,
            "percentage_change": variant.percentage_change,
        }

    # Log all results to JSON
    simulation_result = SimulationResult(
        timestamp=logger.get_timestamp_filename().split("/")[-1].replace(".json", ""),
        config=config,
        matchup="baseline",
        total_games=total_games,
        player_wins={k: v["player_wins"] for k, v in all_results.items()},
        win_rates={k: v["win_rates"] for k, v in all_results.items()},
        avg_decision_times={k: v["avg_decision_times"] for k, v in all_results.items()},
        cache_stats={k: v["cache_stats"] for k, v in all_results.items()},
        parameter_variants={"baseline": variant_info},
    )

    filename = logger.log_results(simulation_result)
    print(f"\n{'=' * 80}")
    print(f"BASELINE RESULTS LOGGED TO: {filename}")
    print(f"TOTAL GAMES PLAYED: {total_games}")
    print(f"{'=' * 80}")

    return all_results


if __name__ == "__main__":
    run_baseline_simulations()
