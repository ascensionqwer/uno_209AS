#!/usr/bin/env python3
"""Results analysis script for simplified UNO (batch_black) simulation logs.

Parses JSON logs from results/ directory and prints formatted summaries for
simplified game (numbered cards only, no special cards).
"""

import json
import os
import glob
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np


class ResultsBlackAnalyzer:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.logs = []
        self.black_logs = []
        self.baseline_logs = []

    def load_logs(self) -> List[Dict[str, Any]]:
        """Load all JSON logs from results directory."""
        if not os.path.exists(self.results_dir):
            print(f"Results directory '{self.results_dir}' not found.")
            return []

        json_files = glob.glob(os.path.join(self.results_dir, "*.json"))
        if not json_files:
            print(f"No JSON files found in '{self.results_dir}'.")
            return []

        logs = []
        for file_path in sorted(json_files):
            try:
                with open(file_path, "r") as f:
                    log_data = json.load(f)
                    log_data["file_path"] = file_path
                    logs.append(log_data)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        self.logs = logs

        # Separate black (simplified) and baseline (standard) logs
        for log in logs:
            matchup_type = log.get("matchup", "")
            if "simplified_baseline" in matchup_type:
                self.black_logs.append(log)
            elif matchup_type == "baseline":
                self.baseline_logs.append(log)

        return logs

    def print_summary(self):
        """Print formatted summary of simplified game results."""
        if not self.black_logs:
            print("No simplified game (batch_black) logs found.")
            print("Run: python batch_black.py")
            return

        print("=" * 80)
        print("SIMPLIFIED UNO (BATCH_BLACK) RESULTS ANALYSIS")
        print("=" * 80)
        print(f"Found {len(self.black_logs)} simplified game log(s)")
        print()

        self._print_black_results(self.black_logs)

        # Print comparison with standard baseline if available
        if self.baseline_logs:
            print("\n" + "=" * 80)
            print("COMPARISON: SIMPLIFIED vs STANDARD UNO")
            print("=" * 80)
            self._print_comparison()

    def _print_black_results(self, logs: List[Dict[str, Any]]):
        """Print results for simplified game simulations."""
        print("SIMPLIFIED UNO BASELINE RESULTS")
        print("=" * 50)

        for log in logs:
            timestamp = log.get("timestamp", "Unknown")
            total_games = log.get("total_games", 0)

            # Extract game parameters
            param_variants = log.get("parameter_variants", {})
            simplified_params = param_variants.get("simplified_game", {})
            max_number = simplified_params.get("max_number", "N/A")
            max_colors = simplified_params.get("max_colors", "N/A")
            deck_size = simplified_params.get("deck_size", "N/A")

            print(f"\nTimestamp: {timestamp}")
            print(f"Total Games: {total_games}")
            print("Game Configuration:")
            print(f"  Max Number: {max_number} (cards 0-{max_number})")
            print(f"  Max Colors: {max_colors}")
            print(f"  Deck Size: {deck_size} cards")
            print("-" * 40)

            # Print win rates by matchup
            win_rates = log.get("win_rates", {})
            player_wins = log.get("player_wins", {})

            if win_rates:
                print("\nWin Rates by Matchup:")
                for matchup_key in sorted(win_rates.keys()):
                    rates = win_rates[matchup_key]
                    wins = player_wins.get(matchup_key, {})

                    print(f"\n  {matchup_key}:")
                    if isinstance(rates, dict):
                        p1_rate = rates.get("player1", 0)
                        p2_rate = rates.get("player2", 0)
                        no_winner_rate = rates.get("no_winner", 0)
                        p1_wins = (
                            wins.get("player1", 0) if isinstance(wins, dict) else 0
                        )
                        p2_wins = (
                            wins.get("player2", 0) if isinstance(wins, dict) else 0
                        )
                        no_winner = (
                            wins.get("no_winner", 0) if isinstance(wins, dict) else 0
                        )

                        print(f"    Player 1: {p1_rate:.2%} ({p1_wins} wins)")
                        print(f"    Player 2: {p2_rate:.2%} ({p2_wins} wins)")
                        if no_winner > 0:
                            print(
                                f"    No Winner: {no_winner_rate:.2%} ({no_winner} games)"
                            )

            # Print decision times
            decision_times = log.get("avg_decision_times", {})
            if decision_times:
                print("\n\nDecision Times (seconds):")
                for matchup_key in sorted(decision_times.keys()):
                    times = decision_times[matchup_key]
                    print(f"\n  {matchup_key}:")
                    if isinstance(times, dict):
                        for player, time_val in times.items():
                            if time_val > 0:
                                print(f"    {player}: {time_val:.6f}s")

            # Print cache stats
            cache_stats = log.get("cache_stats", {})
            if cache_stats:
                print("\n\nCache Statistics:")
                has_cache = False
                for matchup_key in sorted(cache_stats.keys()):
                    stats = cache_stats[matchup_key]
                    if isinstance(stats, dict):
                        matchup_has_cache = False
                        cache_output = []
                        for player, cache_size in stats.items():
                            if (
                                cache_size
                                and cache_size != "N/A (no cache)"
                                and cache_size > 0
                            ):
                                cache_output.append(f"    {player}: {cache_size:.1f}")
                                matchup_has_cache = True
                        if matchup_has_cache:
                            if not has_cache:
                                has_cache = True
                            print(f"\n  {matchup_key}:")
                            for line in cache_output:
                                print(line)

            # Print turn statistics if available
            # Note: This isn't in the current SimulationResult structure but could be added
            print()

    def _print_comparison(self):
        """Print comparison between simplified and standard UNO."""
        if not self.black_logs or not self.baseline_logs:
            return

        # Get most recent of each type
        black_log = self.black_logs[-1]
        baseline_log = self.baseline_logs[-1]

        black_wins = black_log.get("win_rates", {})
        baseline_wins = baseline_log.get("win_rates", {})

        black_times = black_log.get("avg_decision_times", {})
        baseline_times = baseline_log.get("avg_decision_times", {})

        # Compare each matchup type
        matchup_types = [
            "PlayerType.NAIVE vs PlayerType.NAIVE",
            "PlayerType.NAIVE vs PlayerType.PARTICLE_POLICY",
            "PlayerType.PARTICLE_POLICY vs PlayerType.PARTICLE_POLICY",
        ]

        print("\nWin Rate Comparison:")
        print("-" * 60)
        for matchup in matchup_types:
            black_rate = black_wins.get(matchup, {})
            baseline_rate = baseline_wins.get(matchup, {})

            if black_rate and baseline_rate:
                print(f"\n{matchup}:")
                if isinstance(black_rate, dict) and isinstance(baseline_rate, dict):
                    p1_black = black_rate.get("player1", 0)
                    p1_baseline = baseline_rate.get("player1", 0)
                    p1_diff = p1_black - p1_baseline

                    p2_black = black_rate.get("player2", 0)
                    p2_baseline = baseline_rate.get("player2", 0)
                    p2_diff = p2_black - p2_baseline

                    print("  Player 1 Win Rate:")
                    print(f"    Simplified: {p1_black:.2%}")
                    print(f"    Standard:   {p1_baseline:.2%}")
                    print(f"    Difference: {p1_diff:+.2%}")

                    print("  Player 2 Win Rate:")
                    print(f"    Simplified: {p2_black:.2%}")
                    print(f"    Standard:   {p2_baseline:.2%}")
                    print(f"    Difference: {p2_diff:+.2%}")

        print("\n\nDecision Time Comparison:")
        print("-" * 60)
        for matchup in matchup_types:
            black_time = black_times.get(matchup, {})
            baseline_time = baseline_times.get(matchup, {})

            if black_time and baseline_time:
                print(f"\n{matchup}:")
                if isinstance(black_time, dict) and isinstance(baseline_time, dict):
                    for player in ["player1", "player2"]:
                        t_black = black_time.get(player, 0)
                        t_baseline = baseline_time.get(player, 0)
                        if t_black > 0 and t_baseline > 0:
                            speedup = t_baseline / t_black
                            print(f"  {player}:")
                            print(f"    Simplified: {t_black:.6f}s")
                            print(f"    Standard:   {t_baseline:.6f}s")
                            print(f"    Speedup:    {speedup:.2f}x")

    def create_visualizations(self, output_dir: str = "results/plots"):
        """Create visualization plots for simplified game analysis."""
        if not self.black_logs:
            print("No simplified game logs to visualize.")
            return

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Create plots
        self._plot_win_rates(output_dir)

        if self.baseline_logs:
            self._plot_comparison(output_dir)

        print(f"\nVisualizations saved to: {output_dir}/")

    def _plot_win_rates(self, output_dir: str):
        """Create win rate bar chart for simplified game."""
        if not self.black_logs:
            return

        log = self.black_logs[-1]
        win_rates = log.get("win_rates", {})
        param_variants = log.get("parameter_variants", {})
        simplified_params = param_variants.get("simplified_game", {})
        max_number = simplified_params.get("max_number", "N/A")
        max_colors = simplified_params.get("max_colors", "N/A")

        matchup_types = sorted(win_rates.keys())
        if not matchup_types:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        x_pos = np.arange(len(matchup_types))
        width = 0.35

        player1_rates = []
        player2_rates = []

        for matchup in matchup_types:
            rates = win_rates[matchup]
            if isinstance(rates, dict):
                player1_rates.append(rates.get("player1", 0))
                player2_rates.append(rates.get("player2", 0))
            else:
                player1_rates.append(0)
                player2_rates.append(0)

        ax.bar(x_pos - width / 2, player1_rates, width, label="Player 1", alpha=0.8)
        ax.bar(x_pos + width / 2, player2_rates, width, label="Player 2", alpha=0.8)

        ax.set_xlabel("Matchup Type")
        ax.set_ylabel("Win Rate")
        ax.set_title(
            f"Simplified UNO Win Rates\n(max_number={max_number}, max_colors={max_colors})"
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [m.replace("PlayerType.", "").replace("_", " ") for m in matchup_types],
            rotation=15,
            ha="right",
        )
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/simplified_win_rates.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_comparison(self, output_dir: str):
        """Create comparison plots between simplified and standard UNO."""
        if not self.black_logs or not self.baseline_logs:
            return

        black_log = self.black_logs[-1]
        baseline_log = self.baseline_logs[-1]

        black_wins = black_log.get("win_rates", {})
        baseline_wins = baseline_log.get("win_rates", {})
        black_times = black_log.get("avg_decision_times", {})
        baseline_times = baseline_log.get("avg_decision_times", {})

        # Get common matchup types
        matchup_types = sorted(set(black_wins.keys()) & set(baseline_wins.keys()))
        if not matchup_types:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(
            "Simplified vs Standard UNO Comparison", fontsize=16, fontweight="bold"
        )

        # Win Rate Comparison
        x_pos = np.arange(len(matchup_types))
        width = 0.2

        black_p1_rates = []
        black_p2_rates = []
        baseline_p1_rates = []
        baseline_p2_rates = []

        for matchup in matchup_types:
            black_rate = black_wins[matchup]
            baseline_rate = baseline_wins[matchup]

            if isinstance(black_rate, dict):
                black_p1_rates.append(black_rate.get("player1", 0))
                black_p2_rates.append(black_rate.get("player2", 0))
            else:
                black_p1_rates.append(0)
                black_p2_rates.append(0)

            if isinstance(baseline_rate, dict):
                baseline_p1_rates.append(baseline_rate.get("player1", 0))
                baseline_p2_rates.append(baseline_rate.get("player2", 0))
            else:
                baseline_p1_rates.append(0)
                baseline_p2_rates.append(0)

        ax1.bar(
            x_pos - 1.5 * width,
            black_p1_rates,
            width,
            label="Simplified P1",
            alpha=0.8,
            color="steelblue",
        )
        ax1.bar(
            x_pos - 0.5 * width,
            black_p2_rates,
            width,
            label="Simplified P2",
            alpha=0.8,
            color="lightblue",
        )
        ax1.bar(
            x_pos + 0.5 * width,
            baseline_p1_rates,
            width,
            label="Standard P1",
            alpha=0.8,
            color="darkgreen",
        )
        ax1.bar(
            x_pos + 1.5 * width,
            baseline_p2_rates,
            width,
            label="Standard P2",
            alpha=0.8,
            color="lightgreen",
        )

        ax1.set_xlabel("Matchup Type")
        ax1.set_ylabel("Win Rate")
        ax1.set_title("Win Rate Comparison")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(
            [m.replace("PlayerType.", "").replace("_", " ") for m in matchup_types],
            rotation=15,
            ha="right",
            fontsize=9,
        )
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3, axis="y")

        # Decision Time Comparison
        black_avg_times = []
        baseline_avg_times = []

        for matchup in matchup_types:
            black_time = black_times.get(matchup, {})
            baseline_time = baseline_times.get(matchup, {})

            if isinstance(black_time, dict):
                times = [
                    t
                    for t in [
                        black_time.get("player1", 0),
                        black_time.get("player2", 0),
                    ]
                    if t > 0
                ]
                black_avg_times.append(sum(times) / len(times) if times else 0)
            else:
                black_avg_times.append(0)

            if isinstance(baseline_time, dict):
                times = [
                    t
                    for t in [
                        baseline_time.get("player1", 0),
                        baseline_time.get("player2", 0),
                    ]
                    if t > 0
                ]
                baseline_avg_times.append(sum(times) / len(times) if times else 0)
            else:
                baseline_avg_times.append(0)

        x_pos2 = np.arange(len(matchup_types))
        width2 = 0.35

        ax2.bar(
            x_pos2 - width2 / 2,
            black_avg_times,
            width2,
            label="Simplified",
            alpha=0.8,
            color="steelblue",
        )
        ax2.bar(
            x_pos2 + width2 / 2,
            baseline_avg_times,
            width2,
            label="Standard",
            alpha=0.8,
            color="darkgreen",
        )

        ax2.set_xlabel("Matchup Type")
        ax2.set_ylabel("Avg Decision Time (seconds)")
        ax2.set_title("Decision Time Comparison")
        ax2.set_xticks(x_pos2)
        ax2.set_xticklabels(
            [m.replace("PlayerType.", "").replace("_", " ") for m in matchup_types],
            rotation=15,
            ha="right",
            fontsize=9,
        )
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/simplified_vs_standard_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def main():
    """Main function to run simplified game results analysis."""
    import sys

    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    create_plots = "--plots" in sys.argv or "--visualize" in sys.argv

    analyzer = ResultsBlackAnalyzer(results_dir)
    analyzer.load_logs()
    analyzer.print_summary()

    if create_plots:
        analyzer.create_visualizations()


if __name__ == "__main__":
    main()
