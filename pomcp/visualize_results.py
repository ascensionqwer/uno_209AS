#!/usr/bin/env python3
"""UNO Simulation Results Visualizer

Creates comprehensive matplotlib graphs for batch baseline and batch black results.
Saves all visualizations to the results/ directory.

Color scheme:
- Red shades for naive agents
- Blue shades for POMCP agents
- Different shades for same-type matchups
"""

import json
import os
import glob
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Color scheme
COLORS = {
    'naive_light': '#FF6B6B',    # Light red
    'naive_dark': '#C92A2A',      # Dark red
    'pomcp_light': '#74C0FC',    # Light blue
    'pomcp_dark': '#1864AB',      # Dark blue
    'baseline_gray': '#868E96',   # For comparisons
    'grid': '#E9ECEF'            # Light gray for grids
}

class ResultsVisualizer:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.logs = []
        self.baseline_logs = []
        self.simplified_logs = []
        self.variant_logs = []
        
    def load_logs(self) -> bool:
        """Load all JSON logs from results directory."""
        if not os.path.exists(self.results_dir):
            print(f"Results directory '{self.results_dir}' not found.")
            return False

        json_files = glob.glob(os.path.join(self.results_dir, "*.json"))
        if not json_files:
            print(f"No JSON files found in '{self.results_dir}'.")
            return False

        for file_path in sorted(json_files):
            try:
                with open(file_path, "r") as f:
                    log_data = json.load(f)
                    log_data["file_path"] = file_path
                    self.logs.append(log_data)
                    
                    # Categorize logs
                    matchup_type = log_data.get("matchup", "")
                    if matchup_type == "baseline":
                        self.baseline_logs.append(log_data)
                    elif "simplified_baseline" in matchup_type:
                        self.simplified_logs.append(log_data)
                    elif matchup_type == "all_matchups_with_variants":
                        self.variant_logs.append(log_data)
                        
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                return False

        print(f"Loaded {len(self.logs)} result files:")
        print(f"  - {len(self.baseline_logs)} baseline logs")
        print(f"  - {len(self.simplified_logs)} simplified logs")
        print(f"  - {len(self.variant_logs)} variant logs")
        return True

    def create_all_visualizations(self):
        """Create all visualization graphs."""
        if not self.logs:
            print("No logs loaded. Run load_logs() first.")
            return

        print("Creating visualizations...")
        
        # Create output directory
        output_dir = self.results_dir
        os.makedirs(output_dir, exist_ok=True)

        # Create individual graphs
        self._create_win_rate_comparison(output_dir)
        self._create_decision_time_comparison(output_dir)
        self._create_cache_size_analysis(output_dir)
        self._create_turn_statistics_comparison(output_dir)
        self._create_simplified_vs_baseline_comparison(output_dir)
        self._create_summary_dashboard(output_dir)
        
        if self.variant_logs:
            self._create_parameter_impact_visualization(output_dir)

        print(f"All visualizations saved to: {output_dir}/")

    def _get_matchup_data(self, log: Dict[str, Any]) -> Dict[str, Any]:
        """Extract matchup data from log."""
        win_rates = log.get("win_rates", {})
        decision_times = log.get("avg_decision_times", {})
        cache_stats = log.get("cache_stats", {})
        
        return {
            'win_rates': win_rates,
            'decision_times': decision_times,
            'cache_stats': cache_stats,
            'total_games': log.get("total_games", 0),
            'timestamp': log.get("timestamp", "Unknown")
        }

    def _create_win_rate_comparison(self, output_dir: str):
        """Create win rate comparison chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Win Rate Comparison Across All Matchups', fontsize=16, fontweight='bold')

        # Baseline results
        if self.baseline_logs:
            data = self._get_matchup_data(self.baseline_logs[-1])
            self._plot_win_rates_on_axis(ax1, data, "Standard UNO")

        # Simplified results
        if self.simplified_logs:
            data = self._get_matchup_data(self.simplified_logs[-1])
            self._plot_win_rates_on_axis(ax2, data, "Simplified UNO")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/win_rates_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_win_rates_on_axis(self, ax, data: Dict[str, Any], title: str):
        """Plot win rates on a specific axis."""
        win_rates = data['win_rates']
        matchups = list(win_rates.keys())
        
        if not matchups:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return

        x_pos = np.arange(len(matchups))
        width = 0.35

        player1_rates = []
        player2_rates = []

        for matchup in matchups:
            rates = win_rates[matchup]
            if isinstance(rates, dict):
                player1_rates.append(rates.get("player1", 0))
                player2_rates.append(rates.get("player2", 0))
            else:
                player1_rates.append(0)
                player2_rates.append(0)

        # Determine colors based on matchup type
        p1_colors = []
        p2_colors = []
        
        for matchup in matchups:
            if "naive_vs_naive" in matchup:
                p1_colors.append(COLORS['naive_light'])
                p2_colors.append(COLORS['naive_dark'])
            elif "particle_policy_vs_particle_policy" in matchup:
                p1_colors.append(COLORS['pomcp_light'])
                p2_colors.append(COLORS['pomcp_dark'])
            elif "naive_vs_particle_policy" in matchup:
                p1_colors.append(COLORS['naive_dark'])
                p2_colors.append(COLORS['pomcp_dark'])
            else:
                p1_colors.append(COLORS['baseline_gray'])
                p2_colors.append(COLORS['baseline_gray'])

        # Create bars with individual colors
        for i, (x, p1_rate, p2_rate, p1_color, p2_color) in enumerate(zip(x_pos, player1_rates, player2_rates, p1_colors, p2_colors)):
            ax.bar(x - width/2, p1_rate, width, color=p1_color, alpha=0.8)
            ax.bar(x + width/2, p2_rate, width, color=p2_color, alpha=0.8)

        ax.set_xlabel('Matchup Type')
        ax.set_ylabel('Win Rate')
        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.replace('_', ' ').replace('naive', 'Naive').replace('particle policy', 'POMCP') for m in matchups], 
                          rotation=15, ha='right')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y', color=COLORS['grid'])
        
        # Add custom legend based on player types
        legend_elements = []
        if any('naive' in m for m in matchups):
            legend_elements.append(mpatches.Rectangle((0,0),1,1, color=COLORS['naive_dark'], alpha=0.8, label='Naive Agent'))
        if any('particle_policy' in m for m in matchups):
            legend_elements.append(mpatches.Rectangle((0,0),1,1, color=COLORS['pomcp_dark'], alpha=0.8, label='POMCP Agent'))
        if legend_elements:
            ax.legend(handles=legend_elements)

    def _create_decision_time_comparison(self, output_dir: str):
        """Create decision time comparison chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Decision Time Comparison (Log Scale)', fontsize=16, fontweight='bold')

        # Baseline results
        if self.baseline_logs:
            data = self._get_matchup_data(self.baseline_logs[-1])
            self._plot_decision_times_on_axis(ax1, data, "Standard UNO")

        # Simplified results
        if self.simplified_logs:
            data = self._get_matchup_data(self.simplified_logs[-1])
            self._plot_decision_times_on_axis(ax2, data, "Simplified UNO")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/decision_times_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_decision_times_on_axis(self, ax, data: Dict[str, Any], title: str):
        """Plot decision times on a specific axis."""
        decision_times = data['decision_times']
        matchups = list(decision_times.keys())
        
        if not matchups:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return

        x_pos = np.arange(len(matchups))
        width = 0.35

        player1_times = []
        player2_times = []

        for matchup in matchups:
            times = decision_times[matchup]
            if isinstance(times, dict):
                p1_time = times.get("player1", 0)
                p2_time = times.get("player2", 0)
                # Keep all times in seconds for consistent display
                player1_times.append(p1_time)
                player2_times.append(p2_time)
            else:
                player1_times.append(0)
                player2_times.append(0)

        # Determine colors
        p1_colors = []
        p2_colors = []
        
        for matchup in matchups:
            if "naive_vs_naive" in matchup:
                p1_colors.append(COLORS['naive_light'])
                p2_colors.append(COLORS['naive_dark'])
            elif "particle_policy_vs_particle_policy" in matchup:
                p1_colors.append(COLORS['pomcp_light'])
                p2_colors.append(COLORS['pomcp_dark'])
            elif "naive_vs_particle_policy" in matchup:
                p1_colors.append(COLORS['naive_dark'])
                p2_colors.append(COLORS['pomcp_dark'])
            else:
                p1_colors.append(COLORS['baseline_gray'])
                p2_colors.append(COLORS['baseline_gray'])

        # Create bars
        for i, (x, p1_time, p2_time, p1_color, p2_color) in enumerate(zip(x_pos, player1_times, player2_times, p1_colors, p2_colors)):
            ax.bar(x - width/2, p1_time, width, color=p1_color, alpha=0.8)
            ax.bar(x + width/2, p2_time, width, color=p2_color, alpha=0.8)

        ax.set_xlabel('Matchup Type')
        ax.set_ylabel('Decision Time (seconds)')
        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.replace('_', ' ').replace('naive', 'Naive').replace('particle policy', 'POMCP') for m in matchups], 
                          rotation=15, ha='right')
        ax.grid(True, alpha=0.3, axis='y', color=COLORS['grid'])
        
        # Add custom legend based on player types
        legend_elements = []
        if any('naive' in m for m in matchups):
            legend_elements.append(mpatches.Rectangle((0,0),1,1, color=COLORS['naive_dark'], alpha=0.8, label='Naive Agent'))
        if any('particle_policy' in m for m in matchups):
            legend_elements.append(mpatches.Rectangle((0,0),1,1, color=COLORS['pomcp_dark'], alpha=0.8, label='POMCP Agent'))
        if legend_elements:
            ax.legend(handles=legend_elements)

    def _create_cache_size_analysis(self, output_dir: str):
        """Create cache size analysis chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Cache Size Analysis (POMCP Agents Only)', fontsize=16, fontweight='bold')

        # Baseline results
        if self.baseline_logs:
            data = self._get_matchup_data(self.baseline_logs[-1])
            self._plot_cache_sizes_on_axis(ax1, data, "Standard UNO")

        # Simplified results
        if self.simplified_logs:
            data = self._get_matchup_data(self.simplified_logs[-1])
            self._plot_cache_sizes_on_axis(ax2, data, "Simplified UNO")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/cache_size_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_cache_sizes_on_axis(self, ax, data: Dict[str, Any], title: str):
        """Plot cache sizes on a specific axis."""
        cache_stats = data['cache_stats']
        matchups = list(cache_stats.keys())
        
        if not matchups:
            ax.text(0.5, 0.5, 'No cache data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return

        x_pos = np.arange(len(matchups))
        width = 0.35

        # Extract and fix cache data (naive agents should have 0 cache)
        player1_caches = []
        player2_caches = []
        
        for matchup in matchups:
            stats = cache_stats[matchup]
            if isinstance(stats, dict):
                p1_cache = stats.get("player1", 0) or 0
                p2_cache = stats.get("player2", 0) or 0
                
                # Fix data logging bug: naive agents should have no cache
                if "naive_vs_naive" in matchup:
                    player1_caches.append(0)
                    player2_caches.append(0)
                elif "naive_vs_particle_policy" in matchup:
                    player1_caches.append(0)  # Player 1 is naive
                    player2_caches.append(p2_cache)  # Player 2 is POMCP
                elif "particle_policy_vs_particle_policy" in matchup:
                    player1_caches.append(p1_cache)  # Both are POMCP
                    player2_caches.append(p2_cache)
                else:
                    player1_caches.append(0)
                    player2_caches.append(0)
            else:
                player1_caches.append(0)
                player2_caches.append(0)

        # Only show matchups with POMCP agents that have cache data
        valid_indices = [i for i, (p1, p2) in enumerate(zip(player1_caches, player2_caches)) if p1 > 0 or p2 > 0]
        
        if not valid_indices:
            ax.text(0.5, 0.5, 'No POMCP cache data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return

        # Filter to valid data
        valid_x_pos = [x_pos[i] for i in valid_indices]
        valid_matchups = [matchups[i] for i in valid_indices]
        valid_p1_caches = [player1_caches[i] for i in valid_indices]
        valid_p2_caches = [player2_caches[i] for i in valid_indices]

        # Create bars with proper colors
        for i, (x, p1_cache, p2_cache) in enumerate(zip(valid_x_pos, valid_p1_caches, valid_p2_caches)):
            matchup = valid_matchups[i]
            
            if "particle_policy_vs_particle_policy" in matchup:
                # Both players are POMCP
                if p1_cache > 0:
                    ax.bar(x - width/2, p1_cache, width, color=COLORS['pomcp_light'], alpha=0.8, label='POMCP Player 1' if i == 0 else '')
                if p2_cache > 0:
                    ax.bar(x + width/2, p2_cache, width, color=COLORS['pomcp_dark'], alpha=0.8, label='POMCP Player 2' if i == 0 else '')
            elif "naive_vs_particle_policy" in matchup:
                # Player 1 is naive (no cache), Player 2 is POMCP
                if p2_cache > 0:
                    ax.bar(x + width/2, p2_cache, width, color=COLORS['pomcp_dark'], alpha=0.8, label='POMCP Agent' if i == 0 else '')

        ax.set_xlabel('Matchup Type')
        ax.set_ylabel('Cache Size')
        ax.set_title(title)
        ax.set_xticks(valid_x_pos)
        ax.set_xticklabels([m.replace('_', ' ').replace('naive', 'Naive').replace('particle policy', 'POMCP') for m in valid_matchups], 
                          rotation=15, ha='right')
        ax.grid(True, alpha=0.3, axis='y', color=COLORS['grid'])
        
        # Add legend for POMCP agents only
        legend_elements = [
            mpatches.Rectangle((0,0),1,1, color=COLORS['pomcp_light'], alpha=0.8, label='POMCP Player 1'),
            mpatches.Rectangle((0,0),1,1, color=COLORS['pomcp_dark'], alpha=0.8, label='POMCP Player 2')
        ]
        ax.legend(handles=legend_elements)

    def _create_turn_statistics_comparison(self, output_dir: str):
        """Create turn statistics comparison chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Game Length (Turn Count) Comparison', fontsize=16, fontweight='bold')

        # Baseline results
        if self.baseline_logs:
            data = self._get_matchup_data(self.baseline_logs[-1])
            self._plot_turn_stats_on_axis(ax1, data, "Standard UNO")

        # Simplified results
        if self.simplified_logs:
            data = self._get_matchup_data(self.simplified_logs[-1])
            self._plot_turn_stats_on_axis(ax2, data, "Simplified UNO")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/turn_statistics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_turn_stats_on_axis(self, ax, data: Dict[str, Any], title: str):
        """Plot turn statistics on a specific axis."""
        # Handle both flat turn_stats and nested structure from player_wins
        player_wins = data.get('player_wins', {})
        
        if not player_wins:
            ax.text(0.5, 0.5, 'No turn statistics available\n(Run new simulations to generate)', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return

        # Extract turn statistics from nested structure in player_wins
        matchups = []
        avg_turns = []
        min_turns = []
        max_turns = []

        for matchup_key, matchup_data in player_wins.items():
            if isinstance(matchup_data, dict) and 'turn_stats' in matchup_data:
                turn_stats = matchup_data['turn_stats']
                if isinstance(turn_stats, dict):
                    matchups.append(matchup_key)
                    avg_turns.append(turn_stats.get('average', 0))
                    min_turns.append(turn_stats.get('min', 0))
                    max_turns.append(turn_stats.get('max', 0))

        if not matchups:
            ax.text(0.5, 0.5, 'No valid turn statistics found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return

        x_pos = np.arange(len(matchups))
        width = 0.25

        # Create grouped bar chart
        ax.bar(x_pos - width, avg_turns, width, label='Average', color=COLORS['pomcp_dark'], alpha=0.8)
        ax.bar(x_pos, min_turns, width, label='Minimum', color=COLORS['pomcp_light'], alpha=0.8)
        ax.bar(x_pos + width, max_turns, width, label='Maximum', color=COLORS['baseline_gray'], alpha=0.8)

        ax.set_xlabel('Parameter Variant')
        ax.set_ylabel('Number of Turns')
        ax.set_title(title)
        ax.set_xticks(x_pos)
        
        # Clean up matchup names for display
        clean_names = []
        for matchup in matchups:
            if 'num_particles_increase' in matchup:
                clean_names.append('Particles +25%')
            elif 'num_particles_decrease' in matchup:
                clean_names.append('Particles -25%')
            elif 'mcts_iterations_increase' in matchup:
                clean_names.append('MCTS Iter +25%')
            elif 'mcts_iterations_decrease' in matchup:
                clean_names.append('MCTS Iter -25%')
            else:
                clean_names.append(matchup.replace('_', ' ').title())
        
        ax.set_xticklabels(clean_names, rotation=15, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y', color=COLORS['grid'])

    def _create_simplified_vs_baseline_comparison(self, output_dir: str):
        """Create side-by-side comparison of simplified vs baseline UNO."""
        if not self.baseline_logs or not self.simplified_logs:
            print("Skipping simplified vs baseline comparison - missing data")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Simplified vs Standard UNO Comparison', fontsize=16, fontweight='bold')

        baseline_data = self._get_matchup_data(self.baseline_logs[-1])
        simplified_data = self._get_matchup_data(self.simplified_logs[-1])

        # Win rates comparison
        self._compare_win_rates(ax1, baseline_data, simplified_data)
        
        # Decision times comparison
        self._compare_decision_times(ax2, baseline_data, simplified_data)
        
        # Cache sizes comparison
        self._compare_cache_sizes(ax3, baseline_data, simplified_data)
        
        # Summary statistics
        self._create_summary_stats(ax4, baseline_data, simplified_data)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/simplified_vs_baseline_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _compare_win_rates(self, ax, baseline_data: Dict[str, Any], simplified_data: Dict[str, Any]):
        """Compare win rates between baseline and simplified."""
        baseline_rates = baseline_data['win_rates']
        simplified_rates = simplified_data['win_rates']
        
        matchups = list(set(baseline_rates.keys()) & set(simplified_rates.keys()))
        if not matchups:
            ax.text(0.5, 0.5, 'No common matchups', ha='center', va='center', transform=ax.transAxes)
            return

        x_pos = np.arange(len(matchups))
        width = 0.2

        for i, matchup in enumerate(matchups):
            base_rates = baseline_rates[matchup]
            simp_rates = simplified_rates[matchup]
            
            if isinstance(base_rates, dict) and isinstance(simp_rates, dict):
                # Player 1
                ax.bar(x_pos[i] - 1.5*width, base_rates.get('player1', 0), width, 
                      color=COLORS['baseline_gray'], alpha=0.8)
                ax.bar(x_pos[i] - 0.5*width, simp_rates.get('player1', 0), width, 
                      color=COLORS['naive_dark'], alpha=0.8)
                
                # Player 2
                ax.bar(x_pos[i] + 0.5*width, base_rates.get('player2', 0), width, 
                      color=COLORS['baseline_gray'], alpha=0.5)
                ax.bar(x_pos[i] + 1.5*width, simp_rates.get('player2', 0), width, 
                      color=COLORS['pomcp_dark'], alpha=0.8)

        ax.set_xlabel('Matchup Type')
        ax.set_ylabel('Win Rate')
        ax.set_title('Win Rate Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.replace('_', ' ').replace('naive', 'Naive').replace('particle policy', 'POMCP') for m in matchups], 
                          rotation=15, ha='right')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y', color=COLORS['grid'])
        
        # Add custom legend
        legend_elements = [
            mpatches.Rectangle((0,0),1,1, color=COLORS['baseline_gray'], alpha=0.8, label='Standard UNO'),
            mpatches.Rectangle((0,0),1,1, color=COLORS['naive_dark'], alpha=0.8, label='Simplified Naive'),
            mpatches.Rectangle((0,0),1,1, color=COLORS['pomcp_dark'], alpha=0.8, label='Simplified POMCP')
        ]
        ax.legend(handles=legend_elements, fontsize=8)

    def _compare_decision_times(self, ax, baseline_data: Dict[str, Any], simplified_data: Dict[str, Any]):
        """Compare decision times between baseline and simplified."""
        baseline_times = baseline_data['decision_times']
        simplified_times = simplified_data['decision_times']
        
        matchups = list(set(baseline_times.keys()) & set(simplified_times.keys()))
        if not matchups:
            ax.text(0.5, 0.5, 'No common matchups', ha='center', va='center', transform=ax.transAxes)
            return

        x_pos = np.arange(len(matchups))
        width = 0.2

        for i, matchup in enumerate(matchups):
            base_times = baseline_times[matchup]
            simp_times = simplified_times[matchup]
            
            if isinstance(base_times, dict) and isinstance(simp_times, dict):
                # Player 1
                p1_base = base_times.get('player1', 0)
                p1_simp = simp_times.get('player1', 0)
                
                ax.bar(x_pos[i] - 1.5*width, p1_base, width, 
                      color=COLORS['baseline_gray'], alpha=0.8)
                ax.bar(x_pos[i] - 0.5*width, p1_simp, width, 
                      color=COLORS['naive_dark'], alpha=0.8)
                
                # Player 2
                p2_base = base_times.get('player2', 0)
                p2_simp = simp_times.get('player2', 0)
                
                ax.bar(x_pos[i] + 0.5*width, p2_base, width, 
                      color=COLORS['baseline_gray'], alpha=0.5)
                ax.bar(x_pos[i] + 1.5*width, p2_simp, width, 
                      color=COLORS['pomcp_dark'], alpha=0.8)

        ax.set_xlabel('Matchup Type')
        ax.set_ylabel('Decision Time (seconds)')
        ax.set_title('Decision Time Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.replace('_', ' ').replace('naive', 'Naive').replace('particle policy', 'POMCP') for m in matchups], 
                          rotation=15, ha='right')
        ax.grid(True, alpha=0.3, axis='y', color=COLORS['grid'])
        
        # Add custom legend
        legend_elements = [
            mpatches.Rectangle((0,0),1,1, color=COLORS['baseline_gray'], alpha=0.8, label='Standard UNO'),
            mpatches.Rectangle((0,0),1,1, color=COLORS['naive_dark'], alpha=0.8, label='Simplified Naive'),
            mpatches.Rectangle((0,0),1,1, color=COLORS['pomcp_dark'], alpha=0.8, label='Simplified POMCP')
        ]
        ax.legend(handles=legend_elements, fontsize=8)

    def _compare_cache_sizes(self, ax, baseline_data: Dict[str, Any], simplified_data: Dict[str, Any]):
        """Compare cache sizes between baseline and simplified."""
        baseline_caches = baseline_data['cache_stats']
        simplified_caches = simplified_data['cache_stats']
        
        matchups = list(set(baseline_caches.keys()) & set(simplified_caches.keys()))
        if not matchups:
            ax.text(0.5, 0.5, 'No common matchups', ha='center', va='center', transform=ax.transAxes)
            return

        x_pos = np.arange(len(matchups))
        width = 0.2

        for i, matchup in enumerate(matchups):
            base_caches = baseline_caches[matchup]
            simp_caches = simplified_caches[matchup]
            
            if isinstance(base_caches, dict) and isinstance(simp_caches, dict):
                # Player 1
                p1_base = base_caches.get('player1', 0) or 0
                p1_simp = simp_caches.get('player1', 0) or 0
                
                if p1_base > 0:
                    ax.bar(x_pos[i] - 1.5*width, p1_base, width, 
                          color=COLORS['baseline_gray'], alpha=0.8)
                if p1_simp > 0:
                    ax.bar(x_pos[i] - 0.5*width, p1_simp, width, 
                          color=COLORS['pomcp_light'], alpha=0.8)
                
                # Player 2
                p2_base = base_caches.get('player2', 0) or 0
                p2_simp = simp_caches.get('player2', 0) or 0
                
                if p2_base > 0:
                    ax.bar(x_pos[i] + 0.5*width, p2_base, width, 
                          color=COLORS['baseline_gray'], alpha=0.5)
                if p2_simp > 0:
                    ax.bar(x_pos[i] + 1.5*width, p2_simp, width, 
                          color=COLORS['pomcp_dark'], alpha=0.8)

        ax.set_xlabel('Matchup Type')
        ax.set_ylabel('Cache Size')
        ax.set_title('Cache Size Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.replace('_', ' ').replace('naive', 'Naive').replace('particle policy', 'POMCP') for m in matchups], 
                          rotation=15, ha='right')
        ax.grid(True, alpha=0.3, axis='y', color=COLORS['grid'])
        
        # Add custom legend
        legend_elements = [
            mpatches.Rectangle((0,0),1,1, color=COLORS['baseline_gray'], alpha=0.8, label='Standard UNO'),
            mpatches.Rectangle((0,0),1,1, color=COLORS['pomcp_light'], alpha=0.8, label='Simplified POMCP P1'),
            mpatches.Rectangle((0,0),1,1, color=COLORS['pomcp_dark'], alpha=0.8, label='Simplified POMCP P2')
        ]
        ax.legend(handles=legend_elements, fontsize=8)

    def _create_summary_stats(self, ax, baseline_data: Dict[str, Any], simplified_data: Dict[str, Any]):
        """Create summary statistics comparison."""
        # Calculate summary statistics
        stats_text = []
        
        # Total games
        stats_text.append(f"Total Games:")
        stats_text.append(f"  Baseline: {baseline_data['total_games']}")
        stats_text.append(f"  Simplified: {simplified_data['total_games']}")
        stats_text.append("")
        
        # Average win rates for POMCP agents
        baseline_pomcp_wins = []
        simplified_pomcp_wins = []
        
        for matchup, rates in baseline_data['win_rates'].items():
            if isinstance(rates, dict):
                if "particle_policy" in matchup:
                    if "naive_vs_particle_policy" in matchup:
                        baseline_pomcp_wins.append(rates.get('player2', 0))
                    elif "particle_policy_vs_particle_policy" in matchup:
                        baseline_pomcp_wins.extend([rates.get('player1', 0), rates.get('player2', 0)])
        
        for matchup, rates in simplified_data['win_rates'].items():
            if isinstance(rates, dict):
                if "particle_policy" in matchup:
                    if "naive_vs_particle_policy" in matchup:
                        simplified_pomcp_wins.append(rates.get('player2', 0))
                    elif "particle_policy_vs_particle_policy" in matchup:
                        simplified_pomcp_wins.extend([rates.get('player1', 0), rates.get('player2', 0)])
        
        if baseline_pomcp_wins:
            stats_text.append(f"POMCP Avg Win Rate:")
            stats_text.append(f"  Baseline: {np.mean(baseline_pomcp_wins):.1%}")
        if simplified_pomcp_wins:
            stats_text.append(f"  Simplified: {np.mean(simplified_pomcp_wins):.1%}")
        
        stats_text.append("")
        
        # Average decision times for POMCP agents
        baseline_pomcp_times = []
        simplified_pomcp_times = []
        
        for matchup, times in baseline_data['decision_times'].items():
            if isinstance(times, dict):
                if "particle_policy" in matchup:
                    if "naive_vs_particle_policy" in matchup:
                        baseline_pomcp_times.append(times.get('player2', 0))
                    elif "particle_policy_vs_particle_policy" in matchup:
                        baseline_pomcp_times.extend([times.get('player1', 0), times.get('player2', 0)])
        
        for matchup, times in simplified_data['decision_times'].items():
            if isinstance(times, dict):
                if "particle_policy" in matchup:
                    if "naive_vs_particle_policy" in matchup:
                        simplified_pomcp_times.append(times.get('player2', 0))
                    elif "particle_policy_vs_particle_policy" in matchup:
                        simplified_pomcp_times.extend([times.get('player1', 0), times.get('player2', 0)])
        
        if baseline_pomcp_times:
            avg_baseline_time = np.mean([t for t in baseline_pomcp_times if t > 1e-6])  # Filter out naive times
            stats_text.append(f"POMCP Avg Decision Time:")
            stats_text.append(f"  Baseline: {avg_baseline_time:.3f}s")
        if simplified_pomcp_times:
            avg_simplified_time = np.mean([t for t in simplified_pomcp_times if t > 1e-6])
            stats_text.append(f"  Simplified: {avg_simplified_time:.3f}s")

        ax.text(0.05, 0.95, '\n'.join(stats_text), transform=ax.transAxes, 
               fontsize=9, verticalalignment='top', fontfamily='monospace')
        ax.set_title('Summary Statistics')
        ax.axis('off')

    def _create_summary_dashboard(self, output_dir: str):
        """Create a comprehensive summary dashboard."""
        fig = plt.figure(figsize=(22, 14))
        fig.suptitle('UNO Simulation Results Dashboard', fontsize=18, fontweight='bold')

        # Create grid layout with more spacing
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4)

        # Initialize data variables
        baseline_data = None
        simplified_data = None

        # Win rates (top left, spanning 2 cols)
        ax_win = fig.add_subplot(gs[0, :2])
        if self.baseline_logs and self.simplified_logs:
            baseline_data = self._get_matchup_data(self.baseline_logs[-1])
            simplified_data = self._get_matchup_data(self.simplified_logs[-1])
            self._create_dashboard_win_rates(ax_win, baseline_data, simplified_data)

        # Performance metrics (top right)
        ax_perf = fig.add_subplot(gs[0, 2])
        if baseline_data and simplified_data:
            self._create_dashboard_performance(ax_perf, baseline_data, simplified_data)

        # Decision times (middle left)
        ax_time = fig.add_subplot(gs[1, 0])
        if self.simplified_logs:
            data = self._get_matchup_data(self.simplified_logs[-1])
            self._plot_decision_times_on_axis(ax_time, data, "Decision Times")

        # Cache sizes (middle center)
        ax_cache = fig.add_subplot(gs[1, 1])
        if self.simplified_logs:
            data = self._get_matchup_data(self.simplified_logs[-1])
            self._plot_cache_sizes_on_axis(ax_cache, data, "Cache Sizes")

        # Summary text (middle right)
        ax_summary = fig.add_subplot(gs[1, 2])
        if baseline_data and simplified_data:
            self._create_summary_stats(ax_summary, baseline_data, simplified_data)

        # Bottom row - individual matchup details
        ax_detail1 = fig.add_subplot(gs[2, 0])
        ax_detail2 = fig.add_subplot(gs[2, 1])
        ax_detail3 = fig.add_subplot(gs[2, 2])
        
        if self.simplified_logs:
            data = self._get_matchup_data(self.simplified_logs[-1])
            self._create_matchup_detail(ax_detail1, data, "naive_vs_naive", "Naive vs Naive")
            self._create_matchup_detail(ax_detail2, data, "naive_vs_particle_policy", "Naive vs POMCP")
            self._create_matchup_detail(ax_detail3, data, "particle_policy_vs_particle_policy", "POMCP vs POMCP")

        plt.savefig(f"{output_dir}/summary_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_dashboard_win_rates(self, ax, baseline_data: Dict[str, Any], simplified_data: Dict[str, Any]):
        """Create win rate section for dashboard."""
        matchups = list(set(baseline_data['win_rates'].keys()) & set(simplified_data['win_rates'].keys()))
        
        # Filter to only matchups with POMCP agents
        pomcp_matchups = [m for m in matchups if "particle_policy" in m]
        
        if not pomcp_matchups:
            ax.text(0.5, 0.5, 'No POMCP matchups found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('POMCP Performance: Standard vs Simplified UNO')
            return
            
        x_pos = np.arange(len(pomcp_matchups))
        width = 0.35

        baseline_pomcp_rates = []
        simplified_pomcp_rates = []

        for matchup in pomcp_matchups:
            base_rates = baseline_data['win_rates'][matchup]
            simp_rates = simplified_data['win_rates'][matchup]
            
            if isinstance(base_rates, dict) and isinstance(simp_rates, dict):
                if "naive_vs_particle_policy" in matchup:
                    baseline_pomcp_rates.append(base_rates.get('player2', 0))
                    simplified_pomcp_rates.append(simp_rates.get('player2', 0))
                elif "particle_policy_vs_particle_policy" in matchup:
                    baseline_pomcp_rates.append((base_rates.get('player1', 0) + base_rates.get('player2', 0)) / 2)
                    simplified_pomcp_rates.append((simp_rates.get('player1', 0) + simp_rates.get('player2', 0)) / 2)

        ax.bar(x_pos - width/2, baseline_pomcp_rates, width, 
               color=COLORS['baseline_gray'], alpha=0.8, label='Standard UNO')
        ax.bar(x_pos + width/2, simplified_pomcp_rates, width, 
               color=COLORS['pomcp_dark'], alpha=0.8, label='Simplified UNO')

        ax.set_ylabel('POMCP Win Rate')
        ax.set_title('POMCP Performance: Standard vs Simplified UNO')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.replace('naive_vs_particle_policy', 'N vs P').replace('particle_policy_vs_particle_policy', 'P vs P') for m in pomcp_matchups])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y', color=COLORS['grid'])

    def _create_dashboard_performance(self, ax, baseline_data: Dict[str, Any], simplified_data: Dict[str, Any]):
        """Create performance metrics section for dashboard."""
        # Calculate speedup
        baseline_times = []
        simplified_times = []
        
        for matchup, times in baseline_data['decision_times'].items():
            if isinstance(times, dict) and "particle_policy" in matchup:
                if "naive_vs_particle_policy" in matchup:
                    baseline_times.append(times.get('player2', 0))
                elif "particle_policy_vs_particle_policy" in matchup:
                    baseline_times.extend([times.get('player1', 0), times.get('player2', 0)])
        
        for matchup, times in simplified_data['decision_times'].items():
            if isinstance(times, dict) and "particle_policy" in matchup:
                if "naive_vs_particle_policy" in matchup:
                    simplified_times.append(times.get('player2', 0))
                elif "particle_policy_vs_particle_policy" in matchup:
                    simplified_times.extend([times.get('player1', 0), times.get('player2', 0)])
        
        metrics = [
            f"Performance Metrics",
            f"",
            f"Speedup: N/A",
            f"",
            f"Avg Decision Time:",
            f"  Standard: N/A",
            f"  Simplified: N/A",
            f"",
            f"Games Analyzed:",
            f"  Standard: {baseline_data['total_games'] if baseline_data else 0}",
            f"  Simplified: {simplified_data['total_games'] if simplified_data else 0}"
        ]
        
        if baseline_times and simplified_times:
            avg_baseline = np.mean([t for t in baseline_times if t > 1e-6])
            avg_simplified = np.mean([t for t in simplified_times if t > 1e-6])
            speedup = avg_baseline / avg_simplified if avg_simplified > 0 else 0
            
            metrics = [
                f"Performance Metrics",
                f"",
                f"Speedup: {speedup:.1f}x",
                f"",
                f"Avg Decision Time:",
                f"  Standard: {avg_baseline:.3f}s",
                f"  Simplified: {avg_simplified:.3f}s",
                f"",
                f"Games Analyzed:",
                f"  Standard: {baseline_data['total_games']}",
                f"  Simplified: {simplified_data['total_games']}"
            ]
        
        ax.text(0.05, 0.95, '\n'.join(metrics), transform=ax.transAxes, 
               fontsize=9, verticalalignment='top', fontfamily='monospace')
        ax.set_title('Performance Summary')
        ax.axis('off')

    def _create_matchup_detail(self, ax, data: Dict[str, Any], matchup_key: str, title: str):
        """Create detailed view for a specific matchup."""
        if matchup_key not in data['win_rates']:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return

        win_rates = data['win_rates'][matchup_key]
        decision_times = data['decision_times'].get(matchup_key, {})
        cache_stats = data['cache_stats'].get(matchup_key, {})

        # Create a small summary
        details = [
            title,
            "",
            f"Win Rates:",
        ]
        
        if isinstance(win_rates, dict):
            for player, rate in win_rates.items():
                if player != 'no_winner':
                    details.append(f"  {player}: {rate:.1%}")
        
        details.append("")
        details.append("Decision Times:")
        
        if isinstance(decision_times, dict):
            for player, time_val in decision_times.items():
                if time_val > 1e-6:  # Only show POMCP times
                    details.append(f"  {player}: {time_val:.3f}s")
        
        if isinstance(cache_stats, dict):
            has_cache = any(v and v > 0 for v in cache_stats.values())
            if has_cache:
                details.append("")
                details.append("Cache Sizes:")
                for player, cache_size in cache_stats.items():
                    if cache_size and cache_size > 0:
                        details.append(f"  {player}: {cache_size:.1f}")

        ax.text(0.05, 0.95, '\n'.join(details), transform=ax.transAxes, 
               fontsize=8, verticalalignment='top', fontfamily='monospace')
        ax.set_title(title)
        ax.axis('off')

    def _create_parameter_impact_visualization(self, output_dir: str):
        """Create parameter impact visualization for variant results."""
        if not self.variant_logs:
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Parameter Impact Analysis', fontsize=16, fontweight='bold')

        # Extract parameter data
        param_data = self._extract_parameter_data()
        
        if not param_data:
            print("No parameter data found for visualization")
            plt.close()
            return

        # Create visualizations
        self._plot_parameter_win_rates(ax1, param_data)
        self._plot_parameter_decision_times(ax2, param_data)
        self._plot_parameter_cache_sizes(ax3, param_data)
        self._plot_parameter_overview(ax4, param_data)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/parameter_impact_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _extract_parameter_data(self) -> Dict[str, Any]:
        """Extract parameter variation data from variant logs."""
        if not self.variant_logs:
            return {}

        log = self.variant_logs[0]  # Use first variant log
        player_wins = log.get("player_wins", {})
        parameter_variants = log.get("parameter_variants", {})
        
        param_data = {}
        
        for variant_key, variant_data in player_wins.items():
            if variant_key.startswith("variant_"):
                # Extract parameter info
                param_info = parameter_variants.get(variant_key, {})
                param_path = param_info.get("parameter_path", "Unknown")
                variation_type = param_info.get("variation_type", "Unknown")
                
                if param_path not in param_data:
                    param_data[param_path] = {"increase": [], "decrease": []}
                
                # Extract metrics
                metrics = self._extract_variant_metrics(variant_data)
                param_data[param_path][variation_type].append(metrics)
        
        return param_data

    def _extract_variant_metrics(self, variant_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract key metrics from variant data."""
        metrics = {
            'pomp_win_rate': 0.0,
            'decision_time': 0.0,
            'cache_size': 0.0
        }
        
        # Extract from naive_vs_particle_policy
        nvp_data = variant_data.get("naive_vs_particle_policy", {})
        if nvp_data:
            win_rates = nvp_data.get("win_rates", {})
            metrics['pomp_win_rate'] = win_rates.get("player2", 0)
            
            decision_times = nvp_data.get("avg_decision_times", {})
            metrics['decision_time'] = decision_times.get("player2", 0)
            
            cache_stats = nvp_data.get("cache_stats", {})
            metrics['cache_size'] = cache_stats.get("player2", 0) or 0
        
        return metrics

    def _plot_parameter_win_rates(self, ax, param_data: Dict[str, Any]):
        """Plot parameter impact on win rates."""
        params = list(param_data.keys())
        x_pos = np.arange(len(params))
        width = 0.35

        increase_rates = []
        decrease_rates = []

        for param in params:
            inc_data = param_data[param]["increase"]
            dec_data = param_data[param]["decrease"]
            
            increase_rates.append(np.mean([m['pomp_win_rate'] for m in inc_data]) if inc_data else 0)
            decrease_rates.append(np.mean([m['pomp_win_rate'] for m in dec_data]) if dec_data else 0)

        ax.bar(x_pos - width/2, increase_rates, width, color='green', alpha=0.8, label='Increase 25%')
        ax.bar(x_pos + width/2, decrease_rates, width, color='red', alpha=0.8, label='Decrease 25%')

        ax.set_xlabel('Parameters')
        ax.set_ylabel('POMCP Win Rate')
        ax.set_title('Parameter Impact on Win Rates')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([p.split('.')[-1] for p in params], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y', color=COLORS['grid'])

    def _plot_parameter_decision_times(self, ax, param_data: Dict[str, Any]):
        """Plot parameter impact on decision times."""
        params = list(param_data.keys())
        x_pos = np.arange(len(params))
        width = 0.35

        increase_times = []
        decrease_times = []

        for param in params:
            inc_data = param_data[param]["increase"]
            dec_data = param_data[param]["decrease"]
            
            increase_times.append(np.mean([m['decision_time'] for m in inc_data if m['decision_time'] > 0]) if inc_data else 0)
            decrease_times.append(np.mean([m['decision_time'] for m in dec_data if m['decision_time'] > 0]) if dec_data else 0)

        ax.bar(x_pos - width/2, increase_times, width, color='green', alpha=0.8, label='Increase 25%')
        ax.bar(x_pos + width/2, decrease_times, width, color='red', alpha=0.8, label='Decrease 25%')

        ax.set_xlabel('Parameters')
        ax.set_ylabel('Decision Time (seconds)')
        ax.set_title('Parameter Impact on Decision Times')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([p.split('.')[-1] for p in params], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y', color=COLORS['grid'])

    def _plot_parameter_cache_sizes(self, ax, param_data: Dict[str, Any]):
        """Plot parameter impact on cache sizes."""
        params = list(param_data.keys())
        x_pos = np.arange(len(params))
        width = 0.35

        increase_cache = []
        decrease_cache = []

        for param in params:
            inc_data = param_data[param]["increase"]
            dec_data = param_data[param]["decrease"]
            
            increase_cache.append(np.mean([m['cache_size'] for m in inc_data if m['cache_size'] > 0]) if inc_data else 0)
            decrease_cache.append(np.mean([m['cache_size'] for m in dec_data if m['cache_size'] > 0]) if dec_data else 0)

        ax.bar(x_pos - width/2, increase_cache, width, color='green', alpha=0.8, label='Increase 25%')
        ax.bar(x_pos + width/2, decrease_cache, width, color='red', alpha=0.8, label='Decrease 25%')

        ax.set_xlabel('Parameters')
        ax.set_ylabel('Cache Size')
        ax.set_title('Parameter Impact on Cache Sizes')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([p.split('.')[-1] for p in params], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y', color=COLORS['grid'])

    def _plot_parameter_overview(self, ax, param_data: Dict[str, Any]):
        """Create parameter overview text."""
        overview_text = ["Parameter Variations Tested", ""]
        
        for param, data in param_data.items():
            overview_text.append(f"{param}:")
            overview_text.append(f"  Increase variants: {len(data['increase'])}")
            overview_text.append(f"  Decrease variants: {len(data['decrease'])}")
            overview_text.append("")
        
        ax.text(0.05, 0.95, '\n'.join(overview_text), transform=ax.transAxes, 
               fontsize=9, verticalalignment='top', fontfamily='monospace')
        ax.set_title('Parameter Overview')
        ax.axis('off')


def main():
    """Main function to run visualization."""
    visualizer = ResultsVisualizer()
    
    if not visualizer.load_logs():
        print("Failed to load result files.")
        return
    
    visualizer.create_all_visualizations()
    print("Visualization complete!")


if __name__ == "__main__":
    main()