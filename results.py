#!/usr/bin/env python3
"""Results analysis script for UNO simulation logs.

Parses JSON logs from results/ directory and prints formatted summaries.
"""

import json
import os
import glob
from typing import Dict, List, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


class ResultsAnalyzer:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.logs = []
        
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
                with open(file_path, 'r') as f:
                    log_data = json.load(f)
                    log_data['file_path'] = file_path
                    logs.append(log_data)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        self.logs = logs
        return logs
    
    def print_summary(self):
        """Print formatted summary of all results."""
        if not self.logs:
            print("No logs to analyze. Run simulations first.")
            return
            
        print("=" * 80)
        print("UNO SIMULATION RESULTS ANALYSIS")
        print("=" * 80)
        print(f"Found {len(self.logs)} log files")
        print()
        
        # Group logs by type
        matchup_logs = []
        sensitivity_logs = []
        
        for log in self.logs:
            if log.get('matchup') == 'parameter_sensitivity_analysis':
                sensitivity_logs.append(log)
            else:
                matchup_logs.append(log)
        
        # Print matchup results
        if matchup_logs:
            self._print_matchup_results(matchup_logs)
        
        # Print sensitivity analysis results
        if sensitivity_logs:
            self._print_sensitivity_results(sensitivity_logs)
    
    def _print_matchup_results(self, logs: List[Dict[str, Any]]):
        """Print results for standard matchup simulations."""
        print("STANDARD MATCHUP RESULTS")
        print("=" * 50)
        
        for log in logs:
            timestamp = log.get('timestamp', 'Unknown')
            matchup = log.get('matchup', 'Unknown')
            total_games = log.get('total_games', 0)
            
            print(f"\nTimestamp: {timestamp}")
            print(f"Matchup: {matchup}")
            print(f"Total Games: {total_games}")
            print("-" * 40)
            
            # Print win rates
            win_rates = log.get('win_rates', {})
            if win_rates:
                print("Win Rates:")
                for matchup_key, rates in win_rates.items():
                    if isinstance(rates, dict):
                        for player, rate in rates.items():
                            print(f"  {matchup_key} - {player}: {rate:.2%}")
                    else:
                        print(f"  {matchup_key}: {rates:.2%}")
            
            # Print decision times
            decision_times = log.get('avg_decision_times', {})
            if decision_times:
                print("\nDecision Times (seconds):")
                for matchup_key, times in decision_times.items():
                    if isinstance(times, dict):
                        for player, time_val in times.items():
                            print(f"  {matchup_key} - {player}: {time_val:.6f}")
                    else:
                        print(f"  {matchup_key}: {times:.6f}")
            
            # Print cache stats
            cache_stats = log.get('cache_stats', {})
            if cache_stats:
                print("\nCache Statistics:")
                for matchup_key, stats in cache_stats.items():
                    if isinstance(stats, dict):
                        for player, cache_size in stats.items():
                            if cache_size and cache_size != 'N/A (no cache)':
                                print(f"  {matchup_key} - {player}: {cache_size:.1f}")
                    elif stats and stats != 'N/A (no cache)':
                        print(f"  {matchup_key}: {stats}")
    
    def _print_sensitivity_results(self, logs: List[Dict[str, Any]]):
        """Print results for parameter sensitivity analysis."""
        print("\nPARAMETER SENSITIVITY ANALYSIS RESULTS")
        print("=" * 50)
        
        for log in logs:
            timestamp = log.get('timestamp', 'Unknown')
            total_games = log.get('total_games', 0)
            parameter_variants = log.get('parameter_variants', {})
            
            print(f"\nTimestamp: {timestamp}")
            print(f"Total Games: {total_games}")
            print(f"Variants Tested: {len(parameter_variants)}")
            print("-" * 40)
            
            # Group variants by parameter
            param_groups = {}
            for variant_key, variant_info in parameter_variants.items():
                param_path = variant_info.get('parameter_path', 'Unknown')
                if param_path not in param_groups:
                    param_groups[param_path] = []
                param_groups[param_path].append({
                    'key': variant_key,
                    'info': variant_info,
                    'results': self._extract_variant_results(log, variant_key)
                })
            
            # Print results for each parameter
            for param_path, variants in param_groups.items():
                print(f"\nParameter: {param_path}")
                print("-" * 30)
                
                for variant in variants:
                    info = variant['info']
                    results = variant['results']
                    
                    variation_type = info.get('variation_type', 'Unknown')
                    original_val = info.get('original_value', 'N/A')
                    variant_val = info.get('variant_value', 'N/A')
                    change_pct = info.get('percentage_change', 0) * 100
                    
                    print(f"  {variation_type.title()} {change_pct:.0f}%: {original_val} -> {variant_val}")
                    
                    if results:
                        print(f"    Particle vs Naive Win Rate: {results.get('pvn_win_rate', 'N/A'):.2%}")
                        print(f"    Particle vs Particle (Same) Win Rate: {results.get('pvp_same_win_rate', 'N/A'):.2%}")
                        print(f"    Particle vs Particle (Mixed) Win Rate: {results.get('pvp_mixed_win_rate', 'N/A'):.2%}")
                        print(f"    Avg Decision Time: {results.get('avg_decision_time', 'N/A'):.6f}s")
                        print(f"    Avg Cache Size: {results.get('avg_cache_size', 'N/A'):.1f}")
    
    def _extract_variant_results(self, log: Dict[str, Any], variant_key: str) -> Dict[str, Any]:
        """Extract key metrics for a specific variant."""
        win_rates = log.get('win_rates', {})
        decision_times = log.get('avg_decision_times', {})
        cache_stats = log.get('cache_stats', {})
        
        # Extract win rates for different matchup types
        variant_win_rates = win_rates.get(variant_key, {})
        pvn_win_rate = 0
        pvp_same_win_rate = 0
        pvp_mixed_win_rate = 0
        
        if isinstance(variant_win_rates, str):
            # Parse string format like "particle_vs_naive: {'player1': 0.6, 'player2': 0.4}"
            try:
                if 'particle_vs_naive:' in variant_win_rates:
                    pvn_part = variant_win_rates.split('particle_vs_naive:')[1].split(',')[0].strip()
                    pvn_data = eval(pvn_part)  # Safe here since we control the format
                    pvn_win_rate = pvn_data.get('player1', 0)
            except:
                pass
        elif isinstance(variant_win_rates, dict):
            pvn_win_rate = variant_win_rates.get('particle_vs_naive', {}).get('player1', 0)
            pvp_same_win_rate = variant_win_rates.get('particle_vs_particle_same', {}).get('player1', 0)
            pvp_mixed_win_rate = variant_win_rates.get('particle_vs_particle_mixed', {}).get('player1', 0)
        
        # Extract decision times
        variant_decision_times = decision_times.get(variant_key, {})
        avg_decision_time = 0
        if isinstance(variant_decision_times, str):
            try:
                if 'particle_vs_naive:' in variant_decision_times:
                    time_part = variant_decision_times.split('particle_vs_naive:')[1].split(',')[0].strip()
                    time_data = eval(time_part)
                    avg_decision_time = time_data.get('player1', 0)
            except:
                pass
        elif isinstance(variant_decision_times, dict):
            avg_decision_time = variant_decision_times.get('particle_vs_naive', {}).get('player1', 0)
        
        # Extract cache stats
        variant_cache_stats = cache_stats.get(variant_key, {})
        avg_cache_size = 0
        if isinstance(variant_cache_stats, str):
            try:
                if 'particle_vs_naive:' in variant_cache_stats:
                    cache_part = variant_cache_stats.split('particle_vs_naive:')[1].strip()
                    if cache_part and cache_part != 'N/A (no cache)':
                        cache_data = eval(cache_part)
                        avg_cache_size = cache_data.get('player1', 0)
            except:
                pass
        elif isinstance(variant_cache_stats, dict):
            avg_cache_size = variant_cache_stats.get('particle_vs_naive', {}).get('player1', 0)
        
        return {
            'pvn_win_rate': pvn_win_rate,
            'pvp_same_win_rate': pvp_same_win_rate,
            'pvp_mixed_win_rate': pvp_mixed_win_rate,
            'avg_decision_time': avg_decision_time,
            'avg_cache_size': avg_cache_size
        }
    
    def print_parameter_impact(self):
        """Print parameter impact analysis."""
        if not self.logs:
            print("No logs to analyze.")
            return
            
        # Find sensitivity analysis logs
        sensitivity_logs = [log for log in self.logs if log.get('matchup') == 'parameter_sensitivity_analysis']
        
        if not sensitivity_logs:
            print("No sensitivity analysis logs found.")
            return
        
        print("\nPARAMETER IMPACT ANALYSIS")
        print("=" * 50)
        
        # Aggregate results across all sensitivity logs
        param_impacts = {}
        
        for log in sensitivity_logs:
            parameter_variants = log.get('parameter_variants', {})
            
            for variant_key, variant_info in parameter_variants.items():
                param_path = variant_info.get('parameter_path', 'Unknown')
                variation_type = variant_info.get('variation_type', 'Unknown')
                results = self._extract_variant_results(log, variant_key)
                
                if param_path not in param_impacts:
                    param_impacts[param_path] = {'increase': [], 'decrease': []}
                
                param_impacts[param_path][variation_type].append(results)
        
        # Print impact summary for each parameter
        for param_path, impacts in param_impacts.items():
            print(f"\n{param_path}")
            print("-" * len(param_path))
            
            for variation_type, results_list in impacts.items():
                if not results_list:
                    continue
                    
                avg_win_rate = sum(r['pvn_win_rate'] for r in results_list) / len(results_list)
                avg_decision_time = sum(r['avg_decision_time'] for r in results_list) / len(results_list)
                avg_cache_size = sum(r['avg_cache_size'] for r in results_list) / len(results_list)
                
                print(f"  {variation_type.title()} ({len(results_list)} variants):")
                print(f"    Avg Win Rate: {avg_win_rate:.2%}")
                print(f"    Avg Decision Time: {avg_decision_time:.6f}s")
                print(f"    Avg Cache Size: {avg_cache_size:.1f}")


    def create_visualizations(self, output_dir: str = "results/plots"):
        """Create visualization plots for the analysis results."""
        if not self.logs:
            print("No logs to visualize.")
            return
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Find sensitivity analysis logs
        sensitivity_logs = [log for log in self.logs if log.get('matchup') == 'parameter_sensitivity_analysis']
        
        if not sensitivity_logs:
            print("No sensitivity analysis logs found for visualization.")
            return
        
        # Aggregate data for visualization
        param_data = self._aggregate_visualization_data(sensitivity_logs)
        
        # Create plots
        self._plot_parameter_impact(param_data, output_dir)
        self._plot_win_rate_comparison(param_data, output_dir)
        self._plot_performance_metrics(param_data, output_dir)
        
        print(f"\nVisualizations saved to: {output_dir}/")
    
    def _aggregate_visualization_data(self, sensitivity_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate data for visualization from sensitivity logs."""
        param_data = {}
        
        for log in sensitivity_logs:
            parameter_variants = log.get('parameter_variants', {})
            
            for variant_key, variant_info in parameter_variants.items():
                param_path = variant_info.get('parameter_path', 'Unknown')
                variation_type = variant_info.get('variation_type', 'Unknown')
                original_val = variant_info.get('original_value', 0)
                variant_val = variant_info.get('variant_value', 0)
                
                if param_path not in param_data:
                    param_data[param_path] = {
                        'original_value': original_val,
                        'increase': {'win_rates': [], 'decision_times': [], 'cache_sizes': []},
                        'decrease': {'win_rates': [], 'decision_times': [], 'cache_sizes': []}
                    }
                
                results = self._extract_variant_results(log, variant_key)
                
                param_data[param_path][variation_type]['win_rates'].append(results['pvn_win_rate'])
                param_data[param_path][variation_type]['decision_times'].append(results['avg_decision_time'])
                param_data[param_path][variation_type]['cache_sizes'].append(results['avg_cache_size'])
        
        return param_data
    
    def _plot_parameter_impact(self, param_data: Dict[str, Any], output_dir: str):
        """Create parameter impact bar charts."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Parameter Impact Analysis', fontsize=16, fontweight='bold')
        
        params = list(param_data.keys())
        x_pos = np.arange(len(params))
        width = 0.35
        
        # Win Rate Impact
        ax1 = axes[0, 0]
        increase_wins = [np.mean(param_data[p]['increase']['win_rates']) if param_data[p]['increase']['win_rates'] else 0 for p in params]
        decrease_wins = [np.mean(param_data[p]['decrease']['win_rates']) if param_data[p]['decrease']['win_rates'] else 0 for p in params]
        
        ax1.bar(x_pos - width/2, increase_wins, width, label='Increase 25%', alpha=0.8, color='green')
        ax1.bar(x_pos + width/2, decrease_wins, width, label='Decrease 25%', alpha=0.8, color='red')
        ax1.set_xlabel('Parameters')
        ax1.set_ylabel('Win Rate')
        ax1.set_title('Win Rate Impact')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([p.split('.')[-1] for p in params], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Decision Time Impact
        ax2 = axes[0, 1]
        increase_times = [np.mean(param_data[p]['increase']['decision_times']) if param_data[p]['increase']['decision_times'] else 0 for p in params]
        decrease_times = [np.mean(param_data[p]['decrease']['decision_times']) if param_data[p]['decrease']['decision_times'] else 0 for p in params]
        
        ax2.bar(x_pos - width/2, increase_times, width, label='Increase 25%', alpha=0.8, color='green')
        ax2.bar(x_pos + width/2, decrease_times, width, label='Decrease 25%', alpha=0.8, color='red')
        ax2.set_xlabel('Parameters')
        ax2.set_ylabel('Decision Time (seconds)')
        ax2.set_title('Decision Time Impact')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([p.split('.')[-1] for p in params], rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Cache Size Impact
        ax3 = axes[1, 0]
        increase_cache = [np.mean(param_data[p]['increase']['cache_sizes']) if param_data[p]['increase']['cache_sizes'] else 0 for p in params]
        decrease_cache = [np.mean(param_data[p]['decrease']['cache_sizes']) if param_data[p]['decrease']['cache_sizes'] else 0 for p in params]
        
        ax3.bar(x_pos - width/2, increase_cache, width, label='Increase 25%', alpha=0.8, color='green')
        ax3.bar(x_pos + width/2, decrease_cache, width, label='Decrease 25%', alpha=0.8, color='red')
        ax3.set_xlabel('Parameters')
        ax3.set_ylabel('Cache Size')
        ax3.set_title('Cache Size Impact')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([p.split('.')[-1] for p in params], rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Combined Impact (normalized)
        ax4 = axes[1, 1]
        # Normalize values for comparison
        norm_increase_wins = [(w - min(increase_wins + decrease_wins)) / (max(increase_wins + decrease_wins) - min(increase_wins + decrease_wins)) if max(increase_wins + decrease_wins) != min(increase_wins + decrease_wins) else 0 for w in increase_wins]
        norm_decrease_wins = [(w - min(increase_wins + decrease_wins)) / (max(increase_wins + decrease_wins) - min(increase_wins + decrease_wins)) if max(increase_wins + decrease_wins) != min(increase_wins + decrease_wins) else 0 for w in decrease_wins]
        
        ax4.bar(x_pos - width/2, norm_increase_wins, width, label='Increase 25%', alpha=0.8, color='green')
        ax4.bar(x_pos + width/2, norm_decrease_wins, width, label='Decrease 25%', alpha=0.8, color='red')
        ax4.set_xlabel('Parameters')
        ax4.set_ylabel('Normalized Impact (0-1)')
        ax4.set_title('Normalized Win Rate Impact')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([p.split('.')[-1] for p in params], rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/parameter_impact_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_win_rate_comparison(self, param_data: Dict[str, Any], output_dir: str):
        """Create win rate comparison scatter plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Win Rate Comparison Analysis', fontsize=16, fontweight='bold')
        
        params = list(param_data.keys())
        
        # Original vs Modified Win Rates
        original_values = []
        modified_values = []
        change_types = []
        param_names = []
        
        for param in params:
            orig_val = param_data[param]['original_value']
            
            for change_type in ['increase', 'decrease']:
                win_rates = param_data[param][change_type]['win_rates']
                if win_rates:
                    for win_rate in win_rates:
                        original_values.append(orig_val)
                        modified_values.append(win_rate)
                        change_types.append(change_type)
                        param_names.append(param.split('.')[-1])
        
        # Scatter plot by parameter
        unique_params = list(set(param_names))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_params)))
        param_colors = {param: colors[i] for i, param in enumerate(unique_params)}
        
        for i, param in enumerate(unique_params):
            mask = [p == param for p in param_names]
            x_vals = [original_values[j] for j, m in enumerate(mask) if m]
            y_vals = [modified_values[j] for j, m in enumerate(mask) if m]
            c_types = [change_types[j] for j, m in enumerate(mask) if m]
            
            for x, y, c_type in zip(x_vals, y_vals, c_types):
                marker = '^' if c_type == 'increase' else 'v'
                ax1.scatter(x, y, c=[param_colors[param]], marker=marker, s=100, alpha=0.7)
        
        ax1.set_xlabel('Original Parameter Value')
        ax1.set_ylabel('Win Rate')
        ax1.set_title('Win Rate vs Parameter Value')
        ax1.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color=param_colors[param], label=param) for param in unique_params
        ] + [
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=10, label='Increase'),
            plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='gray', markersize=10, label='Decrease')
        ]
        ax1.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Win rate distribution by change type
        increase_rates = []
        decrease_rates = []
        
        for param in params:
            increase_rates.extend(param_data[param]['increase']['win_rates'])
            decrease_rates.extend(param_data[param]['decrease']['win_rates'])
        
        ax2.hist(increase_rates, bins=20, alpha=0.7, label='Increase 25%', color='green', density=True)
        ax2.hist(decrease_rates, bins=20, alpha=0.7, label='Decrease 25%', color='red', density=True)
        ax2.set_xlabel('Win Rate')
        ax2.set_ylabel('Density')
        ax2.set_title('Win Rate Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/win_rate_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_metrics(self, param_data: Dict[str, Any], output_dir: str):
        """Create performance metrics visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Metrics Analysis', fontsize=16, fontweight='bold')
        
        params = list(param_data.keys())
        
        # Decision Time vs Win Rate scatter
        ax1 = axes[0, 0]
        for param in params:
            for change_type in ['increase', 'decrease']:
                win_rates = param_data[param][change_type]['win_rates']
                decision_times = param_data[param][change_type]['decision_times']
                
                if win_rates and decision_times:
                    color = 'green' if change_type == 'increase' else 'red'
                    ax1.scatter(decision_times, win_rates, alpha=0.7, c=color, s=50, label=f"{param.split('.')[-1]} {change_type}")
        
        ax1.set_xlabel('Decision Time (seconds)')
        ax1.set_ylabel('Win Rate')
        ax1.set_title('Decision Time vs Win Rate')
        ax1.grid(True, alpha=0.3)
        
        # Cache Size vs Win Rate scatter
        ax2 = axes[0, 1]
        for param in params:
            for change_type in ['increase', 'decrease']:
                win_rates = param_data[param][change_type]['win_rates']
                cache_sizes = param_data[param][change_type]['cache_sizes']
                
                if win_rates and cache_sizes:
                    color = 'green' if change_type == 'increase' else 'red'
                    ax2.scatter(cache_sizes, win_rates, alpha=0.7, c=color, s=50, label=f"{param.split('.')[-1]} {change_type}")
        
        ax2.set_xlabel('Cache Size')
        ax2.set_ylabel('Win Rate')
        ax2.set_title('Cache Size vs Win Rate')
        ax2.grid(True, alpha=0.3)
        
        # Decision Time Distribution
        ax3 = axes[1, 0]
        all_increase_times = []
        all_decrease_times = []
        
        for param in params:
            all_increase_times.extend(param_data[param]['increase']['decision_times'])
            all_decrease_times.extend(param_data[param]['decrease']['decision_times'])
        
        ax3.hist(all_increase_times, bins=20, alpha=0.7, label='Increase 25%', color='green', density=True)
        ax3.hist(all_decrease_times, bins=20, alpha=0.7, label='Decrease 25%', color='red', density=True)
        ax3.set_xlabel('Decision Time (seconds)')
        ax3.set_ylabel('Density')
        ax3.set_title('Decision Time Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Cache Size Distribution
        ax4 = axes[1, 1]
        all_increase_cache = []
        all_decrease_cache = []
        
        for param in params:
            all_increase_cache.extend(param_data[param]['increase']['cache_sizes'])
            all_decrease_cache.extend(param_data[param]['decrease']['cache_sizes'])
        
        ax4.hist(all_increase_cache, bins=20, alpha=0.7, label='Increase 25%', color='green', density=True)
        ax4.hist(all_decrease_cache, bins=20, alpha=0.7, label='Decrease 25%', color='red', density=True)
        ax4.set_xlabel('Cache Size')
        ax4.set_ylabel('Density')
        ax4.set_title('Cache Size Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function to run results analysis."""
    import sys
    
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    create_plots = '--plots' in sys.argv or '--visualize' in sys.argv
    
    analyzer = ResultsAnalyzer(results_dir)
    analyzer.load_logs()
    analyzer.print_summary()
    analyzer.print_parameter_impact()
    
    if create_plots:
        analyzer.create_visualizations()


if __name__ == "__main__":
    main()