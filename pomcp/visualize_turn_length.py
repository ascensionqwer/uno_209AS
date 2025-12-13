#!/usr/bin/env python3
"""Simple Turn Length Visualization for UNO Results

Aggregates turn statistics across all simulations and creates clean visualizations.
"""

import json
import os
import glob
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np

# Color scheme
COLORS = {
    'naive': '#FF6B6B',      # Red
    'pomcp': '#1864AB',      # Blue  
    'avg': '#74C0FC',         # Light blue
    'min_max': '#868E96',     # Gray
    'grid': '#E9ECEF'        # Light gray for grids
}

def load_all_turn_data():
    """Load and aggregate turn data from all result files."""
    results_dir = "results"
    if not os.path.exists(results_dir):
        print(f"Results directory '{results_dir}' not found.")
        return None

    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    if not json_files:
        print(f"No JSON files found in '{results_dir}'.")
        return None

    # Aggregate turn data by matchup type
    aggregated_data = {
        'naive_vs_naive': {'turns': [], 'count': 0},
        'naive_vs_pomcp': {'turns': [], 'count': 0}, 
        'pomcp_vs_pomcp': {'turns': [], 'count': 0}
    }

    total_files = 0
    for file_path in sorted(json_files):
        try:
            with open(file_path, "r") as f:
                log_data = json.load(f)
                total_files += 1
                
                # Extract turn data from different structures
                turn_data = extract_turns_from_file(log_data)
                if turn_data:
                    for matchup, turns in turn_data.items():
                        if matchup in aggregated_data:
                            aggregated_data[matchup]['turns'].extend(turns)
                            aggregated_data[matchup]['count'] += len(turns)
                            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    print(f"Processed {total_files} files")
    return aggregated_data

def extract_turns_from_file(log_data):
    """Extract turn data from different file structures."""
    turn_data = {
        'naive_vs_naive': [],
 
        'naive_vs_pomcp': [],
        'pomcp_vs_pomcp': []
    }
    
    # Check for nested structure (variant files)
    player_wins = log_data.get("player_wins", {})
    
    for variant_key, variant_data in player_wins.items():
        if isinstance(variant_data, dict):
            for matchup_key, matchup_data in variant_data.items():
                if isinstance(matchup_data, dict) and "turn_stats" in matchup_data:
                    stats = matchup_data["turn_stats"]
                    if isinstance(stats, dict):
                        avg_turns = stats.get('average', 0)
                        total_turns = stats.get('total', 0)
                        
                        # Convert to list of individual turns (approximate)
                        if total_turns > 0 and avg_turns > 0:
                            num_games = total_turns / avg_turns if avg_turns > 0 else 1
                            turns_list = [avg_turns] * int(num_games)
                        else:
                            turns_list = []
                        
                        # Map to our categories
                        if 'naive_vs_naive' in matchup_key:
                            turn_data['naive_vs_naive'].extend(turns_list)
                        elif 'naive_vs_particle_policy' in matchup_key:
                            turn_data['naive_vs_pomcp'].extend(turns_list)
                        elif 'particle_policy_vs_particle_policy' in matchup_key:
                            turn_data['pomcp_vs_pomcp'].extend(turns_list)
    
    return turn_data if any(turn_data.values()) else None

def create_turn_length_visualization():
    """Create clean turn length visualization."""
    data = load_all_turn_data()
    if not data:
        print("No turn data found.")
        return

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('UNO Game Length Analysis', fontsize=16, fontweight='bold')

    # 1. Average turns by matchup
    plot_average_turns(ax1, data)
    
    # 2. Turn distribution  
    plot_turn_distribution(ax2, data)
    
    # 3. Box plot comparison
    plot_box_comparison(ax3, data)
    
    # 4. Summary statistics
    plot_summary_stats(ax4, data)

    plt.tight_layout()
    plt.savefig('results/turn_length_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Turn length visualization saved to: results/turn_length_analysis.png")

def plot_average_turns(ax, data):
    """Plot average turns for each matchup."""
    matchups = []
    avg_turns = []
    colors = []
    
    for matchup, matchup_data in data.items():
        if matchup_data['turns']:
            matchups.append(matchup.replace('_', ' ').replace('naive', 'Naive').replace('pomcp', 'POMCP'))
            avg_turns.append(np.mean(matchup_data['turns']))
            colors.append(COLORS['naive'] if 'naive' in matchup else COLORS['pomcp'])
    
    bars = ax.bar(matchups, avg_turns, color=colors, alpha=0.8)
    
    # Add value labels
    for bar, value in zip(bars, avg_turns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Average Turns')
    ax.set_title('Average Game Length by Matchup')
    ax.grid(True, alpha=0.3, axis='y', color=COLORS['grid'])

def plot_turn_distribution(ax, data):
    """Plot distribution of turn counts."""
    all_turns = []
    labels = []
    
    for matchup, matchup_data in data.items():
        if matchup_data['turns']:
            all_turns.extend(matchup_data['turns'])
            labels.extend([matchup.replace('_', ' ').replace('naive', 'Naive').replace('pomcp', 'POMCP')] * len(matchup_data['turns']))
    
    if all_turns:
        ax.hist(all_turns, bins=20, alpha=0.7, color=COLORS['avg'], edgecolor='black')
        ax.set_xlabel('Number of Turns')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Game Lengths')
        ax.grid(True, alpha=0.3, axis='y', color=COLORS['grid'])

def plot_box_comparison(ax, data):
    """Plot box plot comparison."""
    matchups = []
    turn_lists = []
    
    for matchup, matchup_data in data.items():
        if matchup_data['turns']:
            matchups.append(matchup.replace('_', ' ').replace('naive', 'Naive').replace('pomcp', 'POMCP'))
            turn_lists.append(matchup_data['turns'])
    
    if turn_lists:
        ax.boxplot(turn_lists, labels=matchups, patch_artist=True)
        
        # Color the boxes
        colors = [COLORS['naive'] if 'naive' in m else COLORS['pomcp'] for m in matchups]
        for patch, color in zip(ax.artists, colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax.set_ylabel('Number of Turns')
    ax.set_title('Game Length Distribution by Matchup')
    ax.grid(True, alpha=0.3, axis='y', color=COLORS['grid'])

def plot_summary_stats(ax, data):
    """Plot summary statistics."""
    stats_text = ["GAME LENGTH SUMMARY", ""]
    
    for matchup, matchup_data in data.items():
        if matchup_data['turns']:
            turns = matchup_data['turns']
            clean_name = matchup.replace('_', ' ').replace('naive', 'Naive').replace('pomcp', 'POMCP')
            
            stats_text.extend([
                f"{clean_name}:",
                f"  Games: {matchup_data['count']}",
                f"  Avg: {np.mean(turns):.1f}",
                f"  Min: {np.min(turns)}",
                f"  Max: {np.max(turns)}",
                f"  Std: {np.std(turns):.1f}",
                ""
            ])
    
    # Overall stats
    all_turns = []
    for matchup_data in data.values():
        all_turns.extend(matchup_data['turns'])
    
    if all_turns:
        stats_text.extend([
            "OVERALL:",
            f"  Total Games: {len(all_turns)}",
            f"  Avg Turns: {np.mean(all_turns):.1f}",
            f"  Range: {np.min(all_turns)} - {np.max(all_turns)}",
            f"  Std Dev: {np.std(all_turns):.1f}"
        ])
    
    ax.text(0.05, 0.95, '\n'.join(stats_text), transform=ax.transAxes, 
           fontsize=9, verticalalignment='top', fontfamily='monospace')
    ax.set_title('Summary Statistics')
    ax.axis('off')

if __name__ == "__main__":
    create_turn_length_visualization()
    print("Turn length analysis complete!")