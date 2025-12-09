import matplotlib.pyplot as plt
import numpy as np

def plot_uno_results(winner_list):
    """
    Generates two graphs:
    1. A Split Cumulative Win Chart (P1 grows Up, P2 grows Down).
    2. A Total Win Rate Bar Chart.
    """

    # --- Data Preparation ---
    total_games = len(winner_list)
    games_range = np.arange(0, total_games + 1)

    p1_wins = winner_list.count(1)
    p2_wins = winner_list.count(2)

    # Calculate Cumulative Wins separately
    # Example: [0, 1, 1, 2, 2, 3...]
    p1_progress = [0] + [1 if w == 1 else 0 for w in winner_list]
    p2_progress = [0] + [1 if w == 2 else 0 for w in winner_list]

    p1_cumulative = np.cumsum(p1_progress)
    p2_cumulative = np.cumsum(p2_progress)

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 12))
    fig.suptitle(f'Uno Simulation Analysis ({total_games} Games)', fontsize=18, weight='bold')

    # ========================================================
    # GRAPH 1: Split Cumulative Wins (P1 Up / P2 Down)
    # ========================================================

    # Player 1 (Positive Direction)
    ax1.plot(games_range, p1_cumulative, color='#2980b9', linewidth=2, label='P1 Cumulative Wins')
    ax1.fill_between(games_range, 0, p1_cumulative, color='#3498db', alpha=0.5)

    # Player 2 (Negative Direction)
    # We invert the values to make them "face down"
    ax1.plot(games_range, -p2_cumulative, color='#c0392b', linewidth=2, label='P2 Cumulative Wins')
    ax1.fill_between(games_range, 0, -p2_cumulative, color='#e74c3c', alpha=0.5)

    # Center Line
    ax1.axhline(0, color='black', linestyle='-', linewidth=1)

    # Annotations at the end of the graph
    ax1.text(total_games + 1, p1_cumulative[-1], f"{p1_cumulative[-1]} Wins",
             color='#2980b9', va='center', weight='bold')
    ax1.text(total_games + 1, -p2_cumulative[-1], f"{p2_cumulative[-1]} Wins",
             color='#c0392b', va='center', weight='bold')

    # Styling
    ax1.set_title('Cumulative Wins Tracker (Split Direction)', fontsize=14)
    ax1.set_ylabel('Total Wins\n(Up = P1, Down = P2)', fontsize=12)
    ax1.set_xlabel('Game Number', fontsize=12)
    ax1.set_xlim(0, total_games + (total_games * 0.1)) # Add space for text labels

    # Set Y-Limits strictly to Total Games (e.g., -100 to 100)
    # This visualizes the wins relative to the maximum possible games
    ax1.set_ylim(-total_games, total_games)

    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='upper left')


    # ========================================================
    # GRAPH 2: Overall Win Rate (Same as before)
    # ========================================================
    players = ['Player 1', 'Player 2']
    counts = [p1_wins, p2_wins]
    percentages = [(p1_wins/total_games)*100, (p2_wins/total_games)*100]
    bar_colors = ['#5dade2', '#ec7063']

    bars = ax2.bar(players, percentages, color=bar_colors, edgecolor='#444444', linewidth=1.2, width=0.5)

    ax2.set_title('Final Win Distribution', fontsize=14)
    ax2.set_ylabel('Win Percentage (%)')
    ax2.set_ylim(0, 105)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)

    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                 f'{height:.1f}%',
                 ha='center', va='bottom', fontsize=14, weight='bold', color='#333333')
        ax2.text(bar.get_x() + bar.get_width()/2., height/2,
                 f'{counts[i]} Wins',
                 ha='center', va='center', fontsize=12, color='white', weight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()