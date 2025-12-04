import matplotlib.pyplot as plt
import numpy as np
import re

def parse_log_file(filename):
    with open(filename, 'r') as f:
        content = f.read()
        
    experiments = []
    
    # Regex to find experiment blocks
    # === Experiment: Optimal (P1) vs Random (P2) ===
    # ...
    # Overall P1 Wins: 2853 (57.1%)
    # ...
    #   When P1 Starts (2500 games): P1 Wins 1914 (76.6%)
    #   When P2 Starts (2500 games): P2 Wins 1561 (62.4%)
    
    blocks = content.split("=== Experiment: ")
    for block in blocks[1:]: # Skip first empty split
        lines = block.strip().split('\n')
        name = lines[0].strip(" =")
        
        data = {"name": name}
        
        for line in lines:
            if "Overall P1 Wins:" in line:
                # Overall P1 Wins: 2853 (57.1%)
                match = re.search(r"Overall P1 Wins: \d+ \((\d+\.\d+)%\)", line)
                if match: data["overall_p1"] = float(match.group(1))
                
            elif "When P1 Starts" in line:
                # When P1 Starts (2500 games): P1 Wins 1914 (76.6%)
                match = re.search(r"P1 Wins \d+ \((\d+\.\d+)%\)", line)
                if match: data["p1_start_p1"] = float(match.group(1))
                
            elif "When P2 Starts" in line:
                # When P2 Starts (2500 games): P2 Wins 1561 (62.4%)
                # Note: This line reports P2 Wins. We want P1 Wins (100 - P2 Wins)
                match = re.search(r"P2 Wins \d+ \((\d+\.\d+)%\)", line)
                if match: 
                    p2_win_rate = float(match.group(1))
                    data["p2_start_p1"] = 100.0 - p2_win_rate
                    
        experiments.append(data)
        
    return experiments

def plot_results(experiments):
    labels = [e["name"].replace(" (P1)", "").replace(" (P2)", "") for e in experiments]
    overall = [e["overall_p1"] for e in experiments]
    p1_starts = [e["p1_start_p1"] for e in experiments]
    p2_starts = [e["p2_start_p1"] for e in experiments]

    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, p1_starts, width, label='When P1 Starts', color='#4CAF50')
    rects2 = ax.bar(x, overall, width, label='Overall', color='#2196F3')
    rects3 = ax.bar(x + width, p2_starts, width, label='When P2 Starts', color='#F44336')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Player 1 Win Rate (%)')
    ax.set_title('Mini Uno: Player 1 Win Rate by Starting Player (5000 Games)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Add a horizontal line at 50%
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)

    ax.set_ylim(0, 100)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()

    plt.savefig('stage_1_mini_uno/win_rate_analysis.png')
    print("Graph saved to stage_1_mini_uno/win_rate_analysis.png")

if __name__ == "__main__":
    data = parse_log_file("stage_1_mini_uno/experiments_log_5000.txt")
    plot_results(data)
