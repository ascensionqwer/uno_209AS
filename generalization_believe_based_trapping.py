import itertools
from math import comb

# --- IMPORT MODULES ---
# Assumes 'cards.py' and 'belief.py' are in the same directory
from cards import RED, GREEN, Card
from belief import Belief

# --- CONFIGURATION ---
MICRO_DECK = [
    (RED, 1), (RED, 2), (RED, 3),
    (GREEN, 1), (GREEN, 2), (GREEN, 3),
]
LOG_FILE = "exhaustive_scenario_log.txt"

# --- HELPER FUNCTIONS ---
def is_legal(card: Card, top_card: Card) -> bool:
    """Standard UNO rules: Match Color or Match Value."""
    return card[0] == top_card[0] or card[1] == top_card[1]

def calculate_q_value(belief: Belief, move: Card) -> float:
    """
    Returns 1.0 if the move forces opponent to DRAW (Trap).
    Returns 0.0 if the opponent CAN PLAY (No Trap).
    """
    new_top_card = move
    can_play = False
    
    # Check if ANY card in the Unknown Set L is legal on the new top card
    for card in belief.L:
        if is_legal(card, new_top_card):
            can_play = True
            break
            
    return 0.0 if can_play else 1.0

def run_exhaustive_search():
    print(f"Starting Fully Exhaustive Search on 6-Card Deck...")
    print(f"Logging to {LOG_FILE}...\n")
    
    total_scenarios = 0
    optimal_found = 0
    no_preference = 0  # Changed: lumps both "all traps" and "all bad" together
    
    seen_states = set()

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("=== EXHAUSTIVE UNO MICRO-DECK ANALYSIS ===\n")
        f.write("Deck: R1, R2, R3, G1, G2, G3\n")
        f.write("Criteria: H1 must have >= 2 choices. Finding TRAPS (Q=1.0).\n\n")

        for perm in itertools.permutations(MICRO_DECK):
            for h1_len in range(2, 5): 
                remaining_after_h1 = 6 - h1_len
                for h2_len in range(1, remaining_after_h1): 
                    remaining_after_h2 = remaining_after_h1 - h2_len
                    for p_len in range(1, remaining_after_h2 + 1):
                        
                        idx = 0
                        H_1 = list(perm[idx : idx + h1_len])
                        idx += h1_len
                        H_2_truth = list(perm[idx : idx + h2_len])
                        idx += h2_len
                        P = list(perm[idx : idx + p_len])
                        D_g = list(perm[idx + p_len:])
                        P_t = P[-1]
                        
                        legal_plays = [c for c in H_1 if is_legal(c, P_t)]
                        if len(legal_plays) < 2:
                            continue 
                        
                        state_id = (tuple(sorted(H_1)), len(H_2_truth), tuple(P))
                        if state_id in seen_states:
                            continue
                        seen_states.add(state_id)

                        total_scenarios += 1
                        observation = (H_1, len(H_2_truth), len(D_g), P, P_t, "Active")
                        b = Belief(observation, deck_template=MICRO_DECK)
                        
                        # Calculate probability for this scenario
                        num_worlds = comb(len(b.L), len(H_2_truth))
                        prob_scenario = 1.0 / num_worlds if num_worlds > 0 else 0.0
                        
                        moves_data = []
                        
                        has_trap = False
                        has_bad = False
                        best_move = None
                        best_val = -1.0

                        for move in legal_plays:
                            val = calculate_q_value(b, move)
                            moves_data.append((move, val))
                            if val == 1.0: 
                                has_trap = True
                            if val == 0.0: 
                                has_bad = True
                            if val > best_val:
                                best_val = val
                                best_move = move
                        
                        # Determine Result Type: CHANGED LOGIC
                        if has_trap and has_bad:
                            optimal_found += 1
                            result_str = f"YES (Optimal Play: {best_move})"
                        else:  # Changed: lumps "all traps" and "all bad" together
                            no_preference += 1
                            if has_trap:
                                result_str = "NO (All moves trap)"
                            else:
                                result_str = "NO (All moves allow play)"

                        f.write(f"Scenario #{total_scenarios}\n")
                        f.write(f"  [Obs] Legal Plays: {legal_plays}\n")
                        f.write(f"  [Worlds] {num_worlds} possible world(s), prob per world: {prob_scenario:.6f}\n")
                        f.write(f"  [Q-Values] {moves_data}\n")
                        f.write(f"  [Optimal?] {result_str}\n")
                        f.write("-" * 50 + "\n")

    print(f"Done.")
    print(f"Total Unique Scenarios Analyzed: {total_scenarios}")
    print(f"Scenarios with OPTIMAL Preference: {optimal_found}")
    print(f"Scenarios with NO Preference: {no_preference}")
    print(f"Full logs written to {LOG_FILE}")


if __name__ == "__main__":
    run_exhaustive_search()