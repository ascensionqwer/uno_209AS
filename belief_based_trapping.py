from math import comb
from cards import RED, GREEN, Card
from belief import Belief

# --- HELPER: Simulation Logic ---
def is_legal(card: Card, top_card: Card) -> bool:
    """UNO rule: Match Color or Match Value."""
    return card[0] == top_card[0] or card[1] == top_card[1]

def calculate_q_value(belief: Belief, move: Card) -> float:
    """
    Calculates the 'Trapping Value' of a move.
    1.0 = Opponent is forced to DRAW (Good).
    0.0 = Opponent can PLAY (Bad).
    """
    new_top_card = move
    # The opponent holds a card from the Unknown Set L.
    # We check if ANY card in L is legal on the new top card.
    # (In this deterministic scenario, L has only 1 card, so it's exact).
    
    can_play = False
    for card in belief.L:
        if is_legal(card, new_top_card):
            can_play = True
            break
            
    return 0.0 if can_play else 1.0

# --- MAIN SCENARIO ---
def run_formatted_analysis():
    # 1. SETUP: The Micro-Deck (6 Cards)
    micro_deck = [
        (RED, 1), (RED, 2), (RED, 3),
        (GREEN, 1), (GREEN, 2), (GREEN, 3),
    ]

    # 2. STATE DEFINITION
    # H1: We hold R1, G2
    # P: R2, R3, G1 (Top) are played
    # Truth: Opponent holds G3 (the only card left)
    H_1 = [(RED, 1), (GREEN, 2)]
    H_2_truth = [(GREEN, 3)]
    D_g_truth = [] # Deck is empty
    
    P_t = (GREEN, 1)
    # Note: P includes the top card for calculation
    P = [(RED, 2), (RED, 3), P_t] 
    
    h2_size = 1
    dg_size = 0
    
    # 3. OBSERVATION
    observation = (H_1, h2_size, dg_size, P, P_t, "Active")
    
    # 4. INITIALIZE BELIEF
    b = Belief(observation, deck_template=micro_deck)
    
    # --- PRINTING THE FORMATTED OUTPUT ---
    print(f"Scenario #ColorTrap_01")
    
    # [God] View (The Ground Truth)
    print(f"  [God] H1_truth: {H_1}")
    print(f"  [God] H2_truth: {H_2_truth}")
    print(f"  [God] Dg_truth: {D_g_truth}")
    print(f"  [God] P: {P}")
    print(f"  [God] P_t: {P_t}")
    
    # [Obs] View (What the Agent Sees)
    print(f"  [Obs] H1: {H_1}")
    print(f"  [Obs] |H2|: {h2_size} |D_g|: {dg_size}")
    print(f"  [Obs] P: {P}")
    print(f"  [Obs] P_t: {P_t}")
    
    legal_plays = [c for c in H_1 if is_legal(c, P_t)]
    print(f"  [Obs] Legal plays: {legal_plays}")
    
    # [Belief] Internals
    # Using the __repr__ from your class, plus custom details
    print(f"  [Belief] {b}") 
    print(f"  [Belief] L (unknown cards): {b.L}")
    
    # N(P_t) is the subset of L that matches the CURRENT top card (G1)
    # G3 matches G1, so N(P_t) should contain G3.
    print(f"  [Belief] N(P_t) (legal unknown on current top): {b.N_Pt}")
    
    # Generate Consistent Worlds based on L
    # Since L has 1 card and H2 needs 1 card, there is only 1 world.
    num_worlds = comb(len(b.L), h2_size)
    prob_per_world = 1.0 / num_worlds if num_worlds > 0 else 0.0
    
    print(f"  [Belief] Number of consistent worlds: {num_worlds}")
    print(f"  [Belief] Probability per world: {prob_per_world:.6f} (1/{num_worlds})")
    
    for world_idx in range(1, num_worlds + 1):
        print(f"    World #{world_idx}: H2={b.L}, D_g=[] (prob={prob_per_world:.6f})")
        
    # Q-Values Calculation
    print(f"  [Belief] Exact Q-values under Trapping_Strategy:")
    
    best_move = None
    best_val = -1.0
    
    for move in legal_plays:
        val = calculate_q_value(b, move)
        print(f"    {move} -> {val:.1f}")
        
        if val > best_val:
            best_val = val
            best_move = move
            
    # Final Verdict
    has_optimal = "YES" if best_val == 1.0 else "NO"
    print(f"  [Obs] Has optimal play: {has_optimal}, best = {best_move}")

if __name__ == "__main__":
    run_formatted_analysis()