import random
import itertools
from typing import List, Tuple, Set
from cards import RED, GREEN, BLUE, YELLOW, Card
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
    can_play = False
    for card in belief.L:
        if is_legal(card, new_top_card):
            can_play = True
            break
    return 0.0 if can_play else 1.0

def update_belief_after_opponent_play(belief: Belief, played_card: Card):
    """
    Updates belief after opponent plays a card.
    We learn that opponent had this card, so remove it from L.
    """
    current_L = set(belief.L)
    if played_card in current_L:
        current_L.remove(played_card)
    return current_L

def update_belief_after_opponent_draw(belief: Belief, top_card: Card, drawn_card: Card = None):
    """
    Updates the belief state after observing the opponent DRAW instead of PLAY.
    Eliminates all cards from L that would have been legal on top_card.
    """
    current_L = set(belief.L)
    legal_on_top = {card for card in current_L if is_legal(card, top_card)}
    new_L = current_L - legal_on_top
    
    if drawn_card is not None:
        new_L.add(drawn_card)
    
    return new_L, legal_on_top

# --- RANDOM STATE GENERATOR ---
def generate_random_state(deck_template: List[Card], 
                         h1_size: int = 3, 
                         h2_size: int = 3,
                         played_min: int = 1,
                         played_max: int = 5):
    """
    Generates a random valid UNO game state.
    
    Returns:
        H_1: Our hand
        H_2_truth: Opponent's hand (ground truth)
        D_g_truth: Remaining deck
        P: Played cards (including top)
        P_t: Current top card
    """
    # Shuffle and deal cards
    deck = deck_template.copy()
    random.shuffle(deck)
    
    # Deal hands
    H_1 = deck[:h1_size]
    H_2_truth = deck[h1_size:h1_size + h2_size]
    
    # Played pile (including top card)
    played_count = random.randint(played_min, played_max)
    remaining_cards = deck[h1_size + h2_size:]
    
    if len(remaining_cards) < played_count:
        played_count = len(remaining_cards)
    
    P = remaining_cards[:played_count]
    P_t = P[-1] if P else deck[h1_size + h2_size]  # Top card
    
    # Remaining deck
    D_g_truth = remaining_cards[played_count:]
    
    return H_1, H_2_truth, D_g_truth, P, P_t

# --- VERIFICATION FUNCTIONS ---
def verify_opponent_play(scenario_num: int, deck_template: List[Card]):
    """
    Generate random state, simulate opponent playing a legal card,
    and verify belief update is correct.
    """
    print(f"\n{'='*70}")
    print(f"SCENARIO #{scenario_num}: Random State - Opponent PLAYS")
    print(f"{'='*70}")
    
    H_1, H_2_truth, D_g_truth, P, P_t = generate_random_state(deck_template)
    
    # Check if opponent can play
    legal_h2 = [c for c in H_2_truth if is_legal(c, P_t)]
    
    if not legal_h2:
        # Opponent can't play, skip this scenario
        print("  [Skip] Opponent has no legal plays")
        return False
    
    # Opponent plays a random legal card
    played_card = random.choice(legal_h2)
    
    h2_size = len(H_2_truth)
    dg_size = len(D_g_truth)
    
    print(f"\n  [God] H1_truth: {H_1}")
    print(f"  [God] H2_truth: {H_2_truth}")
    print(f"  [God] Dg_truth: {D_g_truth}")
    print(f"  [God] P: {P}")
    print(f"  [God] P_t: {P_t}")
    
    # Initial observation
    observation = (H_1, h2_size, dg_size, P, P_t, "Active")
    b = Belief(observation, deck_template=deck_template)
    
    print(f"\n  [Initial Belief]")
    print(f"    L (unknown): {sorted(b.L)}")
    print(f"    N(P_t) (legal on {P_t}): {sorted(b.N_Pt)}")
    print(f"    |L|={len(b.L)}, |N(P_t)|={len(b.N_Pt)}")
    
    # Check if all H2 cards are in L
    h2_in_L = all(c in b.L for c in H_2_truth)
    print(f"    All H2 cards in L: {h2_in_L}")
    
    # Opponent plays
    print(f"\n  [Observation] Opponent PLAYS: {played_card}")
    print(f"    Legal on {P_t}? {is_legal(played_card, P_t)}")
    
    # Update belief
    new_L = update_belief_after_opponent_play(b, played_card)
    
    print(f"\n  [Updated Belief]")
    print(f"    Removed: {played_card}")
    print(f"    New L: {sorted(new_L)}")
    print(f"    Size: {len(b.L)} -> {len(new_L)}")
    
    # Verify correctness
    H_2_after_play = [c for c in H_2_truth if c != played_card]
    remaining_in_L = all(c in new_L for c in H_2_after_play)
    no_extra = len([c for c in H_2_after_play if c in new_L]) == len([c for c in new_L if c in H_2_after_play or c in D_g_truth])
    
    print(f"\n  [Verification]")
    print(f"    H2 after play: {H_2_after_play}")
    print(f"    All remaining H2 in new L: {remaining_in_L}")
    print(f"    Played card removed: {played_card not in new_L}")
    
    return True

def verify_opponent_draw(scenario_num: int, deck_template: List[Card]):
    """
    Generate random state where opponent must draw,
    and verify belief update eliminates correct cards.
    """
    print(f"\n{'='*70}")
    print(f"SCENARIO #{scenario_num}: Random State - Opponent DRAWS")
    print(f"{'='*70}")
    
    # Try to generate a state where opponent can't play
    max_attempts = 50
    for attempt in range(max_attempts):
        H_1, H_2_truth, D_g_truth, P, P_t = generate_random_state(deck_template)
        
        # Check if opponent can play
        legal_h2 = [c for c in H_2_truth if is_legal(c, P_t)]
        
        if not legal_h2 and D_g_truth:  # Opponent can't play and deck is not empty
            break
    else:
        print("  [Skip] Could not generate state where opponent must draw")
        return False
    
    h2_size = len(H_2_truth)
    dg_size = len(D_g_truth)
    
    print(f"\n  [God] H1_truth: {H_1}")
    print(f"  [God] H2_truth: {H_2_truth}")
    print(f"  [God] Dg_truth: {D_g_truth}")
    print(f"  [God] P: {P}")
    print(f"  [God] P_t: {P_t}")
    
    # Initial observation
    observation = (H_1, h2_size, dg_size, P, P_t, "Active")
    b = Belief(observation, deck_template=deck_template)
    
    print(f"\n  [Initial Belief]")
    print(f"    L (unknown): {sorted(b.L)}")
    print(f"    N(P_t) (legal on {P_t}): {sorted(b.N_Pt)}")
    print(f"    |L|={len(b.L)}, |N(P_t)|={len(b.N_Pt)}")
    
    # Opponent draws
    drawn_card = D_g_truth[0] if D_g_truth else None
    
    print(f"\n  [Observation] Opponent DRAWS")
    print(f"    Drawn card: {drawn_card}")
    print(f"    This reveals: Opponent had NO legal cards on {P_t}")
    
    # Update belief
    new_L, eliminated = update_belief_after_opponent_draw(b, P_t, drawn_card)
    
    print(f"\n  [Updated Belief]")
    print(f"    Eliminated: {sorted(eliminated)}")
    print(f"    New L: {sorted(new_L)}")
    print(f"    Size: {len(b.L)} -> {len(new_L)}")
    
    # Verify correctness
    print(f"\n  [Verification]")
    
    # All H2 cards should still be in new_L (they weren't legal)
    h2_preserved = all(c in new_L for c in H_2_truth)
    print(f"    All H2 cards still in L: {h2_preserved}")
    
    # No eliminated card should be in H2
    no_h2_eliminated = all(c not in H_2_truth for c in eliminated)
    print(f"    No H2 card was eliminated: {no_h2_eliminated}")
    
    # Drawn card added to L
    if drawn_card:
        drawn_added = drawn_card in new_L
        print(f"    Drawn card added to L: {drawn_added}")
    
    # All eliminated cards were legal on P_t
    all_legal = all(is_legal(c, P_t) for c in eliminated)
    print(f"    All eliminated cards were legal: {all_legal}")
    
    return True

# --- MAIN TEST RUNNER ---
def run_random_tests(num_scenarios: int = 10):
    """
    Run multiple random scenarios to verify belief updates.
    """
    # Standard UNO micro-deck for testing
    deck_template = [
        (RED, 1), (RED, 2), (RED, 3), (RED, 4),
        (GREEN, 1), (GREEN, 2), (GREEN, 3), (GREEN, 4),
        (BLUE, 1), (BLUE, 2), (BLUE, 3), (BLUE, 4),
        (YELLOW, 1), (YELLOW, 2), (YELLOW, 3), (YELLOW, 4),
    ]
    
    random.seed(42)  # For reproducibility
    
    print("="*70)
    print("RANDOM STATE BELIEF VERIFICATION")
    print("="*70)
    print(f"Testing {num_scenarios} random scenarios...")
    
    play_success = 0
    draw_success = 0
    
    # Test opponent plays
    for i in range(1, num_scenarios // 2 + 1):
        if verify_opponent_play(i, deck_template):
            play_success += 1
    
    # Test opponent draws
    for i in range(num_scenarios // 2 + 1, num_scenarios + 1):
        if verify_opponent_draw(i, deck_template):
            draw_success += 1
    
    # Summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Opponent PLAY scenarios: {play_success} successful")
    print(f"Opponent DRAW scenarios: {draw_success} successful")
    print(f"Total successful: {play_success + draw_success}/{num_scenarios}")
    print(f"{'='*70}")

if __name__ == "__main__":
    run_random_tests(num_scenarios=10)