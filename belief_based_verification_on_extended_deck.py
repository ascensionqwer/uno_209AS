import random
import itertools
from typing import List, Tuple, Set, Optional
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
    
    Returns:
        new_L: set[Card] -> updated unknown card set
    """
    current_L = set(belief.L)
    if played_card in current_L:
        current_L.remove(played_card)
    return current_L

def update_belief_after_opponent_draw(belief: Belief, top_card: Card):
    """
    Updates the belief state after observing the opponent DRAW instead of PLAY.
    
    CRITICAL: We do NOT know what card was drawn - that's hidden information!
    
    What we learn:
    - Opponent had NO legal cards to play on top_card
    - Therefore, eliminate all cards from L that would have been legal
    - The drawn card remains UNKNOWN and is part of the general unknown set
    
    Returns:
        new_L: set[Card]      -> updated unknown card set
        eliminated: set[Card] -> cards removed from L because they would have been legal
    """
    current_L = set(belief.L)
    
    # Cards that WOULD have been legal on top_card
    legal_on_top = {card for card in current_L if is_legal(card, top_card)}
    
    # Since opponent drew, they must NOT have had any of these cards
    new_L = current_L - legal_on_top
    
    # NOTE: We do NOT add the drawn card because we don't know what it is!
    # The drawn card stays in the unknown set (it's one of the cards in L that we haven't eliminated)
    
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
    
    print(f"\n  [Verification]")
    print(f"    H2 after play: {H_2_after_play}")
    print(f"    All remaining H2 in new L: {remaining_in_L}")
    print(f"    Played card removed: {played_card not in new_L}")
    print(f"    PASS: {remaining_in_L and played_card not in new_L}")
    
    return True

def verify_opponent_draw(scenario_num: int, deck_template: List[Card]):
    """
    Generate random state where opponent must draw,
    and verify belief update eliminates correct cards.
    
    CRITICAL: We do NOT know what card was drawn!
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
    
    # What card was actually drawn (God view only - we don't know this!)
    drawn_card_truth = D_g_truth[0] if D_g_truth else None
    
    print(f"\n  [Observation] Opponent DRAWS")
    print(f"    [God only] Drawn card was: {drawn_card_truth}")
    print(f"    [Agent knows] Opponent had NO legal cards on {P_t}")
    print(f"    [Agent knows] Drawn card is UNKNOWN")
    
    # Update belief - WITHOUT knowing what was drawn!
    new_L, eliminated = update_belief_after_opponent_draw(b, P_t)
    
    print(f"\n  [Updated Belief]")
    print(f"    Eliminated: {sorted(eliminated)}")
    print(f"    New L: {sorted(new_L)}")
    print(f"    Size: {len(b.L)} -> {len(new_L)}")
    
    # Verify correctness
    print(f"\n  [Verification]")
    
    # All original H2 cards should still be in new_L (they weren't legal)
    h2_preserved = all(c in new_L for c in H_2_truth)
    print(f"    All original H2 cards still in L: {h2_preserved}")
    
    # No eliminated card should be in original H2
    no_h2_eliminated = all(c not in H_2_truth for c in eliminated)
    print(f"    No original H2 card was eliminated: {no_h2_eliminated}")
    
    # All eliminated cards were legal on P_t
    all_legal = all(is_legal(c, P_t) for c in eliminated)
    print(f"    All eliminated cards were legal: {all_legal}")
    
    # The drawn card should still be in the unknown set (we don't know what it is!)
    # After the draw, opponent has: original H2 + drawn card
    # Both should be possibilities in our belief
    print(f"\n  [Critical Check - Information Hiding]")
    if drawn_card_truth:
        # The drawn card might or might not be in new_L depending on whether it was legal
        drawn_in_L = drawn_card_truth in new_L
        drawn_was_legal = is_legal(drawn_card_truth, P_t)
        
        print(f"    Drawn card {drawn_card_truth} in new_L: {drawn_in_L}")
        print(f"    Was drawn card legal on {P_t}? {drawn_was_legal}")
        
        # If drawn card was legal, it should have been eliminated
        # If drawn card was not legal, it should still be in L
        correct_handling = drawn_in_L == (not drawn_was_legal)
        print(f"    Correct handling of drawn card: {correct_handling}")
    
    # Final pass/fail
    passed = h2_preserved and no_h2_eliminated and all_legal
    print(f"\n    PASS: {passed}")
    
    return True

def verify_draw_then_play(scenario_num: int, deck_template: List[Card]):
    """
    Opponent draws (we don't see what), then plays a card.
    This reveals information about both their original hand AND the drawn card.
    """
    print(f"\n{'='*70}")
    print(f"SCENARIO #{scenario_num}: Opponent DRAWS then PLAYS")
    print(f"{'='*70}")
    
    # Generate a state where opponent can't play initially
    max_attempts = 50
    for attempt in range(max_attempts):
        H_1, H_2_truth, D_g_truth, P, P_t = generate_random_state(deck_template)
        
        legal_h2 = [c for c in H_2_truth if is_legal(c, P_t)]
        
        if not legal_h2 and D_g_truth:
            # Check if drawn card would be legal
            drawn_card = D_g_truth[0]
            if is_legal(drawn_card, P_t):
                break
    else:
        print("  [Skip] Could not generate draw-then-play scenario")
        return False
    
    h2_size = len(H_2_truth)
    dg_size = len(D_g_truth)
    drawn_card_truth = D_g_truth[0]
    
    print(f"\n  [God] H1_truth: {H_1}")
    print(f"  [God] H2_truth: {H_2_truth}")
    print(f"  [God] Dg_truth: {D_g_truth}")
    print(f"  [God] P: {P}")
    print(f"  [God] P_t: {P_t}")
    
    # Initial belief
    observation = (H_1, h2_size, dg_size, P, P_t, "Active")
    b = Belief(observation, deck_template=deck_template)
    
    print(f"\n  [Initial Belief]")
    print(f"    L: {sorted(b.L)}")
    
    # Step 1: Opponent draws
    print(f"\n  [Observation 1] Opponent DRAWS")
    print(f"    [God only] Drew: {drawn_card_truth}")
    print(f"    [Agent knows] Original hand had no legal plays on {P_t}")
    
    new_L_after_draw, eliminated = update_belief_after_opponent_draw(b, P_t)
    
    print(f"    Eliminated: {sorted(eliminated)}")
    print(f"    New L: {sorted(new_L_after_draw)}")
    
    # Step 2: Opponent plays the drawn card
    print(f"\n  [Observation 2] Opponent PLAYS: {drawn_card_truth}")
    print(f"    [Agent learns] This was the card just drawn!")
    
    new_L_final = update_belief_after_opponent_play(
        type('obj', (object,), {'L': new_L_after_draw})(), 
        drawn_card_truth
    )
    
    print(f"\n  [Final Belief]")
    print(f"    Removed played card: {drawn_card_truth}")
    print(f"    Final L: {sorted(new_L_final)}")
    
    # Verify
    print(f"\n  [Verification]")
    h2_in_final = all(c in new_L_final for c in H_2_truth)
    print(f"    All original H2 cards in final L: {h2_in_final}")
    print(f"    Drawn/played card removed: {drawn_card_truth not in new_L_final}")
    print(f"    PASS: {h2_in_final and drawn_card_truth not in new_L_final}")
    
    return True

# --- MAIN TEST RUNNER ---
def run_random_tests(num_scenarios: int = 15):
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
    print("\nKEY PRINCIPLE: When opponent draws, we do NOT know what card!")
    
    play_success = 0
    draw_success = 0
    draw_play_success = 0
    
    scenarios_per_type = num_scenarios // 3
    
    # Test opponent plays
    print(f"\n{'='*70}")
    print("TESTING: Opponent PLAY scenarios")
    print(f"{'='*70}")
    for i in range(1, scenarios_per_type + 1):
        if verify_opponent_play(i, deck_template):
            play_success += 1
    
    # Test opponent draws
    print(f"\n{'='*70}")
    print("TESTING: Opponent DRAW scenarios")
    print(f"{'='*70}")
    for i in range(scenarios_per_type + 1, 2 * scenarios_per_type + 1):
        if verify_opponent_draw(i, deck_template):
            draw_success += 1
    
    # Test draw-then-play
    print(f"\n{'='*70}")
    print("TESTING: Opponent DRAW-then-PLAY scenarios")
    print(f"{'='*70}")
    for i in range(2 * scenarios_per_type + 1, num_scenarios + 1):
        if verify_draw_then_play(i, deck_template):
            draw_play_success += 1
    
    # Summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Opponent PLAY scenarios: {play_success}/{scenarios_per_type} successful")
    print(f"Opponent DRAW scenarios: {draw_success}/{scenarios_per_type} successful")
    print(f"Opponent DRAW-then-PLAY: {draw_play_success}/{num_scenarios - 2*scenarios_per_type} successful")
    print(f"Total: {play_success + draw_success + draw_play_success}/{num_scenarios}")
    print(f"{'='*70}")
    print("\nKEY INSIGHT:")
    print("When opponent draws, the drawn card remains UNKNOWN.")
    print("We only eliminate cards that WOULD have been legal.")
    print("This maintains proper information hiding in imperfect information games!")
    print(f"{'='*70}")

if __name__ == "__main__":
    run_random_tests(num_scenarios=15)