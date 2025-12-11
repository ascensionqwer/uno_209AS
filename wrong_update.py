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

# --- BELIEF UPDATE LOGIC ---
def update_belief_after_opponent_draw(belief: Belief, top_card: Card, drawn_card: Card = None):
    """
    Updates the belief state after observing the opponent DRAW instead of PLAY.

    Key insight: If opponent drew, it means they had NO legal cards to play.
    Therefore, we eliminate all cards from L that would have been legal on top_card.

    Returns:
        new_L: set[Card]      -> updated unknown card set
        eliminated: set[Card] -> cards removed from L because they would have been legal
    """
    # Make sure we're working with a set
    current_L = set(belief.L)

    # Cards that WOULD have been legal on top_card
    legal_on_top = {card for card in current_L if is_legal(card, top_card)}

    # Since opponent drew, they must NOT have had any of these cards
    new_L = current_L - legal_on_top

    # If we know what card was drawn, add it to the unknown set
    if drawn_card is not None:
        new_L.add(drawn_card)

    return new_L, legal_on_top

# --- SCENARIO 1: Simple Wrong Assumption ---
def scenario_wrong_assumption():
    """
    Scenario where we incorrectly assume opponent has G1, but they draw on a green top card.
    """
    print("="*70)
    print("SCENARIO #1: Wrong Assumption - Opponent Draws on Green")
    print("="*70)
    
    # Micro-deck for simplicity
    micro_deck = [
        (RED, 1), (RED, 2), (RED, 3),
        (GREEN, 1), (GREEN, 2), (GREEN, 3),
        (BLUE, 1), (BLUE, 2),
    ]
    
    # Initial State
    H_1 = [(RED, 1), (GREEN, 2)]  # Our hand
    H_2_truth = [(BLUE, 1)]  # Opponent actually holds B1 (not G1!)
    P = [(RED, 2), (GREEN, 1)]  # Played cards
    P_t = (GREEN, 1)  # Current top card (GREEN)
    
    h2_size = 1
    dg_size = 2  # Remaining deck has 2 cards
    
    print("\n[INITIAL STATE]")
    print(f"  [God] H1_truth: {H_1}")
    print(f"  [God] H2_truth: {H_2_truth}")
    print(f"  [God] P: {P}")
    print(f"  [God] P_t: {P_t} (GREEN)")
    print(f"  [God] Remaining deck: {[c for c in micro_deck if c not in H_1 + H_2_truth + P]}")
    
    # Agent's observation
    observation = (H_1, h2_size, dg_size, P, P_t, "Active")
    b = Belief(observation, deck_template=micro_deck)
    
    print(f"\n[INITIAL BELIEF]")
    print(f"  Unknown cards L: {sorted(b.L)}")
    print(f"  N(P_t) - Legal on GREEN: {sorted(b.N_Pt)}")
    print(f"  |L|={len(b.L)}, |N(P_t)|={len(b.N_Pt)}")
    
    # Calculate initial probabilities
    if len(b.L) > 0:
        prob_can_play = len(b.N_Pt) / len(b.L)
        print(f"  P(opponent can play on GREEN) ≈ {prob_can_play:.2%}")
    
    # Show what we initially think are possible
    print(f"\n[INITIAL ASSUMPTION]")
    print(f"  We think opponent MIGHT have: {sorted(b.L)}")
    print(f"  If top is GREEN, opponent COULD play: {sorted(b.N_Pt)}")
    
    # CRITICAL OBSERVATION: Opponent DRAWS instead of playing!
    print(f"\n[OBSERVATION: Opponent DRAWS on GREEN top card]")
    print(f"  This reveals: Opponent did NOT have any GREEN or any '1' cards")
    
    # Update belief
    new_L, eliminated = update_belief_after_opponent_draw(b, P_t, drawn_card=(GREEN, 3))
    
    print(f"\n[UPDATED BELIEF]")
    print(f"  Eliminated from L: {sorted(eliminated)}")
    print(f"  Updated L: {sorted(new_L)}")
    print(f"  Size reduced: {len(b.L)} -> {len(new_L)}")
    
    # Show the refinement
    print(f"\n[BELIEF REFINEMENT]")
    print(f"  BEFORE: Opponent could have {len(b.L)} possible cards")
    print(f"  AFTER: Opponent could have {len(new_L)} possible cards")
    print(f"  TRUTH: Opponent has {H_2_truth[0]}")
    print(f"  Is truth in updated belief? {H_2_truth[0] in new_L}")

# --- SCENARIO 2: Multiple Draws Narrow Down Belief ---
def scenario_multiple_draws():
    """
    Scenario showing how multiple opponent draws progressively narrow the belief space.
    """
    print("\n" + "="*70)
    print("SCENARIO #2: Multiple Draws Narrow Belief Space")
    print("="*70)
    
    # Larger deck
    deck = [
        (RED, 1), (RED, 2), (RED, 3), (RED, 4),
        (GREEN, 1), (GREEN, 2), (GREEN, 3), (GREEN, 4),
        (BLUE, 1), (BLUE, 2), (BLUE, 3), (BLUE, 4),
    ]
    
    # Initial State
    H_1 = [(RED, 1), (GREEN, 1)]  # Our hand
    H_2_truth = [(BLUE, 4)]  # Opponent has B4 (no match to most cards)
    P = [(RED, 2)]  # Only one card played
    P_t = (RED, 2)  # Current top (RED)
    
    h2_size = 1
    dg_size = 9
    
    print("\n[INITIAL STATE]")
    print(f"  Our hand: {H_1}")
    print(f"  Opponent hand (truth): {H_2_truth}")
    print(f"  Top card: {P_t}")
    
    observation = (H_1, h2_size, dg_size, P, P_t, "Active")
    b = Belief(observation, deck_template=deck)
    
    print(f"\n[BELIEF EVOLUTION]")
    print(f"  Initial |L| = {len(b.L)}")
    
    # Simulate sequence of opponent draws on different top cards
    draws_sequence = [
        ((RED, 2), "Opponent draws on RED 2"),
        ((GREEN, 3), "We play G1, opp draws on GREEN 3"),
        ((BLUE, 1), "We play ?, opp draws on BLUE 1"),
    ]
    
    current_L = set(b.L)   # convert belief list → set
    
    for i, (top, description) in enumerate(draws_sequence, 1):
        print(f"\n  Draw #{i}: {description}")
        print(f"    Top card: {top}")
        
        # Calculate what would be legal
        legal_on_this_top = {card for card in current_L if is_legal(card, top)}
        print(f"    Cards that WOULD be legal: {sorted(legal_on_this_top)}")
        
        # Update: remove legal cards since opponent didn't play them
        current_L = current_L - legal_on_this_top
        print(f"    Updated |L| = {len(current_L)}")
        print(f"    Remaining possibilities: {sorted(current_L)}")
        
        # Check if truth is still in belief
        still_possible = H_2_truth[0] in current_L
        print(f"    Is B4 (truth) still possible? {still_possible}")
    
    print(f"\n[FINAL BELIEF]")
    print(f"  Started with {len(b.L)} possible cards")
    print(f"  After 3 draws, narrowed to {len(current_L)} possible cards")
    print(f"  Final candidates: {sorted(current_L)}")
    print(f"  Ground truth {H_2_truth[0]} is in final belief: {H_2_truth[0] in current_L}")

# --- SCENARIO 3: Learning from Draw Then Play ---
def scenario_draw_then_play():
    """
    Opponent draws a card, then plays it immediately. What can we learn?
    """
    print("\n" + "="*70)
    print("SCENARIO #3: Opponent Draws, Then Plays Drawn Card")
    print("="*70)
    
    deck = [
        (RED, 1), (RED, 2), (RED, 3),
        (GREEN, 1), (GREEN, 2), (GREEN, 3),
        (BLUE, 1), (BLUE, 2), (BLUE, 3),
    ]
    
    H_1 = [(RED, 1), (BLUE, 1)]
    H_2_initial = [(GREEN, 1)]  # Opponent starts with G1
    P = [(RED, 2)]
    P_t = (RED, 2)  # RED top
    
    h2_size = 1
    dg_size = 5
    
    print("\n[SETUP]")
    print(f"  Our hand: {H_1}")
    print(f"  Opponent starts with: {H_2_initial}")
    print(f"  Top card: {P_t} (RED)")
    
    observation = (H_1, h2_size, dg_size, P, P_t, "Active")
    b = Belief(observation, deck_template=deck)
    
    print(f"\n[INITIAL BELIEF]")
    print(f"  Unknown cards L: {sorted(b.L)}")
    print(f"  Legal on RED: {sorted([c for c in b.L if is_legal(c, P_t)])}")
    
    # Opponent draws
    drawn_card = (RED, 3)  # They draw R3
    print(f"\n[OBSERVATION: Opponent DRAWS]")
    print(f"  Drawn card: {drawn_card}")
    print(f"  This means: Original hand (G1) was NOT legal on RED")
    
    new_L, eliminated = update_belief_after_opponent_draw(b, P_t, drawn_card)
    
    print(f"  Eliminated: {sorted(eliminated)}")
    print(f"  Updated L (includes drawn card): {sorted(new_L)}")
    
    # Opponent then plays the drawn card
    print(f"\n[OBSERVATION: Opponent PLAYS {drawn_card}]")
    print(f"  This confirms: Drawn card WAS legal on {P_t}")
    print(f"  New info: Opponent's hand now has 1 card (the original G1)")
    
    # Update further: remove played card from L
    final_L = new_L - {drawn_card}
    print(f"  Final L: {sorted(final_L)}")
    print(f"  We now know opponent has one of: {sorted(final_L)}")
    print(f"  Truth (G1) in belief: {(GREEN, 1) in final_L}")

# --- MAIN ---
if __name__ == "__main__":
    scenario_wrong_assumption()
    scenario_multiple_draws()
    scenario_draw_then_play()
    
    print("\n" + "="*70)
    print("KEY TAKEAWAYS:")
    print("="*70)
    print("1. When opponent DRAWS, eliminate all cards from L that were legal")
    print("2. Multiple draws progressively narrow the belief space")
    print("3. Each draw reveals: opponent did NOT have {color} or {value}")
    print("4. Drawn cards enter the unknown set (opponent's new hand)")
    print("5. Belief refinement is cumulative and monotonic")
    print("="*70)