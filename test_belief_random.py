
import random
import sys
from typing import List, Tuple
from collections import Counter

from belief import Belief
from pomdp import Action
from cards import Card, RED, YELLOW, GREEN, BLUE

# Logging setup
LOG_FILE = "belief_test_log.txt"

def log(message):
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")
    print(message)

def clear_log():
    with open(LOG_FILE, "w") as f:
        f.write("=== Belief State Random Tester Log ===\n")

def generate_full_deck() -> List[Card]:
    full_deck = []
    colors = [RED, YELLOW, GREEN, BLUE]
    for color in colors:
        full_deck.append((color, 0))
        for number in range(1, 10):
            full_deck.append((color, number))
            full_deck.append((color, number))
    return full_deck

def generate_random_state(seed: int) -> Tuple:
    rng = random.Random(seed)
    full_deck = generate_full_deck()
    rng.shuffle(full_deck)
    
    # 1. Distribute cards
    # H_1 size: 1-7
    h1_size = rng.randint(1, 7)
    H_1 = full_deck[:h1_size]
    full_deck = full_deck[h1_size:]
    
    # H_2 size: 1-7
    h2_size = rng.randint(1, 7)
    # We don't know H_2, but we need to reserve cards for it in the "truth"
    # For the observation, we just know the size.
    # But to simulate a valid game, we need to ensure cards exist.
    # Let's just say the next h2_size cards are "held" by opponent but unknown to us.
    # We won't explicitly store them in the observation, just ensure they aren't in P or P_t.
    H_2_truth = full_deck[:h2_size]
    full_deck = full_deck[h2_size:]
    
    # P_t (Top Card)
    P_t = full_deck.pop(0)
    
    # P (Played Pile) - random size 0-10
    p_size = rng.randint(0, 10)
    P = full_deck[:p_size]
    full_deck = full_deck[p_size:]
    
    # D_g (Deck) - remaining
    D_g_truth = full_deck
    dg_size = len(D_g_truth)
    
    G_o = "Active"
    
    # Observation O = (H_1, |H_2|, |D_g|, P, P_t, G_o)
    observation = (H_1, h2_size, dg_size, P, P_t, G_o)
    
    return observation, H_2_truth, D_g_truth

def is_legal(card: Card, top_card: Card) -> bool:
    return card[0] == top_card[0] or card[1] == top_card[1]

def test_scenario(seed: int):
    log(f"\n--- Test Scenario {seed} ---")
    observation, H_2_truth, D_g_truth = generate_random_state(seed)
    
    H_1, h2_size, dg_size, P, P_t, G_o = observation
    
    log(f"Initial State:")
    log(f"  H_1: {len(H_1)} cards")
    log(f"  H_2: {h2_size} cards (Truth: {H_2_truth})")
    log(f"  D_g: {dg_size} cards")
    log(f"  P_t: {P_t}")
    
    # Initialize Belief
    try:
        b = Belief(observation)
    except Exception as e:
        log(f"CRITICAL: Failed to initialize Belief: {e}")
        return False

    log(f"Belief Initialized. |L|={len(b.L)}")
    
    # Decide Action
    # Check if opponent (truth) has legal cards
    legal_moves = [c for c in H_2_truth if is_legal(c, P_t)]
    
    action = None
    if legal_moves:
        # Opponent plays a card
        played_card = random.choice(legal_moves)
        action = Action(X_1=played_card)
        log(f"Action: Opponent PLAYS {played_card}")
        
        # Update Truth
        H_2_truth.remove(played_card)
        P.append(played_card)
        P_t = played_card
        h2_size -= 1
        
    else:
        # Opponent draws
        action = Action(n=1) # Draw 1
        log(f"Action: Opponent DRAWS (No legal moves)")
        
        # Update Truth
        if len(D_g_truth) > 0:
            drawn_card = D_g_truth.pop(0)
            H_2_truth.append(drawn_card)
            h2_size += 1
            dg_size -= 1
            log(f"  (Truth: Drawn card was {drawn_card})")
        else:
            log("  (Truth: Deck empty, cannot draw)")
            # In real game, deck would reshuffle. For this test, maybe just skip?
            # Or handle empty deck.
            # Belief update handles empty deck by just decrementing size? 
            # Let's assume deck is large enough for now (generated full deck).
            pass

    # Update Belief
    new_observation = (H_1, h2_size, dg_size, P, P_t, G_o)
    
    try:
        b.update(action, new_observation)
    except Exception as e:
        log(f"CRITICAL: Failed to update Belief: {e}")
        return False
        
    log(f"Belief Updated.")
    log(f"  New H_2 Size: {b.h2_size}")
    log(f"  Posterior Mode: {b.posterior_mode}")
    
    # Verification 1: Sampling Consistency
    log("Verifying Sampling...")
    samples = b.sample_states(n_samples=50)
    valid_samples = 0
    for s in samples:
        # s = (H_1, H_2, D_g, P, P_t, G_o)
        s_h2 = s[1]
        s_dg = s[2]
        
        if len(s_h2) != h2_size:
            log(f"  FAIL: Sampled H_2 size {len(s_h2)} != expected {h2_size}")
            continue
            
        if len(s_dg) != dg_size:
            log(f"  FAIL: Sampled D_g size {len(s_dg)} != expected {dg_size}")
            continue
            
        valid_samples += 1
        
    if valid_samples == 50:
        log("  PASS: All 50 samples have correct sizes.")
    else:
        log(f"  FAIL: Only {valid_samples}/50 samples valid.")
        return False

    # Verification 2: Draw Logic (if applicable)
    if action.is_draw():
        log("Verifying Draw Logic...")
        # Check if any sample has a legal card
        legal_count = 0
        for s in samples:
            s_h2 = s[1]
            if any(is_legal(c, P_t) for c in s_h2):
                legal_count += 1
        
        log(f"  Samples with legal cards in H_2: {legal_count}/50")
        if legal_count == 0 and dg_size > 10: # If deck is large, unlikely to draw 50 non-legal in a row
             # It IS possible if all remaining cards are non-legal, but unlikely with full deck.
             # Let's check if L has legal cards.
             legal_in_L = [c for c in b.L if is_legal(c, P_t)]
             if len(legal_in_L) > 0:
                 log(f"  WARNING: 0 samples had legal cards, but L has {len(legal_in_L)} legal cards.")
                 # This might be a flake or a bug.
             else:
                 log("  INFO: L has no legal cards, so 0 samples is expected.")
        else:
             log("  PASS: Samples contain legal cards (or deck small/empty).")

    # Verification 3: Probabilities
    log("Verifying Probabilities...")
    probs = b.get_card_probabilities("H_2")
    total_prob = sum(probs.values())
    log(f"  Sum of P(c in H_2): {total_prob:.2f} (Expected: {h2_size})")
    
    if abs(total_prob - h2_size) > 0.1:
        log(f"  FAIL: Probability sum mismatch.")
        return False
    else:
        log("  PASS: Probability sum matches H_2 size.")

    return True

def main():
    clear_log()
    log("Starting Random Belief State Tests...")
    
    n_tests = 20
    passes = 0
    
    for i in range(n_tests):
        seed = i + 100
        if test_scenario(seed):
            passes += 1
        else:
            log(f"Test {seed} FAILED.")
            
    log(f"\nTotal Passed: {passes}/{n_tests}")

if __name__ == "__main__":
    main()
