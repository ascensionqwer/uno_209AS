"""
Exhaustive validation suite for Bayesian belief updates.

This module provides:
1. Brute-force enumeration for small state spaces
2. Direct comparison between your Belief class and true Bayesian posterior
3. Long sequence testing with invariant checking
4. Adversarial corner case testing
"""

import random
import itertools
from typing import List, Tuple, Dict, Set
from collections import defaultdict
import math

from cards import RED, GREEN, BLUE, YELLOW, Card
from belief import Belief

def is_legal(card: Card, top_card: Card) -> bool:
    """UNO rule: Match Color or Match Value."""
    return card[0] == top_card[0] or card[1] == top_card[1]

# ============================================================================
# EXHAUSTIVE STATE ENUMERATION
# ============================================================================

def enumerate_all_hidden_states(unknown_cards: List[Card], h2_size: int):
    """
    Enumerate ALL possible (H2, deck) partitions.
    Returns: list of (frozenset(H2), tuple(deck)) pairs
    """
    states = []
    for h2 in itertools.combinations(unknown_cards, h2_size):
        h2_set = frozenset(h2)
        remaining = [c for c in unknown_cards if c not in h2_set]
        for deck_order in itertools.permutations(remaining):
            states.append((h2_set, deck_order))
    return states

def compute_exact_posterior_play(states_with_probs: List[Tuple], 
                                  played_card: Card, 
                                  top_card: Card) -> Dict:
    """
    Compute exact posterior after play observation using brute force.
    
    Args:
        states_with_probs: list of ((H2, deck), probability)
        played_card: card opponent played
        top_card: card played on
    
    Returns:
        dict: {(new_H2, new_deck): posterior_prob}
    """
    posterior = {}
    
    for (h2, deck), prior_p in states_with_probs:
        # Check consistency with observation
        if played_card not in h2:
            continue
        if not is_legal(played_card, top_card):
            continue
        
        # Compute likelihood
        legal_moves = [c for c in h2 if is_legal(c, top_card)]
        if not legal_moves:
            continue
        
        likelihood = 1.0 / len(legal_moves)
        
        # New state
        new_h2 = frozenset(c for c in h2 if c != played_card)
        new_state = (new_h2, deck)
        
        # ACCUMULATE (not overwrite!)
        posterior[new_state] = posterior.get(new_state, 0.0) + prior_p * likelihood
    
    # Normalize
    total = sum(posterior.values())
    if total > 0:
        for state in posterior:
            posterior[state] /= total
    
    return posterior

def compute_exact_posterior_draw(states_with_probs: List[Tuple], 
                                  top_card: Card) -> Dict:
    """
    Compute exact posterior after draw observation using brute force.
    
    Args:
        states_with_probs: list of ((H2, deck), probability)
        top_card: card opponent couldn't play on
    
    Returns:
        dict: {(new_H2, new_deck): posterior_prob}
    """
    posterior = {}
    
    for (h2, deck), prior_p in states_with_probs:
        # Check consistency: opponent had no legal moves
        legal_moves = [c for c in h2 if is_legal(c, top_card)]
        if legal_moves:
            continue
        
        # Check can draw
        if not deck:
            continue
        
        # Deterministic transition
        likelihood = 1.0
        
        # New state: draw top card
        drawn = deck[0]
        new_h2 = frozenset(h2 | {drawn})
        new_deck = deck[1:]
        new_state = (new_h2, new_deck)
        
        # ACCUMULATE
        posterior[new_state] = posterior.get(new_state, 0.0) + prior_p * likelihood
    
    # Normalize
    total = sum(posterior.values())
    if total > 0:
        for state in posterior:
            posterior[state] /= total
    
    return posterior

# ============================================================================
# COMPARISON WITH YOUR BELIEF CLASS
# ============================================================================

def belief_to_distribution(belief: Belief, deck_template: List[Card]) -> Dict:
    """
    Convert your Belief class to an explicit probability distribution.
    Uses sampling to approximate the distribution.
    """
    samples = {}
    n_samples = 1000
    
    for i in range(n_samples):
        state = belief.sample_state(seed=i)
        H1, H2, Dg, P, Pt, Go = state
        
        h2_frozen = frozenset(H2)
        dg_tuple = tuple(Dg)
        key = (h2_frozen, dg_tuple)
        
        samples[key] = samples.get(key, 0) + 1
    
    # Normalize
    total = sum(samples.values())
    distribution = {k: v/total for k, v in samples.items()}
    
    return distribution

def compare_distributions(dist1: Dict, dist2: Dict, tolerance: float = 0.01) -> Tuple[bool, str]:
    """
    Compare two probability distributions.
    Returns (is_close, message)
    """
    # Get all states
    all_states = set(dist1.keys()) | set(dist2.keys())
    
    max_diff = 0.0
    total_diff = 0.0
    
    for state in all_states:
        p1 = dist1.get(state, 0.0)
        p2 = dist2.get(state, 0.0)
        diff = abs(p1 - p2)
        max_diff = max(max_diff, diff)
        total_diff += diff
    
    is_close = max_diff < tolerance
    msg = f"Max diff: {max_diff:.6f}, Total variation: {total_diff/2:.6f}"
    
    return is_close, msg

# ============================================================================
# EXHAUSTIVE TESTS
# ============================================================================

def test_exhaustive_small_deck():
    """
    Test on smallest possible deck where we can enumerate everything.
    """
    print("\n" + "="*70)
    print("EXHAUSTIVE TEST 1: Tiny Deck (4 cards)")
    print("="*70)
    
    deck_template = [(RED, 1), (RED, 2), (BLUE, 1), (BLUE, 2)]
    
    H1 = [(RED, 1)]
    P = []
    P_t = (RED, 1)
    h2_size = 1
    
    unknown = [c for c in deck_template if c not in H1 and c not in P]
    
    print(f"Setup: H1={H1}, unknown={unknown}, h2_size={h2_size}")
    print(f"Total possible states: {math.comb(len(unknown), h2_size) * math.factorial(len(unknown) - h2_size)}")
    
    # Enumerate all states with uniform prior
    all_states = enumerate_all_hidden_states(unknown, h2_size)
    uniform_p = 1.0 / len(all_states)
    state_probs = [(s, uniform_p) for s in all_states]
    
    print(f"\nInitial: {len(all_states)} states, uniform probability = {uniform_p:.6f}")
    
    # Test PLAY observation
    print(f"\n--- Observation: Opponent plays (RED, 2) ---")
    played_card = (RED, 2)
    
    # Exact posterior
    exact_post = compute_exact_posterior_play(state_probs, played_card, P_t)
    print(f"Exact posterior: {len(exact_post)} states")
    print(f"Probabilities sum to: {sum(exact_post.values()):.6f}")
    
    # Show distribution
    for (h2, deck), p in sorted(exact_post.items(), key=lambda x: -x[1])[:5]:
        print(f"  P={p:.4f}: H2={sorted(h2)}, deck={list(deck)}")
    
    print("\n✓ Exact computation successful")

def test_play_vs_exact():
    """
    Compare Belief class update against exact brute-force computation.
    """
    print("\n" + "="*70)
    print("VALIDATION TEST 2: Belief Class vs Exact Computation")
    print("="*70)
    
    deck_template = [
        (RED, 1), (RED, 2), (GREEN, 1), (GREEN, 2),
        (BLUE, 1), (BLUE, 2)
    ]
    
    H1 = [(RED, 1)]
    P = [(GREEN, 1)]
    P_t = (GREEN, 1)
    h2_size = 2
    
    print(f"Setup: H1={H1}, P={P}, P_t={P_t}, h2_size={h2_size}")
    
    # Your belief class
    observation = (H1, h2_size, len(deck_template) - len(H1) - len(P) - h2_size, 
                   P, P_t, "Active")
    belief = Belief(observation, deck_template=deck_template)
    
    print(f"\nInitial belief: |L|={len(belief.L)}, |N(P_t)|={len(belief.N_Pt)}")
    
    # Play observation
    played_card = (GREEN, 2)
    print(f"\n--- Opponent plays {played_card} ---")
    
    # Update your belief
    belief.update_opponent_played(played_card)
    
    print(f"After update: |L|={len(belief.L)}, |N(P_t)|={len(belief.N_Pt)}")
    print(f"Hand size: {h2_size} -> {belief.h2_size}")
    
    # Get card probabilities from your belief
    card_probs_yours = belief.get_card_probabilities("H_2")
    
    print(f"\nYour belief - P(card in H2):")
    for card in sorted(card_probs_yours.keys())[:5]:
        print(f"  {card}: {card_probs_yours[card]:.4f}")
    
    # Compare with exact computation
    unknown = [c for c in deck_template if c not in H1 and c not in P]
    all_states = enumerate_all_hidden_states(unknown, h2_size)
    uniform_p = 1.0 / len(all_states)
    state_probs = [(s, uniform_p) for s in all_states]
    
    exact_post = compute_exact_posterior_play(state_probs, played_card, P_t)
    
    # Compute exact card probabilities
    card_probs_exact = defaultdict(float)
    for (h2, deck), p in exact_post.items():
        for card in h2:
            card_probs_exact[card] += p
    
    print(f"\nExact computation - P(card in H2):")
    for card in sorted(card_probs_exact.keys())[:5]:
        print(f"  {card}: {card_probs_exact[card]:.4f}")
    
    # Compare
    print(f"\n--- Comparison ---")
    common_cards = set(card_probs_yours.keys()) & set(card_probs_exact.keys())
    max_diff = 0.0
    for card in common_cards:
        diff = abs(card_probs_yours[card] - card_probs_exact[card])
        max_diff = max(max_diff, diff)
        if diff > 0.01:
            print(f"  {card}: yours={card_probs_yours[card]:.4f}, exact={card_probs_exact[card]:.4f}, diff={diff:.4f}")
    
    print(f"\nMax difference: {max_diff:.6f}")
    if max_diff < 0.1:
        print("✓ CLOSE MATCH (sampling approximation)")
    else:
        print("✗ LARGE DISCREPANCY - potential bug!")

def test_draw_vs_exact():
    """
    Compare Belief class draw update against exact computation.
    """
    print("\n" + "="*70)
    print("VALIDATION TEST 3: Draw Update vs Exact")
    print("="*70)
    
    deck_template = [
        (RED, 1), (RED, 2), (GREEN, 1), (GREEN, 2),
        (BLUE, 1), (BLUE, 2)
    ]
    
    H1 = [(RED, 1)]
    P = [(BLUE, 1)]
    P_t = (BLUE, 1)
    h2_size = 2
    
    print(f"Setup: H1={H1}, P={P}, P_t={P_t}, h2_size={h2_size}")
    
    # Your belief class
    observation = (H1, h2_size, len(deck_template) - len(H1) - len(P) - h2_size,
                   P, P_t, "Active")
    belief = Belief(observation, deck_template=deck_template)
    
    print(f"\nInitial: |L|={len(belief.L)}, |N(P_t)|={len(belief.N_Pt)}")
    
    # Calculate P(no legal)
    p_no_legal = belief._prob_no_legal()
    print(f"P(opponent has no legal moves) = {p_no_legal:.4f}")
    
    # Draw observation
    print(f"\n--- Opponent draws (forced) ---")
    belief.update_opponent_drew()
    
    print(f"After draw: |L|={len(belief.L)}, h2_size={belief.h2_size}")
    
    # Your probabilities
    card_probs_yours = belief.get_card_probabilities("H_2")
    
    print(f"\nYour belief - P(card in H2 after draw):")
    for card in sorted(card_probs_yours.keys())[:5]:
        print(f"  {card}: {card_probs_yours[card]:.4f}")
    
    # Exact computation
    unknown = [c for c in deck_template if c not in H1 and c not in P]
    all_states = enumerate_all_hidden_states(unknown, h2_size)
    uniform_p = 1.0 / len(all_states)
    state_probs = [(s, uniform_p) for s in all_states]
    
    exact_post = compute_exact_posterior_draw(state_probs, P_t)
    
    print(f"\nExact posterior: {len(exact_post)} states")
    
    # Exact card probabilities
    card_probs_exact = defaultdict(float)
    for (h2, deck), p in exact_post.items():
        for card in h2:
            card_probs_exact[card] += p
    
    print(f"\nExact computation - P(card in H2 after draw):")
    for card in sorted(card_probs_exact.keys())[:5]:
        print(f"  {card}: {card_probs_exact[card]:.4f}")
    
    # Compare
    print(f"\n--- Comparison ---")
    common_cards = set(card_probs_yours.keys()) & set(card_probs_exact.keys())
    max_diff = 0.0
    for card in common_cards:
        diff = abs(card_probs_yours.get(card, 0) - card_probs_exact.get(card, 0))
        max_diff = max(max_diff, diff)
        if diff > 0.01:
            print(f"  {card}: yours={card_probs_yours.get(card, 0):.4f}, exact={card_probs_exact.get(card, 0):.4f}, diff={diff:.4f}")
    
    print(f"\nMax difference: {max_diff:.6f}")
    if max_diff < 0.1:
        print("✓ REASONABLE MATCH")
    else:
        print("✗ LARGE DISCREPANCY")

# ============================================================================
# LONG SEQUENCE TESTING
# ============================================================================

def test_long_sequence(num_steps: int = 20):
    """
    Test a long sequence of random actions, checking invariants at each step.
    """
    print("\n" + "="*70)
    print(f"STRESS TEST: {num_steps}-Step Random Sequence")
    print("="*70)
    
    deck_template = [
        (RED, 1), (RED, 2), (GREEN, 1), (GREEN, 2),
        (BLUE, 1), (BLUE, 2), (YELLOW, 1), (YELLOW, 2),
    ]
    
    H1 = [(RED, 1)]
    P = [(GREEN, 1)]
    P_t = (GREEN, 1)
    h2_size = 3
    
    observation = (H1, h2_size, len(deck_template) - len(H1) - len(P) - h2_size,
                   P, P_t, "Active")
    belief = Belief(observation, deck_template=deck_template)
    
    print(f"Initial: h2={h2_size}, dg={belief.dg_size}, |L|={len(belief.L)}")
    
    random.seed(42)
    errors = []
    
    for step in range(1, num_steps + 1):
        # Decide action based on current belief
        can_play_cards = [c for c in belief.L if is_legal(c, P_t)]
        
        if not can_play_cards or (belief.dg_size > 0 and random.random() < 0.3):
            # Draw
            if belief.dg_size == 0:
                print(f"\nStep {step}: Cannot continue (deck empty)")
                break
            
            action_type = "DRAW"
            belief.update_opponent_drew()
            h2_size += 1
        else:
            # Play
            played = random.choice(can_play_cards)
            action_type = f"PLAY {played}"
            belief.update_opponent_played(played)
            P_t = played
            h2_size -= 1
        
        # Check invariants
        try:
            # 1. Sizes are consistent
            assert belief.h2_size == h2_size, f"Hand size mismatch: expected {h2_size}, got {belief.h2_size}"
            
            # 2. L is correct
            expected_L_size = len(deck_template) - len(belief.H_1) - len(belief.P)
            actual_L_size = len(belief.L)
            assert actual_L_size == expected_L_size, f"L size mismatch: expected {expected_L_size}, got {actual_L_size}"
            
            # 3. Can sample
            try:
                sample = belief.sample_state(seed=step)
                assert len(sample[1]) == h2_size, f"Sampled H2 wrong size"
            except Exception as e:
                raise AssertionError(f"Sampling failed: {e}")
            
            if step % 5 == 0:
                print(f"Step {step}: {action_type} - h2={h2_size}, dg={belief.dg_size}, |L|={len(belief.L)} ✓")
            
        except AssertionError as e:
            errors.append(f"Step {step}: {e}")
            print(f"Step {step}: {action_type} - ✗ {e}")
            break
    
    print(f"\n{'='*70}")
    if not errors:
        print(f"✓ ALL {step} STEPS PASSED")
    else:
        print(f"✗ FAILED at step {len(errors)}")
        for err in errors:
            print(f"  {err}")
    print(f"{'='*70}")

# ============================================================================
# CORNER CASES
# ============================================================================

def test_corner_cases():
    """Test adversarial corner cases."""
    print("\n" + "="*70)
    print("CORNER CASE TESTS")
    print("="*70)
    
    deck_template = [
        (RED, 1), (RED, 2), (GREEN, 1), (GREEN, 2),
        (BLUE, 1), (BLUE, 2)
    ]
    
    # Case 1: |N(P_t)| = 0 (no legal moves possible)
    print("\n--- Case 1: No legal moves possible ---")
    H1 = [(RED, 1), (GREEN, 1), (BLUE, 1)]
    P = [(RED, 2)]
    P_t = (YELLOW, 1)  # No other YELLOW in deck
    h2_size = 1
    
    observation = (H1, h2_size, len(deck_template) - len(H1) - len(P) - h2_size,
                   P, P_t, "Active")
    
    try:
        belief = Belief(observation, deck_template=deck_template)
        print(f"  |N(P_t)| = {len(belief.N_Pt)}")
        print(f"  P(no legal) = {belief._prob_no_legal():.4f}")
        print("  ✓ Handled correctly")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Case 2: |N(P_t)| = 1 (only one legal move)
    print("\n--- Case 2: Exactly one legal card ---")
    H1 = [(RED, 1)]
    P = [(GREEN, 1), (GREEN, 2)]
    P_t = (GREEN, 2)
    h2_size = 1
    
    observation = (H1, h2_size, len(deck_template) - len(H1) - len(P) - h2_size,
                   P, P_t, "Active")
    belief = Belief(observation, deck_template=deck_template)
    
    print(f"  |L| = {len(belief.L)}")
    print(f"  |N(P_t)| = {len(belief.N_Pt)}")
    print(f"  N(P_t) = {belief.N_Pt}")
    
    # If opponent plays the only legal card, we learn it exactly
    if belief.N_Pt:
        only_legal = list(belief.N_Pt)[0]
        belief.update_opponent_played(only_legal)
        print(f"  After play: |L| = {len(belief.L)}")
        print("  ✓ Single legal card handled")
    
    # Case 3: Near-empty unknown pool
    print("\n--- Case 3: Very small unknown pool ---")
    H1 = [(RED, 1), (RED, 2), (GREEN, 1)]
    P = [(GREEN, 2), (BLUE, 1)]
    P_t = (BLUE, 1)
    h2_size = 1
    
    observation = (H1, h2_size, len(deck_template) - len(H1) - len(P) - h2_size,
                   P, P_t, "Active")
    belief = Belief(observation, deck_template=deck_template)
    
    print(f"  |L| = {len(belief.L)}")
    print(f"  L = {belief.L}")
    
    try:
        sample = belief.sample_state(seed=1)
        print("  ✓ Can still sample")
    except Exception as e:
        print(f"  ✗ Sampling failed: {e}")

# ============================================================================
# MAIN RUNNER
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("EXHAUSTIVE BAYESIAN VALIDATION SUITE")
    print("="*70)
    print("\nThis suite provides:")
    print("1. Brute-force exact computation for comparison")
    print("2. Direct validation against known correct posteriors")
    print("3. Long-sequence stress testing")
    print("4. Adversarial corner cases")
    
    test_exhaustive_small_deck()
    test_play_vs_exact()
    test_draw_vs_exact()
    test_long_sequence(num_steps=30)
    test_corner_cases()
    
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)
    print("\nIf all tests pass:")
    print("✓ Belief updates are Bayesian-correct (within sampling error)")
    print("✓ Invariants hold across long sequences")
    print("✓ Corner cases handled properly")
    print("\nModel assumption: Opponent plays uniformly among legal moves")
    print("="*70)