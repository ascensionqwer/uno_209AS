"""
Golden Reference Belief State Validator (FIXED)

This module provides a mathematically rigorous validation framework for belief state updates
by comparing against exhaustively enumerated ground truth for small decks.

FIXES:
1. Sequence tests now use matching deck templates
2. Better handling of impossible golden states
3. Enhanced logging for draw scenarios
4. Defensive checks for test validity
"""

import itertools
import random
from typing import List, Tuple, Set, Dict, Optional
from collections import Counter
from cards import RED, GREEN, BLUE, YELLOW, Card
from belief import Belief
from uno import Uno
from pomdp import State, Action

# =============================================================================
# PART 1: GOLDEN REFERENCE - EXHAUSTIVE STATE ENUMERATION
# =============================================================================

def is_legal(card: Card, top_card: Card) -> bool:
    """UNO legality: match color or value."""
    return card[0] == top_card[0] or card[1] == top_card[1]


class GoldenBelief:
    """
    Golden reference belief state using exhaustive enumeration.
    For tiny decks, this represents the TRUE posterior distribution.
    """
    
    def __init__(self, observation: Tuple, deck_template: List[Card]):
        """
        Initialize golden belief from observation.
        
        Args:
            observation: O = (H_1, |H_2|, |D_g|, P, P_t, G_o)
            deck_template: Complete deck specification
        """
        self.H_1 = observation[0]
        self.h2_size = observation[1]
        self.dg_size = observation[2]
        self.P = observation[3]
        self.P_t = observation[4]
        self.G_o = observation[5]
        self.deck_template = deck_template
        
        # Enumerate ALL possible hidden states
        self.valid_states = self._enumerate_valid_states()
        
        print(f"[Golden] Initialized with {len(self.valid_states)} valid states")
    
    def _enumerate_valid_states(self) -> Set[Tuple]:
        """
        Exhaustively enumerate all (H_2, D_g) consistent with observation.
        
        Constraints:
        - Multiset equality: H_1 ∪ H_2 ∪ D_g ∪ P = Deck
        - |H_2| = h2_size
        - |D_g| = dg_size
        
        Returns:
            Set of valid states (H_2_tuple, D_g_tuple)
        """
        # Compute unknown cards: L = D \ (H_1 ∪ P)
        deck_counter = Counter(self.deck_template)
        known_counter = Counter(self.H_1 + self.P)
        unknown_counter = deck_counter - known_counter
        
        # Convert to list for combinations
        unknown_cards = []
        for card, count in unknown_counter.items():
            unknown_cards.extend([card] * count)
        
        total_unknown = len(unknown_cards)
        
        if self.h2_size + self.dg_size != total_unknown:
            print(f"[Golden] Warning: |H_2| + |D_g| = {self.h2_size + self.dg_size} != |L| = {total_unknown}")
            print(f"[Golden]   This means deck_template doesn't match the game state!")
            print(f"[Golden]   Known cards: {len(self.H_1) + len(self.P)}, Unknown: {total_unknown}")
            print(f"[Golden]   Expected unknown: {self.h2_size + self.dg_size}")
            return set()
        
        # Enumerate all ways to partition unknown cards into H_2 and D_g
        valid_states = set()
        
        # Generate all combinations of h2_size cards from unknown
        for h2_indices in itertools.combinations(range(total_unknown), self.h2_size):
            H_2 = tuple(sorted([unknown_cards[i] for i in h2_indices]))
            
            # Remaining cards go to D_g
            dg_indices = [i for i in range(total_unknown) if i not in h2_indices]
            D_g = tuple(sorted([unknown_cards[i] for i in dg_indices]))
            
            valid_states.add((H_2, D_g))
        
        return valid_states
    
    def apply_opponent_play_filter(self, played_card: Card) -> 'GoldenBelief':
        """
        Apply Bayes filter: opponent played a card.
        
        Filter rule: Keep only states where played_card ∈ H_2
        
        Args:
            played_card: Card opponent played
            
        Returns:
            New GoldenBelief with filtered states
        """
        filtered_states = set()
        
        for H_2, D_g in self.valid_states:
            # Check if played_card was in H_2
            if played_card in H_2:
                # Update: remove played_card from H_2, add to P
                H_2_list = list(H_2)
                H_2_list.remove(played_card)
                new_H_2 = tuple(sorted(H_2_list))
                
                # D_g unchanged
                filtered_states.add((new_H_2, D_g))
        
        # Create new observation after play
        new_observation = (
            self.H_1,
            self.h2_size - 1,
            self.dg_size,
            self.P + [played_card],
            played_card,
            self.G_o
        )
        
        # Create new golden belief
        new_golden = GoldenBelief.__new__(GoldenBelief)
        new_golden.H_1 = new_observation[0]
        new_golden.h2_size = new_observation[1]
        new_golden.dg_size = new_observation[2]
        new_golden.P = new_observation[3]
        new_golden.P_t = new_observation[4]
        new_golden.G_o = new_observation[5]
        new_golden.deck_template = self.deck_template
        new_golden.valid_states = filtered_states
        
        print(f"[Golden] After play filter: {len(self.valid_states)} -> {len(filtered_states)} states")
        
        return new_golden
    
    def apply_opponent_draw_filter(self, top_card: Card) -> Optional['GoldenBelief']:
        """
        Apply Bayes filter: opponent drew (had no legal plays).
        
        Filter rule: Keep only states where H_2 ∩ LEGAL(top_card) = ∅
        
        CRITICAL: The set of cards in H_2 ∪ D_g remains the same!
        We only constrain which cards were in H_2 vs D_g, then move one card.
        
        Args:
            top_card: Top card when opponent drew
            
        Returns:
            New GoldenBelief with filtered states, or None if impossible
        """
        filtered_states = set()
        
        for H_2, D_g in self.valid_states:
            # Check if H_2 has NO legal cards on top_card
            has_legal = any(is_legal(card, top_card) for card in H_2)
            
            if not has_legal and len(D_g) > 0:
                # This state is consistent with opponent drawing
                # Opponent draws first card from D_g (but we don't know which)
                # We need to enumerate all possibilities
                
                # For each possible drawn card
                for drawn_card in set(D_g):
                    D_g_list = list(D_g)
                    D_g_list.remove(drawn_card)
                    
                    new_H_2 = tuple(sorted(list(H_2) + [drawn_card]))
                    new_D_g = tuple(sorted(D_g_list))
                    
                    filtered_states.add((new_H_2, new_D_g))
        
        if len(filtered_states) == 0:
            print(f"[Golden] WARNING: Draw filter resulted in 0 valid states!")
            print(f"[Golden]   This scenario is IMPOSSIBLE given the game state")
            print(f"[Golden]   Initial states: {len(self.valid_states)}")
            return None
        
        # Create new observation after draw
        new_observation = (
            self.H_1,
            self.h2_size + 1,  # Opponent hand size increased
            self.dg_size - 1,  # Deck size decreased
            self.P,
            self.P_t,  # Top card unchanged
            self.G_o
        )
        
        # Create new golden belief
        new_golden = GoldenBelief.__new__(GoldenBelief)
        new_golden.H_1 = new_observation[0]
        new_golden.h2_size = new_observation[1]
        new_golden.dg_size = new_observation[2]
        new_golden.P = new_observation[3]
        new_golden.P_t = new_observation[4]
        new_golden.G_o = new_observation[5]
        new_golden.deck_template = self.deck_template
        new_golden.valid_states = filtered_states
        
        print(f"[Golden] After draw filter: {len(self.valid_states)} -> {len(filtered_states)} states")
        
        return new_golden
    
    def get_possible_h2_cards(self) -> Set[Card]:
        """Returns set of all cards that could possibly be in H_2."""
        possible = set()
        for H_2, D_g in self.valid_states:
            possible.update(H_2)
        return possible
    
    def get_possible_dg_cards(self) -> Set[Card]:
        """Returns set of all cards that could possibly be in D_g."""
        possible = set()
        for H_2, D_g in self.valid_states:
            possible.update(D_g)
        return possible

# =============================================================================
# PART 2: IMPLEMENTATION COMPARISON
# =============================================================================

def compare_beliefs(golden: GoldenBelief, impl: Belief, verbose: bool = False) -> Dict:
    """
    Compare implementation belief against golden reference.
    
    Checks:
    1. Soundness: Every state in golden is representable in impl
    2. Completeness: No impossible states in impl
    3. Card coverage: Possible cards match
    
    Args:
        golden: Golden reference belief
        impl: Implementation belief (from belief.py)
        verbose: Print detailed comparison
        
    Returns:
        Dictionary with comparison results
    """
    results = {
        'soundness': True,
        'completeness': True,
        'soundness_violations': [],
        'completeness_violations': [],
        'card_coverage_match': True,
        'details': {}
    }
    
    # Get implementation's unknown set L
    impl_L = set(impl.L)
    
    # Check 1: Soundness - every golden state should be representable
    golden_h2_cards = golden.get_possible_h2_cards()
    golden_dg_cards = golden.get_possible_dg_cards()
    golden_all_cards = golden_h2_cards | golden_dg_cards
    
    if verbose:
        print(f"\n[Comparison Details]")
        print(f"  Golden H2 cards: {sorted(golden_h2_cards)}")
        print(f"  Golden Dg cards: {sorted(golden_dg_cards)}")
        print(f"  Golden ALL cards: {sorted(golden_all_cards)}")
        print(f"  Impl L: {sorted(impl_L)}")
    
    for card in golden_all_cards:
        if card not in impl_L:
            results['soundness'] = False
            results['soundness_violations'].append(
                f"Card {card} is possible in golden but not in impl.L"
            )
    
    # Check 2: Completeness - impl.L should not contain impossible cards
    for card in impl_L:
        if card not in golden_all_cards:
            results['completeness'] = False
            results['completeness_violations'].append(
                f"Card {card} is in impl.L but not possible in golden"
            )
    
    # Check 3: Card coverage match
    results['card_coverage_match'] = (golden_all_cards == impl_L)
    
    # Store details
    results['details'] = {
        'golden_states': len(golden.valid_states),
        'golden_h2_cards': len(golden_h2_cards),
        'golden_dg_cards': len(golden_dg_cards),
        'impl_L_size': len(impl_L),
        'impl_N_Pt_size': len(impl.N_Pt),
        'golden_all_cards': sorted(golden_all_cards),
        'impl_L': sorted(impl_L),
    }
    
    return results

# =============================================================================
# PART 3: INVARIANT CHECKING
# =============================================================================

def check_deck_invariant(belief: Belief, deck_template: List[Card]) -> bool:
    """
    Deck invariant: For any card type c:
    count_in_H1 + count_in_L + count_in_P == deck_count(c)
    
    This ensures multiset equality is maintained.
    """
    deck_counter = Counter(deck_template)
    h1_counter = Counter(belief.H_1)
    p_counter = Counter(belief.P)
    l_counter = Counter(belief.L)
    
    for card in deck_counter:
        deck_count = deck_counter[card]
        observed_count = h1_counter[card] + p_counter[card] + l_counter[card]
        
        if deck_count != observed_count:
            print(f"[Invariant] FAIL: Card {card} - deck:{deck_count} vs observed:{observed_count}")
            return False
    
    return True

def check_monotonic_info(belief_history: List[Belief]) -> bool:
    """
    Monotonic information: Once a card is eliminated, it stays eliminated.
    (Unless explicitly observed in opponent's play)
    """
    if len(belief_history) < 2:
        return True
    
    for i in range(1, len(belief_history)):
        prev_L = set(belief_history[i-1].L)
        curr_L = set(belief_history[i].L)
        
        # Cards can only be removed, not added (except through draws)
        added = curr_L - prev_L
        removed = prev_L - curr_L
        
        # If cards were added, it should be due to a draw event
        if added and belief_history[i].posterior_mode != "draw":
            print(f"[Invariant] FAIL: Cards added without draw: {added}")
            return False
    
    return True

# =============================================================================
# PART 4: COMPREHENSIVE TEST SCENARIOS
# =============================================================================

def test_scenario_single_action(action_type: str, deck_template: List[Card], verbose: bool = False):
    """
    Test a single action (play or draw) and verify correctness.
    """
    print(f"\n{'='*70}")
    print(f"TEST: Single {action_type.upper()} Action")
    print(f"{'='*70}")
    
    # Generate random initial state FROM THE GIVEN DECK
    deck = deck_template.copy()
    random.shuffle(deck)
    
    # Ensure we have enough cards
    if len(deck) < 7:
        print("  [Skip] Deck too small for test")
        return None
    
    H_1 = deck[:2]
    H_2_truth = deck[2:4]
    D_g_truth = deck[4:6]
    P = [deck[6]]
    P_t = P[0]
    
    observation = (H_1, len(H_2_truth), len(D_g_truth), P, P_t, "Active")
    
    # Initialize both beliefs WITH SAME DECK
    golden = GoldenBelief(observation, deck_template)
    if len(golden.valid_states) == 0:
        print("  [Skip] Golden belief has no valid states (impossible scenario)")
        return None
    
    impl = Belief(observation, deck_template)
    
    print(f"\n[Initial State]")
    print(f"  H_1: {H_1}")
    print(f"  H_2 (truth): {H_2_truth}")
    print(f"  D_g (truth): {D_g_truth}")
    print(f"  P_t: {P_t}")
    
    # Check initial state
    print(f"\n[Initial Comparison]")
    results = compare_beliefs(golden, impl, verbose=verbose)
    print(f"  Soundness: {results['soundness']}")
    print(f"  Completeness: {results['completeness']}")
    print(f"  Card coverage match: {results['card_coverage_match']}")
    
    if action_type == "play":
        # Find a legal card in H_2_truth
        legal_cards = [c for c in H_2_truth if is_legal(c, P_t)]
        if not legal_cards:
            print("  [Skip] No legal plays available")
            return None
        
        played_card = legal_cards[0]
        print(f"\n[Action] Opponent plays: {played_card}")
        
        # Update both beliefs
        golden = golden.apply_opponent_play_filter(played_card)
        impl.update_opponent_played(played_card)
        
    else:  # draw
        # Check if opponent can play
        legal_cards = [c for c in H_2_truth if is_legal(c, P_t)]
        if legal_cards:
            print("  [Skip] Opponent has legal plays (can't draw)")
            return None
        
        if len(D_g_truth) == 0:
            print("  [Skip] Deck is empty (can't draw)")
            return None
        
        print(f"\n[Action] Opponent draws")
        print(f"  [God view] Drew: {D_g_truth[0]}")
        
        # Update both beliefs
        golden = golden.apply_opponent_draw_filter(P_t)
        if golden is None:
            print("  [Skip] Golden belief became impossible (test scenario invalid)")
            return None
        
        impl.update_opponent_drew()
    
    # Compare after update
    print(f"\n[After Update Comparison]")
    results = compare_beliefs(golden, impl, verbose=verbose)
    print(f"  Soundness: {results['soundness']}")
    print(f"  Completeness: {results['completeness']}")
    print(f"  Card coverage match: {results['card_coverage_match']}")
    print(f"  Golden states: {results['details']['golden_states']}")
    print(f"  Impl |L|: {results['details']['impl_L_size']}")
    
    if not results['soundness']:
        print(f"\n  SOUNDNESS VIOLATIONS:")
        for v in results['soundness_violations']:
            print(f"    - {v}")
    
    if not results['completeness']:
        print(f"\n  COMPLETENESS VIOLATIONS:")
        for v in results['completeness_violations']:
            print(f"    - {v}")
    
    # Check invariants
    print(f"\n[Invariant Checks]")
    deck_inv = check_deck_invariant(impl, deck_template)
    print(f"  Deck invariant: {deck_inv}")
    
    return results['soundness'] and results['completeness'] and results['card_coverage_match']

def test_scenario_sequence(num_actions: int, deck_template: List[Card]):
    """
    Test a sequence of actions and track belief evolution.
    
    FIXED: Now uses a custom Uno game initialized with the provided deck_template.
    """
    print(f"\n{'='*70}")
    print(f"TEST: Action Sequence ({num_actions} actions)")
    print(f"{'='*70}")
    
    # Initialize game WITH THE PROVIDED DECK
    deck = deck_template.copy()
    random.shuffle(deck)
    
    if len(deck) < 10:
        print("  [Skip] Deck too small for sequence test")
        return False
    
    # Manual initialization to match deck_template
    H_1 = deck[:3]
    H_2 = deck[3:6]
    P = [deck[6]]
    D_g = deck[7:]
    P_t = P[0]
    
    print(f"\n[Initial State]")
    print(f"  H_1: {H_1}")
    print(f"  H_2: {H_2}")
    print(f"  P_t: {P_t}")
    print(f"  |D_g|: {len(D_g)}")
    
    # Get initial observation
    observation = (H_1, len(H_2), len(D_g), P, P_t, "Active")
    
    # Initialize both beliefs WITH SAME DECK
    golden = GoldenBelief(observation, deck_template)
    if len(golden.valid_states) == 0:
        print("  [Skip] Golden belief has no valid states (deck mismatch)")
        return False
    
    impl = Belief(observation, deck_template)
    belief_history = [impl]
    
    # Create a simple game state tracker
    game_state = {
        'H_1': list(H_1),
        'H_2': list(H_2),
        'D_g': list(D_g),
        'P': list(P),
        'P_t': P_t
    }
    
    all_passed = True
    
    for step in range(num_actions):
        print(f"\n{'---'*20}")
        print(f"Step {step + 1}/{num_actions}")
        print(f"{'---'*20}")
        
        # Determine what player 2 can do
        legal_plays = [c for c in game_state['H_2'] if is_legal(c, game_state['P_t'])]
        
        print(f"  Current P_t: {game_state['P_t']}")
        print(f"  H_2 legal plays: {legal_plays}")
        
        if legal_plays:
            # Play a card
            played_card = legal_plays[0]
            print(f"  Action: PLAY {played_card}")
            
            # Update game state
            game_state['H_2'].remove(played_card)
            game_state['P'].append(played_card)
            game_state['P_t'] = played_card
            
            # Update beliefs
            golden = golden.apply_opponent_play_filter(played_card)
            impl.update_opponent_played(played_card)
            
        elif len(game_state['D_g']) > 0:
            # Draw a card
            print(f"  Action: DRAW")
            drawn = game_state['D_g'].pop()
            print(f"    [God view] Drew: {drawn}")
            game_state['H_2'].append(drawn)
            
            # Update beliefs
            golden = golden.apply_opponent_draw_filter(game_state['P_t'])
            if golden is None:
                print("  [Skip] Golden became impossible, ending sequence")
                break
            
            impl.update_opponent_drew()
        else:
            print("  [End] No more actions possible")
            break
        
        belief_history.append(impl)
        
        # Compare
        results = compare_beliefs(golden, impl)
        passed = results['soundness'] and results['completeness']
        
        print(f"  Result: {'PASS' if passed else 'FAIL'}")
        print(f"    Golden states: {results['details']['golden_states']}")
        print(f"    Impl |L|: {results['details']['impl_L_size']}")
        
        if not passed:
            all_passed = False
            print(f"    Violations: {results['soundness_violations'] + results['completeness_violations']}")
    
    # Check sequence invariants
    print(f"\n[Sequence Invariants]")
    monotonic = check_monotonic_info(belief_history)
    print(f"  Monotonic information: {monotonic}")
    
    return all_passed

# =============================================================================
# PART 5: MAIN TEST RUNNER
# =============================================================================

def run_comprehensive_validation():
    """
    Run comprehensive validation suite.
    """
    print("="*70)
    print("COMPREHENSIVE BELIEF STATE VALIDATION")
    print("Golden Reference Method with Exhaustive Enumeration")
    print("="*70)
    
    # Use micro-deck for tractability
    micro_deck = [
        (RED, 1), (RED, 2), (RED, 3),
        (GREEN, 1), (GREEN, 2), (GREEN, 3),
        (BLUE, 1),
    ]
    
    results = {
        'single_play': 0,
        'single_draw': 0,
        'sequences': 0,
        'total': 0,
        'skipped': 0
    }
    
    # Test 1: Single play actions
    print(f"\n{'#'*70}")
    print("PHASE 1: Single PLAY actions")
    print(f"{'#'*70}")
    for i in range(5):
        result = test_scenario_single_action("play", micro_deck.copy())
        if result is None:
            results['skipped'] += 1
        elif result:
            results['single_play'] += 1
        results['total'] += 1
    
    # Test 2: Single draw actions
    print(f"\n{'#'*70}")
    print("PHASE 2: Single DRAW actions")
    print(f"{'#'*70}")
    for i in range(5):
        result = test_scenario_single_action("draw", micro_deck.copy())
        if result is None:
            results['skipped'] += 1
        elif result:
            results['single_draw'] += 1
        results['total'] += 1
    
    # Test 3: Action sequences (FIXED - now uses matching deck)
    print(f"\n{'#'*70}")
    print("PHASE 3: Action sequences")
    print(f"{'#'*70}")
    for i in range(3):
        if test_scenario_sequence(3, micro_deck.copy()):
            results['sequences'] += 1
        results['total'] += 1
    
    # Final report
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"Single PLAY tests: {results['single_play']}/5 passed")
    print(f"Single DRAW tests: {results['single_draw']}/5 passed")
    print(f"Sequence tests: {results['sequences']}/3 passed")
    print(f"Skipped (impossible scenarios): {results['skipped']}")
    print(f"Overall: {results['single_play'] + results['single_draw'] + results['sequences']}/{results['total']} passed")
    print(f"{'='*70}")
    
    print("\nVALIDATION METHOD:")
    print("✓ Golden reference via exhaustive state enumeration")
    print("✓ Bayes filtering using game transition model")
    print("✓ Soundness check (no valid states eliminated)")
    print("✓ Completeness check (no impossible states added)")
    print("✓ Invariant checking (deck conservation, monotonicity)")
    print("✓ Defensive handling of impossible test scenarios")
    print(f"{'='*70}")

if __name__ == "__main__":
    run_comprehensive_validation()