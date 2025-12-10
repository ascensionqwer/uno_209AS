import sys
import os
import random
from collections import Counter

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage_1_mini_uno.flexible_uno import FlexibleUno
from stage_1_mini_uno.online_solver_adapter import MiniUnoAI
from cards import RED, BLUE

def print_belief_stats(ai, label):
    """Prints belief statistics."""
    belief = ai.belief
    probs = belief.get_card_probabilities("H_2")
    
    print(f"\n--- Belief Stats ({label}) ---")
    print(f"Entropy: {belief.entropy():.2f}")
    print(f"Posterior Mode: {belief.posterior_mode}")
    
    # Sort by probability
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    
    print("Top 5 likely cards in Opponent Hand:")
    for card, prob in sorted_probs[:5]:
        print(f"  {card}: {prob:.1%}")
        
    # Also print probability of specific cards if they exist in the deck
    # to see if they drop to 0 when played
    return probs

def run_verification():
    # Use Mini config: 2 colors, 0-2 ranks, 2 copies
    game = FlexibleUno(
        colors=[RED, BLUE],
        ranks=[0, 1, 2],
        copies=2
    )
    game.new_game(seed=42, deal=2)
    
    ai = MiniUnoAI(player_id=1, num_samples=100, lookahead=2)
    ai.init_belief(game)
    
    print("=== Initial State ===")
    game.print_S()
    print_belief_stats(ai, "Initial")
    
    # Run a few turns
    for turn in range(1, 6):
        print(f"\n\n=== Turn {turn} ===")
        
        if turn % 2 != 0: # Player 1 (AI)
            print("Player 1 (AI) Turn")
            action = ai.choose_action()
            print(f"AI Action: {action}")
            game.execute_action(action, player=1)
            
            # AI's own action doesn't update belief about opponent directly, 
            # but changes the known state (P, H_1). 
            # The belief class should handle this via 'update' with new observation,
            # or we assume AI knows its own state changes.
            # In our implementation, update_belief is called with OPPONENT action.
            # But we should also update observation for AI's own move?
            # Actually, init_belief takes game state. 
            # Let's see if we need to manually update belief for AI's move.
            # Usually POMDP updates belief after action + observation.
            # Here we only update when opponent moves.
            # But the 'L' (unknown cards) changes when AI plays (card goes to P, so it's known).
            # Let's force an update with None action to refresh observation.
            ai.update_belief(None) 
            
        else: # Player 2 (Opponent)
            print("Player 2 (Opponent) Turn")
            actions = game.get_legal_actions(player=2)
            if not actions:
                print("Opponent has no legal actions!")
                break
                
            # Pick a specific action to test updates
            # Try to pick a Play action if possible
            play_actions = [a for a in actions if a.is_play()]
            if play_actions:
                action = play_actions[0]
            else:
                action = actions[0]
                
            print(f"Opponent Action: {action}")
            
            # Check prob of this card BEFORE execution
            if action.is_play():
                probs = ai.belief.get_card_probabilities("H_2")
                print(f"Prob of {action.X_1} before play: {probs.get(action.X_1, 0):.1%}")
            
            game.execute_action(action, player=2)
            
            # Update belief
            ai.update_belief(action)
            
            # Check stats AFTER update
            print_belief_stats(ai, "After Opponent Move")
            
            if action.is_play():
                probs = ai.belief.get_card_probabilities("H_2")
                prob_after = probs.get(action.X_1, 0)
                print(f"Prob of {action.X_1} after play: {prob_after:.1%}")
                if prob_after > 0:
                    print("WARNING: Played card should have 0 probability (unless multiple copies exist and one remains)")
                else:
                    print("VERIFIED: Played card probability dropped to 0 (or decreased correctly).")

def run_draw_inference_test():
    print("\n\n" + "="*40)
    print("Running Draw Inference Test")
    print("Scenario: Opponent draws. We should infer they had NO legal cards.")
    print("="*40)
    
    # Setup: P_t is Red 0. Opponent has only Blue cards (no 0s).
    # Deck: Mini Uno (12 cards)
    # H_1 (AI): [Blue 0]
    # H_2 (Opp): [Blue 1, Blue 2]
    # P: [Red 0] (Top)
    # D_g: Remainder
    
    # Manually construct state
    H_1 = [('B', 0)]
    H_2 = [('B', 1), ('B', 2)]
    P = [('R', 0)]
    
    # Remainder of deck
    full_deck = FlexibleUno(colors=[RED, BLUE], ranks=[0, 1, 2], copies=2).build_number_deck()
    used = H_1 + H_2 + P
    D_g = list((Counter(full_deck) - Counter(used)).elements())
    
    game = FlexibleUno(colors=[RED, BLUE], ranks=[0, 1, 2], copies=2)
    game.H_1 = H_1
    game.H_2 = H_2
    game.P = P
    game.D_g = D_g
    game.create_S() # Initialize state
    
    ai = MiniUnoAI(player_id=1, num_samples=200, lookahead=2)
    ai.init_belief(game)
    
    print("=== Initial State ===")
    print(f"P_t: {game.P_t} (Red 0)")
    print(f"Opponent Hand (True): {game.H_2}")
    
    # Check belief before draw
    print_belief_stats(ai, "Before Draw")
    
    # Check prob of a legal card (e.g., Red 1)
    # It should be > 0 initially because we don't know H_2
    probs = ai.belief.get_card_probabilities("H_2")
    r1_prob = probs.get(('R', 1), 0)
    print(f"Prob of Red 1 (Legal Move) before draw: {r1_prob:.1%}")
    
    # Execute Draw Action
    # Opponent must draw
    actions = game.get_legal_actions(player=2)
    draw_action = [a for a in actions if a.is_draw()][0]
    print(f"\nOpponent Action: {draw_action}")
    
    game.execute_action(draw_action, player=2)
    ai.update_belief(draw_action)
    
    # Check belief after draw
    print_belief_stats(ai, "After Draw")
    
    probs = ai.belief.get_card_probabilities("H_2")
    r1_prob_after = probs.get(('R', 1), 0)
    print(f"Prob of Red 1 (Legal Move) after draw: {r1_prob_after:.1%}")
    
    if r1_prob_after == 0.0:
        print("VERIFIED: Probability of legal cards dropped to 0%. Solver inferred opponent had no legal moves.")
    else:
        print(f"WARNING: Probability of legal card is {r1_prob_after:.1%}, expected 0%.")

if __name__ == "__main__":
    run_verification()
    run_draw_inference_test()
