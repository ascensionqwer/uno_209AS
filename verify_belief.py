
from belief import Belief
from cards import RED, BLUE, GREEN, YELLOW
from collections import Counter

def verify_draw_logic():
    print("Verifying Belief State Draw Logic...")

    # 1. Setup a controlled scenario
    # H_1 (My Hand): [RED 1]
    # P (Played): []
    # P_t (Top Card): RED 2
    # D_g (Deck): Small, controlled set
    
    # We want to force a situation where the opponent draws.
    # Opponent Hand Size (k): 1
    
    # Let's define the "Unknown Set" L manually for this test context.
    # In the real code, L is computed from D \ (H_1 U P).
    # We will mock the observation passed to Belief.
    
    # O = (H_1, |H_2|, |D_g|, P, P_t, G_o)
    
    H_1 = [(RED, 1)]
    h2_size = 1
    dg_size = 5 # arbitrary, just needs to be > 0
    P = []
    P_t = (RED, 2)
    G_o = "Active"
    
    observation = (H_1, h2_size, dg_size, P, P_t, G_o)
    
    b = Belief(observation)
    
    # Now, we want to simulate the opponent drawing.
    # This implies they had NO legal cards.
    # Legal cards for RED 2 are: RED * or * 2.
    
    # Let's inspect L to see what's in it.
    print(f"Initial |L|: {len(b.L)}")
    
    # Call the update for opponent drawing
    print("Updating belief: Opponent Drew a card.")
    b.update_opponent_drew()
    
    # Now h2_size should be 2 (1 original + 1 drawn)
    print(f"New |H_2|: {b.h2_size}")
    
    # The crucial check:
    # The opponent had 1 card, which was NOT legal.
    # They drew 1 card.
    # The NEW hand (size 2) consists of:
    #   1. The old card (guaranteed NOT legal)
    #   2. The new card (could be ANYTHING from the remaining deck)
    
    # So, the new hand *could* contain a legal card (the one just drawn).
    
    # Let's sample many states and check the composition of H_2.
    n_samples = 1000
    samples = b.sample_states(n_samples)
    
    legal_in_hand_count = 0
    
    # Helper to check legality against P_t
    def is_legal(card, top_card):
        return card[0] == top_card[0] or card[1] == top_card[1]
        
    for s in samples:
        # s = (H_1, H_2, D_g, P, P_t, G_o)
        h2 = s[1]
        
        has_legal = False
        for card in h2:
            if is_legal(card, P_t):
                has_legal = True
                break
        
        if has_legal:
            legal_in_hand_count += 1
            
    print(f"Samples with at least one legal card in H_2: {legal_in_hand_count}/{n_samples}")
    
    if legal_in_hand_count == 0:
        print("FAILURE: Belief state assumes H_2 NEVER has a legal card after drawing.")
        print("Explanation: The drawn card is randomly taken from the deck, so it MIGHT be legal.")
        print("The current logic likely constrains ALL cards in H_2 to be non-legal.")
    else:
        print("SUCCESS: Belief state correctly allows legal cards in H_2 after drawing.")

if __name__ == "__main__":
    verify_draw_logic()
