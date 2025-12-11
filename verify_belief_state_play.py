# verify_belief_state_play_verbose.py

import itertools
from belief import Belief
from cards import RED, GREEN

# State type: S = (H_1, H_2, D_g, P, P_t, G_o)

def is_legal(card, top_card):
    return card[0] == top_card[0] or card[1] == top_card[1]

def apply_play_action(state, play_card):
    H_1, H_2, D_g, P, P_t, G_o = state
    H_1p = list(H_1)
    H_1p.remove(play_card)
    Pp = list(P) + [play_card]
    P_tp = play_card
    H_2p = list(H_2)
    D_gp = list(D_g)
    G_op = "GameOver" if len(H_1p) == 0 else "Active"
    return (H_1p, H_2p, D_gp, Pp, P_tp, G_op)

def value_keep_green(state):
    H_1, H_2, D_g, P, P_t, G_o = state
    return 1.0 if P_t[0] == GREEN else 0.0


def verify_color_preference_verbose():
    # Full tiny deck
    full_deck = [
        (RED, 1), (RED, 2), (RED, 3),
        (GREEN, 1), (GREEN, 2), (GREEN, 3),
    ]

    # Scenario:
    # H1 = {R1, G2}, P_t = G1, opponent size 1, opponent previously played G3.
    H_1 = [(RED, 1), (GREEN, 2)]
    h2_size = 1
    P = [(GREEN, 3)]
    P_t = (GREEN, 1)
    dg_size = 2
    G_o = "Active"

    observation = (H_1, h2_size, dg_size, P, P_t, G_o)

    b = Belief(observation, deck_template=full_deck)
    print("=== Belief Object Summary ===")
    print(b)

    # Step 1: Show known vs unknown cards
    known = set(H_1 + P)
    L = b.L
    print("\nKnown cards (H1 ∪ P):", known)
    print("Unknown L (from Belief):", L)
    print(f"|L| = {len(L)}")

    # Sanity: manual L = D \ (H1 ∪ P)
    manual_L = [c for c in full_deck if c not in known]
    print("Manual L (for cross-check):", manual_L)

    # Step 2: show LEGAL(P_t) ∩ L
    N_Pt = b.N_Pt
    print("\nLEGAL(P_t) ∩ L (Belief.N_Pt):", N_Pt)
    print(f"|N(P_t)| = {len(N_Pt)}")

    # Step 3: show all possible assignments (H2, Dg) consistent with L and sizes
    print("\nAll possible (H2, D_g) partitions consistent with L and sizes:")
    all_assignments = []
    for h2_card in set(L):
        # H2 must be a single card; Dg is the rest but truncated to dg_size
        H2 = [h2_card]
        remaining = list(L)
        remaining.remove(h2_card)
        # D_g must be any 2-card subset of remaining
        for Dg in itertools.combinations(remaining, dg_size):
            all_assignments.append((tuple(H2), tuple(Dg)))

    # Because prior is uniform over 1-card subsets of L, H2 is uniform over L.
    print(f"Total distinct (H2,D_g) assignments: {len(all_assignments)}")
    for i, (H2, Dg) in enumerate(all_assignments, 1):
        print(f"  #{i}: H2 = {H2}, D_g = {Dg}")

    # Step 4: verify probability mass for each opponent card
    print("\n=== Analytic probabilities for H2 ===")
    print("Prior: uniform over 1-card subsets of L")
    for c in set(L):
        # P(H2 = {c}) = 1/|L|
        p = 1.0 / len(L)
        print(f"  P(H2 = {{{c}}}) = {p:.3f}")
    print(f"Sum over c ∈ L: {len(L)} * (1/{len(L)}) = 1.0")

    # Step 5: compare with Belief.get_card_probabilities
    probs_h2 = b.get_card_probabilities("H_2")
    print("\nBelief.get_card_probabilities('H_2') output:")
    for c, p in sorted(probs_h2.items(), key=lambda x: x[0]):
        print(f"  {c}: {p:.3f}")
    print(f"Sum P(c in H2) = {sum(probs_h2.values()):.3f} (expected {h2_size})")

    # Step 6: legal actions from H1
    legal = [c for c in H_1 if is_legal(c, P_t)]
    print("\nHand:", H_1)
    print("Top:", P_t)
    print("Legal plays:", legal)

    # Step 7: analytic Q-values under value_keep_green
    print("\n=== Analytic Q-values for 'keep green' ===")

    # For this one-step, top card after play is exactly the card you play.
    # So:
    # - If you play G2: top is green → reward 1 ALWAYS.
    # - If you play R1: top is red → reward 0 ALWAYS.
    for card in legal:
        if card[0] == GREEN:
            q_analytic = 1.0
        else:
            q_analytic = 0.0
        print(f"  Analytic Q(play {card}) = {q_analytic:.3f}")

    # Step 8: Monte Carlo estimate via samples
    n_samples = 50
    samples = b.sample_states(n_samples)
    q_mc = {card: 0.0 for card in legal}
    for card in legal:
        total = 0.0
        for s in samples:
            s_prime = apply_play_action(s, card)
            total += value_keep_green(s_prime)
        q_mc[card] = total / n_samples

    print("\nMonte Carlo estimated Q-values (from sampled worlds):")
    for card, val in q_mc.items():
        print(f"  Q_MC(play {card}) ≈ {val:.3f}")

    best = max(q_mc.items(), key=lambda x: x[1])[0]
    print(f"\nBest action under this value (MC): play {best}")


if __name__ == "__main__":
    verify_color_preference_verbose()
