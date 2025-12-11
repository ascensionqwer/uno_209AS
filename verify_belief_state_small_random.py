import random
import itertools
from typing import List, Tuple
from belief import Belief
from cards import RED, GREEN, YELLOW, BLUE, Card

FULL_DECK: List[Card] = [
    (RED, 1), (RED, 2), (RED, 3),
    (GREEN, 1), (GREEN, 2), (GREEN, 3),
]

LOG_FILE = "belief_random_small_log.txt"


def is_legal(card: Card, top_card: Card) -> bool:
    return card[0] == top_card[0] or card[1] == top_card[1]


def apply_play_action(state, play_card: Card):
    H_1, H_2, D_g, P, P_t, G_o = state
    H_1p = list(H_1)
    H_1p.remove(play_card)
    Pp = list(P) + [play_card]
    P_tp = play_card
    H_2p = list(H_2)
    D_gp = list(D_g)
    G_op = "GameOver" if len(H_1p) == 0 else "Active"
    return (H_1p, H_2p, D_gp, Pp, P_tp, G_op)


def value_fn(state) -> float:
    """
    Placeholder value function used only to define 'optimal'.
    Currently: returns 1 if top color is GREEN, else 0.
    """
    H_1, H_2, D_g, P, P_t, G_o = state
    return 1.0 if P_t[0] == GREEN else 0.0


def generate_random_scenario(rng: random.Random):
    """
    God state S = (H1_truth, H2_truth, Dg_truth, P, P_t, Go)
    Observation O = (H1_truth, |H2|, |Dg|, P, P_t, Go).
    """
    full_deck = list(FULL_DECK)
    rng.shuffle(full_deck)

    # H1 gets 2 cards
    H1_truth = full_deck[:2]
    rest = full_deck[2:]

    # H2 has 1 card
    h2_size = 1
    H2_truth = rest[:h2_size]
    rest = rest[h2_size:]

    if len(rest) < 2:
        return None, None, None

    # top card and past card
    P_t = rest.pop(0)
    P = [rest.pop(0)]

    # remaining deck
    Dg_truth = rest
    dg_size = len(Dg_truth)
    G_o = "Active"

    observation = (H1_truth, h2_size, dg_size, P, P_t, G_o)
    return full_deck, (H1_truth, H2_truth, Dg_truth, P, P_t, G_o), observation


def enumerate_belief_worlds(full_deck: List[Card], observation) -> List[Tuple]:
    """
    Enumerate all concrete states S = (H_1, H_2, D_g, P, P_t, G_o)
    consistent with the observation O = (H_1, |H_2|, |D_g|, P, P_t, G_o)
    under the full_deck template.
    """
    H_1, h2_size, dg_size, P, P_t, G_o = observation

    # Known cards
    known = set(H_1 + P + [P_t])
    # Unknown pool L = deck \ known
    L = [c for c in full_deck if c not in known]

    worlds = []
    # Choose H2 as any h2_size-card subset of L
    for H2_tuple in itertools.combinations(L, h2_size):
        remaining = [c for c in L if c not in H2_tuple]
        # D_g must be any dg_size-card subset of remaining
        for Dg_tuple in itertools.combinations(remaining, dg_size):
            H_2 = list(H2_tuple)
            D_g = list(Dg_tuple)
            worlds.append((list(H_1), H_2, D_g, list(P), P_t, G_o))
    return worlds


def analyze_and_log_scenarios(num_trials: int = 1000):
    rng = random.Random(123)

    num_optimal = 0
    num_no_optimal = 0

    with open(LOG_FILE, "w") as f:
        f.write("=== Belief state random scenario log (enumerate all worlds) ===\n")
        f.write(f"Deck used: {FULL_DECK}\n")
        f.write(f"Total trials requested: {num_trials}\n\n")

    for t in range(num_trials):
        full_deck, god_state, observation = generate_random_scenario(rng)
        if observation is None:
            continue

        H1_truth, H2_truth, Dg_truth, P, P_t, G_o = god_state
        H_1, h2_size, dg_size, P_obs, P_t_obs, G_o_obs = observation

        legal = [c for c in H_1 if is_legal(c, P_t)]

        has_optimal = False
        best_card = None
        q = {}

        # Build belief object (for L, N(Pt), summary)
        b = Belief(observation, deck_template=full_deck)

        if len(legal) > 1:
            # Enumerate all concrete worlds
            worlds = enumerate_belief_worlds(full_deck, observation)
            # For this tiny 6-card deck, this list is small.

            # Assume uniform probability over all consistent worlds
            n_worlds = len(worlds)
            if n_worlds > 0:
                for card in legal:
                    total = 0.0
                    for s in worlds:
                        s_prime = apply_play_action(s, card)
                        total += value_fn(s_prime)
                    q[card] = total / n_worlds

                best_card, best_val = max(q.items(), key=lambda x: x[1])
                vals_sorted = sorted(q.values(), reverse=True)
                second_best_val = vals_sorted[1] if len(vals_sorted) > 1 else best_val
                eps = 1e-6
                has_optimal = (best_val - second_best_val) > eps
        # else: 0 or 1 legal play → no optimal by definition

        if has_optimal:
            num_optimal += 1
        else:
            num_no_optimal += 1

        # Log scenario, belief, and all worlds
        log_lines = []
        log_lines.append("Scenario #" + str(t + 1))

        # God view
        log_lines.append("  [God] H1_truth: " + str(H1_truth))
        log_lines.append("  [God] H2_truth: " + str(H2_truth))
        log_lines.append("  [God] Dg_truth: " + str(Dg_truth))
        log_lines.append("  [God] P: " + str(P))
        log_lines.append("  [God] P_t: " + str(P_t))

        # Observation
        log_lines.append("  [Obs] H1: " + str(H_1))
        log_lines.append("  [Obs] |H2|: " + str(h2_size) + " |D_g|: " + str(dg_size))
        log_lines.append("  [Obs] P: " + str(P_obs))
        log_lines.append("  [Obs] P_t: " + str(P_t_obs))
        log_lines.append("  [Obs] Legal plays: " + str(legal))

        # Belief info
        log_lines.append("  [Belief] " + repr(b))
        log_lines.append("  [Belief] L (unknown cards): " + str(b.L))
        log_lines.append("  [Belief] N(P_t) (legal unknown): " + str(b.N_Pt))

        # Enumerated worlds
        worlds = enumerate_belief_worlds(full_deck, observation)
        log_lines.append(f"  [Belief] Number of consistent worlds: {len(worlds)}")
        for i, s in enumerate(worlds, 1):
            H1w, H2w, Dgw, Pw, Pt_w, Go_w = s
            log_lines.append(f"    World #{i}: H2={H2w}, D_g={Dgw}")

        if q:
            log_lines.append("  [Belief] Exact Q-values under value_fn:")
            for card, val in q.items():
                log_lines.append(f"    {card} -> {round(val, 3)}")
        else:
            log_lines.append("  [Belief] Exact Q-values under value_fn: N/A (≤1 legal play)")

        if has_optimal:
            log_lines.append(f"  [Obs] Has optimal play: YES, best = {best_card}")
        else:
            log_lines.append("  [Obs] Has optimal play: NO")

        log_lines.append("")

        with open(LOG_FILE, "a") as f:
            f.write("\n".join(log_lines) + "\n")

    summary_lines = [
        "=== Summary over random scenarios ===",
        "Deck used: " + str(FULL_DECK),
        "Total trials: " + str(num_trials),
        "Scenarios with optimal play: " + str(num_optimal),
        "Scenarios with no optimal play: " + str(num_no_optimal),
    ]
    summary_text = "\n".join(summary_lines)

    print(summary_text)
    print(f"\nAll scenarios logged to {LOG_FILE}")

    with open(LOG_FILE, "a") as f:
        f.write("\n" + summary_text + "\n")


if __name__ == "__main__":
    analyze_and_log_scenarios(num_trials=1000)
