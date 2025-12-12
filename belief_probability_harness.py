"""
Aggregate belief verification results over many random scenarios.

What it measures:
- PASS / FAIL / SKIP counts for:
  (1) opponent plays a card
  (2) opponent draws (no legal moves observed; drawn card is hidden)
  (3) opponent draws then immediately plays (reveals drawn card by playing it)

This aggregates *logical consistency* of belief updates, not policy quality.

Key point:
- The draw-then-play case must NOT fail just because the played card is legal.
  We update using belief.py's own update rules for that case.
"""

import argparse
import random
from collections import Counter
from typing import List, Tuple, Dict

from cards import RED, GREEN, BLUE, YELLOW, Card
from belief import Belief


def is_legal(card: Card, top_card: Card) -> bool:
    """UNO rule: Match Color or Match Value."""
    return card[0] == top_card[0] or card[1] == top_card[1]


def deck_template_extended() -> List[Card]:
    # Extended micro-deck with a few duplicates to stress-test multiset handling.
    return [
        (RED, 1), (RED, 2), (RED, 3), (RED, 4),
        (GREEN, 1), (GREEN, 2), (GREEN, 3), (GREEN, 4),
        (BLUE, 1), (BLUE, 2), (BLUE, 3), (BLUE, 4),
        (YELLOW, 1), (YELLOW, 2), (YELLOW, 3), (YELLOW, 4),
        # Duplicates:
        (RED, 1), (GREEN, 2), (BLUE, 3), (YELLOW, 4),
    ]


def deal_random_state(
    deck_template: List[Card],
    h1_size: int = 3,
    h2_size: int = 3,
    played_min: int = 1,
    played_max: int = 5,
) -> Tuple[List[Card], List[Card], List[Card], List[Card], Card]:
    """
    Generates a random valid UNO game state.

    Returns:
        H1: our hand (truth)
        H2_truth: opponent hand (truth)
        Dg_truth: remaining deck (truth, top is Dg_truth[0] if exists)
        P: played pile (including top)
        P_t: top card
    """
    deck = deck_template.copy()
    random.shuffle(deck)

    H1 = deck[:h1_size]
    H2_truth = deck[h1_size:h1_size + h2_size]

    remaining = deck[h1_size + h2_size:]

    played_count = random.randint(played_min, played_max)
    played_count = min(played_count, len(remaining))

    P = remaining[:played_count]
    if not P:
        # Ensure there is a top card in play
        P = [remaining[0]]
        remaining = remaining[1:]
    P_t = P[-1]

    # Remaining deck after the played pile
    Dg_truth = remaining[played_count:] if played_count <= len(remaining) else []
    return H1, H2_truth, Dg_truth, P, P_t


def multiset_L_from_belief(b: Belief) -> Counter:
    """Belief.L is a list with duplicates, so Counter correctly represents the multiset."""
    return Counter(b.L)


def remove_one(counter: Counter, card: Card) -> Counter:
    """Remove exactly one instance of card from a multiset Counter (if present)."""
    c = counter.copy()
    if c[card] <= 0:
        return c
    c[card] -= 1
    if c[card] == 0:
        del c[card]
    return c


def eliminate_legal_cards(L: Counter, top_card: Card) -> Tuple[Counter, Counter]:
    """
    Eliminate all card-instances from L that would be legal on top_card.
    Returns: (new_L, eliminated_multiset)
    """
    eliminated = Counter()
    new_L = L.copy()
    for card, k in list(L.items()):
        if is_legal(card, top_card):
            eliminated[card] += k
            del new_L[card]
    return new_L, eliminated


def scenario_play(deck_template: List[Card]) -> Tuple[str, Dict]:
    """
    PASS criteria (logical consistency):
      - played_card removed (one instance) from L
      - all remaining truth opponent cards remain feasible in L (as a multiset)
    """
    H1, H2_truth, Dg_truth, P, P_t = deal_random_state(deck_template)

    legal_h2 = [c for c in H2_truth if is_legal(c, P_t)]
    if not legal_h2:
        return "SKIP", {"reason": "no_legal_play_in_truth"}

    played_card = random.choice(legal_h2)

    obs = (H1, len(H2_truth), len(Dg_truth), P, P_t, "Active")
    b = Belief(obs, deck_template=deck_template)

    L = multiset_L_from_belief(b)
    L_after = remove_one(L, played_card)

    # Must remove exactly one instance if present
    removed_ok = (L_after[played_card] == L[played_card] - 1)

    # Remaining truth hand should still be feasible
    H2_after = H2_truth.copy()
    H2_after.remove(played_card)
    need = Counter(H2_after)
    feasible_ok = all(L_after[c] >= need[c] for c in need)

    passed = removed_ok and feasible_ok
    return ("PASS" if passed else "FAIL"), {
        "removed_ok": removed_ok,
        "feasible_ok": feasible_ok,
        "played_card": played_card,
    }


def scenario_draw(deck_template: List[Card], max_attempts: int = 60) -> Tuple[str, Dict]:
    """
    We must find a state where truth opponent has no legal plays and deck not empty.

    PASS criteria:
      - No card from the true opponent hand is eliminated
      - All eliminated cards are legal on the top card
    """
    for _ in range(max_attempts):
        H1, H2_truth, Dg_truth, P, P_t = deal_random_state(deck_template)
        if not Dg_truth:
            continue
        if not any(is_legal(c, P_t) for c in H2_truth):
            break
    else:
        return "SKIP", {"reason": "could_not_construct_forced_draw"}

    obs = (H1, len(H2_truth), len(Dg_truth), P, P_t, "Active")
    b = Belief(obs, deck_template=deck_template)
    L = multiset_L_from_belief(b)

    L_after, eliminated = eliminate_legal_cards(L, P_t)

    h2_counts = Counter(H2_truth)
    eliminated_ok = all(eliminated[c] == 0 for c in h2_counts)
    all_legal_ok = all(is_legal(c, P_t) for c in eliminated)

    passed = eliminated_ok and all_legal_ok
    return ("PASS" if passed else "FAIL"), {
        "eliminated_count": sum(eliminated.values()),
        "eliminated_ok": eliminated_ok,
        "all_legal_ok": all_legal_ok,
    }


def scenario_draw_then_play(deck_template: List[Card], max_attempts: int = 80) -> Tuple[str, Dict]:
    """
    Draw then immediate play.

    IMPORTANT: We must NOT fail simply because the played card is legal.
    In draw-then-play, the played card is often the drawn card, which is hidden at draw time.

    Therefore we update using belief.py's own update rules:
      - b.update_opponent_drew()
      - b.update_opponent_played(played_card)

    PASS criteria:
      - draw update succeeds
      - play update removes exactly one instance of played_card from belief.L
    """
    # Construct: opponent initially has no legal plays, deck not empty,
    # and the (truth) drawn card is legal so an immediate play is plausible.
    for _ in range(max_attempts):
        H1, H2_truth, Dg_truth, P, P_t = deal_random_state(deck_template)
        if not Dg_truth:
            continue
        if any(is_legal(c, P_t) for c in H2_truth):
            continue
        drawn_truth = Dg_truth[0]
        if is_legal(drawn_truth, P_t):
            played_card = drawn_truth
            break
    else:
        return "SKIP", {"reason": "could_not_construct_draw_then_play"}

    obs = (H1, len(H2_truth), len(Dg_truth), P, P_t, "Active")
    b = Belief(obs, deck_template=deck_template)

    # Step 1: opponent draws (unobserved card)
    try:
        b.update_opponent_drew()
    except Exception as e:
        return "FAIL", {"stage": "draw_update_exception", "err": str(e)}

    # Step 2: opponent plays (now observed)
    try:
        before_play = Counter(b.L)
        b.update_opponent_played(played_card)
        after_play = Counter(b.L)
    except Exception as e:
        return "FAIL", {"stage": "play_update_exception", "err": str(e)}

    removed_ok = (after_play[played_card] == before_play[played_card] - 1)
    passed = removed_ok

    return ("PASS" if passed else "FAIL"), {
        "removed_ok": removed_ok,
        "played_card": played_card,
        "L_count_before_play": before_play[played_card],
        "L_count_after_play": after_play.get(played_card, 0),
    }


def run(n: int, seed: int) -> None:
    random.seed(seed)
    deck = deck_template_extended()

    metrics = {
        "play": Counter(),
        "draw": Counter(),
        "draw_then_play": Counter(),
    }
    extra = {
        "draw_eliminated_counts": [],
    }

    for _ in range(n):
        status, info = scenario_play(deck)
        metrics["play"][status] += 1

        status, info = scenario_draw(deck)
        metrics["draw"][status] += 1
        if status == "PASS":
            extra["draw_eliminated_counts"].append(info["eliminated_count"])

        status, info = scenario_draw_then_play(deck)
        metrics["draw_then_play"][status] += 1

    def rate(pass_ct: int, total_ct: int) -> float:
        return (pass_ct / total_ct) if total_ct > 0 else 0.0

    def summarize(name: str, c: Counter) -> None:
        attempted = c["PASS"] + c["FAIL"]
        print(f"\n{name.upper()}")
        print(f"  PASS: {c['PASS']}")
        print(f"  FAIL: {c['FAIL']}")
        print(f"  SKIP: {c['SKIP']}")
        print(f"  Attempted: {attempted}")
        print(f"  Pass rate (over attempted): {rate(c['PASS'], attempted):.3f}")

    print("=" * 70)
    print("AGGREGATED BELIEF UPDATE VERIFICATION")
    print("=" * 70)
    print(f"n={n}, seed={seed}")

    summarize("play", metrics["play"])
    summarize("draw", metrics["draw"])
    summarize("draw_then_play", metrics["draw_then_play"])

    if extra["draw_eliminated_counts"]:
        avg = sum(extra["draw_eliminated_counts"]) / len(extra["draw_eliminated_counts"])
        print(f"\nDRAW: avg eliminated-card-instances (PASS only): {avg:.2f}")

    print("\nReport-friendly line (copy/paste):")
    for k in ["play", "draw", "draw_then_play"]:
        c = metrics[k]
        attempted = c["PASS"] + c["FAIL"]
        print(f"  {k}: {c['PASS']}/{attempted} pass (skipped {c['SKIP']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="iterations per scenario type")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = ap.parse_args()
    run(args.n, args.seed)