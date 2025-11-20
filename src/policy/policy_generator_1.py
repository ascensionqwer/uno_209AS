"""
Policy Generator 1: Based on MATH.md exact value function approach.

Implements value iteration over belief states using sampling/approximation.
Generates lookup table mapping observations to optimal actions.
"""

import random
from typing import List, Dict, Any, Optional, Tuple

from ..uno.cards import (
    Card,
    build_full_deck,
    RED,
    YELLOW,
    GREEN,
    BLUE,
    BLACK,
)
from ..uno.state import Action
from .observation_utils import canonicalize_observation, serialize_action


def is_legal_play(
    card: Card, P_t: Optional[Card], current_color: Optional[str] = None
) -> bool:
    """
    Check if card can be legally played on top of P_t.
    LEGAL(P_t) = {c: c(COLOR) = P_t(COLOR) OR c(VALUE) = P_t(VALUE) OR c(COLOR) = BLACK}
    """
    if P_t is None:
        return False

    card_color, card_value = card
    P_t_color, P_t_value = P_t

    # Wild cards are always legal
    if card_color == BLACK:
        return True

    # Use current_color for Wild cards that were played
    effective_color = current_color if P_t_color == BLACK else P_t_color

    # Match by color
    if card_color == effective_color:
        return True

    # Match by value
    if card_value == P_t_value:
        return True

    return False


def get_legal_actions(
    H_1: List[Card], P_t: Optional[Card], current_color: Optional[str] = None
) -> List[Action]:
    """Get all legal actions for player 1 given hand and top card."""
    legal_actions = []

    if P_t is None:
        return []

    # Check for playable cards
    for card in H_1:
        if is_legal_play(card, P_t, current_color):
            if card[0] == BLACK:  # Wild card - need to choose color
                # Add action for each possible color choice
                for color in [RED, YELLOW, GREEN, BLUE]:
                    legal_actions.append(Action(X_1=card, wild_color=color))
            else:
                legal_actions.append(Action(X_1=card))

    # If no legal plays, must draw 1
    if len(legal_actions) == 0:
        legal_actions.append(Action(n=1))

    return legal_actions


def compute_reward(H_1_prime: List[Card], H_2_prime: List[Card]) -> float:
    """
    R(s', s, a) = +1 if |H_1'| = 0 (player 1 wins)
                 -1 if |H_2'| = 0 (player 2 wins)
                  0 otherwise
    """
    if len(H_1_prime) == 0:
        return 1.0
    if len(H_2_prime) == 0:
        return -1.0
    return 0.0


def sample_belief_states(
    H_1: List[Card],
    opponent_size: int,
    deck_size: int,
    P: List[Card],
    P_t: Optional[Card],
    num_samples: int = 100,
) -> List[Tuple[List[Card], List[Card], List[Card]]]:
    """
    Sample possible states (H_1, H_2, D_g) given observation.
    Returns list of (H_1, H_2, D_g) tuples.
    """
    full_deck = build_full_deck()
    L = [c for c in full_deck if c not in H_1 and c not in P]

    if len(L) < opponent_size + deck_size:
        return []

    samples = []
    for _ in range(num_samples):
        # Sample opponent hand
        if opponent_size > len(L):
            continue
        H_2 = random.sample(L, opponent_size)
        remaining = [c for c in L if c not in H_2]

        # Sample deck
        if deck_size > len(remaining):
            continue
        D_g = random.sample(remaining, min(deck_size, len(remaining)))

        samples.append((H_1.copy(), H_2, D_g))

    return samples


def approximate_value_function(
    H_1: List[Card],
    H_2: List[Card],
    D_g: List[Card],
    P: List[Card],
    P_t: Optional[Card],
    G_o: str,
    gamma: float,
    depth: int = 0,
    max_depth: int = 3,
) -> float:
    """
    Approximate value function V*(b) using recursive computation.
    Simplified version that looks ahead max_depth steps.
    """
    if G_o == "GameOver":
        return compute_reward(H_1, H_2)

    if depth >= max_depth:
        # Terminal value: prefer fewer cards in hand
        return (7 - len(H_1)) * 0.1 - (7 - len(H_2)) * 0.1

    # Get legal actions
    current_color = P_t[0] if P_t and P_t[0] != BLACK else None
    legal_actions = get_legal_actions(H_1, P_t, current_color)

    if len(legal_actions) == 0:
        return 0.0

    best_value = float("-inf")

    for action in legal_actions:
        # Simulate action
        if action.is_play():
            card = action.X_1
            H_1_new = [c for c in H_1 if c != card]
            P_new = P + [card]
            P_t_new = card
            if card[0] == BLACK:
                chosen_color = action.wild_color if action.wild_color else RED
                P_t_new = (chosen_color, card[1])

            # Compute reward
            reward = compute_reward(H_1_new, H_2)

            # Opponent's turn (simplified: assume opponent plays randomly)
            # For approximation, use expected value over opponent actions
            opponent_value = 0.0
            if len(H_2) > 0:
                # Sample a few opponent actions
                opponent_legal = [
                    c
                    for c in H_2
                    if is_legal_play(c, P_t_new, P_t_new[0] if P_t_new else None)
                ]
                if len(opponent_legal) > 0:
                    # Opponent plays a random legal card
                    opp_card = random.choice(opponent_legal)
                    H_2_new = [c for c in H_2 if c != opp_card]
                    P_new_2 = P_new + [opp_card]
                    P_t_new_2 = opp_card
                    if opp_card[0] == BLACK:
                        P_t_new_2 = (
                            random.choice([RED, YELLOW, GREEN, BLUE]),
                            opp_card[1],
                        )

                    opponent_value = approximate_value_function(
                        H_1_new,
                        H_2_new,
                        D_g,
                        P_new_2,
                        P_t_new_2,
                        "Active",
                        gamma,
                        depth + 1,
                        max_depth,
                    )
                else:
                    # Opponent draws
                    if len(D_g) > 0:
                        drawn = D_g[0]
                        H_2_new = H_2 + [drawn]
                        D_g_new = D_g[1:]
                        opponent_value = approximate_value_function(
                            H_1_new,
                            H_2_new,
                            D_g_new,
                            P_new,
                            P_t_new,
                            "Active",
                            gamma,
                            depth + 1,
                            max_depth,
                        )

            value = reward + gamma * opponent_value
            best_value = max(best_value, value)
        else:
            # Draw action
            if len(D_g) > 0:
                drawn = D_g[0]
                H_1_new = H_1 + [drawn]
                D_g_new = D_g[1:]
                value = approximate_value_function(
                    H_1_new, H_2, D_g_new, P, P_t, "Active", gamma, depth + 1, max_depth
                )
                best_value = max(best_value, value)

    return best_value if best_value != float("-inf") else 0.0


def generate_policy_1(
    gamma: float = 0.95,
    num_observations: int = 1000,
    num_belief_samples: int = 50,
    max_depth: int = 3,
) -> Dict[str, Dict[str, Any]]:
    """
    Generate policy lookup table using value function approach.

    Args:
        gamma: Discount factor
        num_observations: Number of observations to generate
        num_belief_samples: Number of belief state samples per observation
        max_depth: Maximum lookahead depth

    Returns:
        Dictionary mapping observation keys to {"action": ..., "value": ...}
    """
    policy = {}
    full_deck = build_full_deck()

    print(f"Generating Policy 1 with {num_observations} observations...")

    for i in range(num_observations):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{num_observations} observations...")

        # Generate random observation
        # Sample hand size (1-10 cards)
        hand_size = random.randint(1, 10)
        opponent_size = random.randint(1, 15)
        deck_size = random.randint(10, 50)

        # Sample hand
        H_1 = random.sample(full_deck, min(hand_size, len(full_deck)))
        remaining = [c for c in full_deck if c not in H_1]

        # Sample pile and top card
        if len(remaining) > 0:
            P_t = random.choice(remaining)
            P = [P_t]
        else:
            P_t = None
            P = []

        # Get legal actions
        current_color = P_t[0] if P_t and P_t[0] != BLACK else None
        legal_actions = get_legal_actions(H_1, P_t, current_color)

        if len(legal_actions) == 0:
            continue

        # Sample belief states
        samples = sample_belief_states(
            H_1, opponent_size, deck_size, P, P_t, num_belief_samples
        )

        if len(samples) == 0:
            continue

        # Evaluate each action
        best_action = None
        best_value = float("-inf")

        for action in legal_actions:
            total_value = 0.0
            count = 0

            for H_1_sample, H_2_sample, D_g_sample in samples:
                try:
                    value = approximate_value_function(
                        H_1_sample,
                        H_2_sample,
                        D_g_sample,
                        P,
                        P_t,
                        "Active",
                        gamma,
                        0,
                        max_depth,
                    )
                    total_value += value
                    count += 1
                except Exception:
                    continue

            if count > 0:
                avg_value = total_value / count
                if avg_value > best_value:
                    best_value = avg_value
                    best_action = action

        if best_action is not None:
            # Canonicalize observation
            obs_key = canonicalize_observation(
                H_1, opponent_size, deck_size, P_t, "Active"
            )
            policy[obs_key] = {
                "action": serialize_action(best_action),
                "value": best_value,
            }

    print(f"Generated {len(policy)} policy entries.")
    return policy
