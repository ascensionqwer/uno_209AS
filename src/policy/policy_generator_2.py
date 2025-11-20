"""
Policy Generator 2: Based on MATH2.md particle filter + MCTS approach.

Uses particle filtering for belief approximation and MCTS for action selection.
Generates lookup table mapping observations to optimal actions.
"""

import random
import math
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
from .policy_generator_1 import is_legal_play, get_legal_actions, compute_reward


class Particle:
    """Represents a particle in the particle filter."""

    def __init__(self, H_2: List[Card], D_g: List[Card], weight: float = 1.0):
        self.H_2 = H_2
        self.D_g = D_g
        self.weight = weight


class MCTSNode:
    """Node in MCTS tree."""

    def __init__(self, state: Tuple, action: Optional[Action] = None, parent=None):
        self.state = state  # (H_1, H_2, D_g, P, P_t, G_o)
        self.action = action
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = []

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0 and len(self.children) > 0

    def best_child(self, c: float = 1.414) -> "MCTSNode":
        """Select best child using UCB1."""
        return max(
            self.children,
            key=lambda child: child.value / (child.visits + 1e-6)
            + c * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6)),
        )


def initialize_particles(
    H_1: List[Card],
    opponent_size: int,
    deck_size: int,
    P: List[Card],
    num_particles: int,
) -> List[Particle]:
    """
    Initialize particle set for belief approximation.
    P = {(s^(i), w^(i))} where we sample H_2^(i) and D_g^(i).
    """
    full_deck = build_full_deck()
    L = [c for c in full_deck if c not in H_1 and c not in P]

    particles = []
    for _ in range(num_particles):
        if len(L) < opponent_size + deck_size:
            continue

        # Sample opponent hand uniformly
        if opponent_size > len(L):
            continue
        H_2 = random.sample(L, opponent_size)
        remaining = [c for c in L if c not in H_2]

        # Sample deck
        if deck_size > len(remaining):
            D_g = remaining
        else:
            D_g = random.sample(remaining, deck_size)

        particles.append(Particle(H_2, D_g, 1.0 / num_particles))

    return particles


def update_particles_opponent_play(
    particles: List[Particle], played_card: Card, P_t: Optional[Card]
) -> List[Particle]:
    """
    Update particles when opponent plays a card.
    Case 1 from MATH2.md: Opponent plays card z
    """
    updated = []
    for particle in particles:
        if played_card in particle.H_2:
            H_2_new = [c for c in particle.H_2 if c != played_card]
            updated.append(Particle(H_2_new, particle.D_g, particle.weight))

    # Normalize weights
    total_weight = sum(p.weight for p in updated)
    if total_weight > 0:
        for p in updated:
            p.weight /= total_weight

    return updated


def update_particles_opponent_draw(
    particles: List[Particle], P_t: Optional[Card]
) -> List[Particle]:
    """
    Update particles when opponent draws (no legal move).
    Case 2 from MATH2.md: Opponent draws card
    """
    updated = []
    for particle in particles:
        # Check if opponent has no legal cards
        legal_cards = [c for c in particle.H_2 if is_legal_play(c, P_t)]
        if len(legal_cards) == 0 and len(particle.D_g) > 0:
            # Sample a card from deck
            drawn = random.choice(particle.D_g)
            H_2_new = particle.H_2 + [drawn]
            D_g_new = [c for c in particle.D_g if c != drawn]
            # Weight by likelihood
            weight = particle.weight * (1.0 / len(particle.D_g))
            updated.append(Particle(H_2_new, D_g_new, weight))

    # Normalize weights
    total_weight = sum(p.weight for p in updated)
    if total_weight > 0:
        for p in updated:
            p.weight /= total_weight

    return updated if updated else particles


def resample_particles(particles: List[Particle], num_particles: int) -> List[Particle]:
    """
    Resample particles using systematic resampling.
    Resample if effective sample size is too low.
    """
    if len(particles) == 0:
        return particles

    # Compute effective sample size
    weights = [p.weight for p in particles]
    ess = 1.0 / sum(w * w for w in weights)

    if ess < num_particles / 2:
        # Resample
        cumulative = []
        total = 0.0
        for w in weights:
            total += w
            cumulative.append(total)

        resampled = []
        step = 1.0 / num_particles
        r = random.uniform(0, step)
        i = 0
        for _ in range(num_particles):
            while r > cumulative[i]:
                i += 1
                if i >= len(particles):
                    i = len(particles) - 1
                    break
            resampled.append(
                Particle(
                    particles[i].H_2.copy(),
                    particles[i].D_g.copy(),
                    1.0 / num_particles,
                )
            )
            r += step

        return resampled

    return particles


def simulate_rollout(
    H_1: List[Card],
    H_2: List[Card],
    D_g: List[Card],
    P: List[Card],
    P_t: Optional[Card],
    G_o: str,
    gamma: float,
    depth: int = 0,
    max_depth: int = 5,
) -> float:
    """
    Simulate a random rollout to estimate value.
    """
    if G_o == "GameOver":
        return compute_reward(H_1, H_2)

    if depth >= max_depth:
        return (7 - len(H_1)) * 0.1 - (7 - len(H_2)) * 0.1

    # Player 1's turn
    current_color = P_t[0] if P_t and P_t[0] != BLACK else None
    legal_actions = get_legal_actions(H_1, P_t, current_color)

    if len(legal_actions) == 0:
        return 0.0

    # Random action
    action = random.choice(legal_actions)

    if action.is_play():
        card = action.X_1
        H_1_new = [c for c in H_1 if c != card]
        P_new = P + [card]
        P_t_new = card
        if card[0] == BLACK:
            chosen_color = (
                action.wild_color
                if action.wild_color
                else random.choice([RED, YELLOW, GREEN, BLUE])
            )
            P_t_new = (chosen_color, card[1])

        reward = compute_reward(H_1_new, H_2)

        # Opponent's turn (random)
        opp_legal = [
            c for c in H_2 if is_legal_play(c, P_t_new, P_t_new[0] if P_t_new else None)
        ]
        if len(opp_legal) > 0:
            opp_card = random.choice(opp_legal)
            H_2_new = [c for c in H_2 if c != opp_card]
            P_new_2 = P_new + [opp_card]
            P_t_new_2 = opp_card
            if opp_card[0] == BLACK:
                P_t_new_2 = (random.choice([RED, YELLOW, GREEN, BLUE]), opp_card[1])

            value = reward + gamma * simulate_rollout(
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
                drawn = random.choice(D_g)
                H_2_new = H_2 + [drawn]
                D_g_new = [c for c in D_g if c != drawn]
                value = reward + gamma * simulate_rollout(
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
            else:
                value = reward

        return value
    else:
        # Draw action
        if len(D_g) > 0:
            drawn = random.choice(D_g)
            H_1_new = H_1 + [drawn]
            D_g_new = [c for c in D_g if c != drawn]
            return gamma * simulate_rollout(
                H_1_new, H_2, D_g_new, P, P_t, "Active", gamma, depth + 1, max_depth
            )
        return 0.0


def mcts_search(
    H_1: List[Card],
    particles: List[Particle],
    P: List[Card],
    P_t: Optional[Card],
    G_o: str,
    gamma: float,
    num_iterations: int = 1000,
    max_depth: int = 5,
) -> Action:
    """
    Monte Carlo Tree Search with particle filter.
    Returns best action.
    """
    # Get legal actions
    current_color = P_t[0] if P_t and P_t[0] != BLACK else None
    legal_actions = get_legal_actions(H_1, P_t, current_color)

    if len(legal_actions) == 0:
        return Action(n=1)

    if len(legal_actions) == 1:
        return legal_actions[0]

    # Evaluate each action using particle-based value estimation
    action_values = {}

    for action in legal_actions:
        total_value = 0.0
        total_weight = 0.0

        # Sample particles for evaluation
        for particle in particles:
            if particle.weight < 1e-6:
                continue

            # Simulate action on this particle
            if action.is_play():
                card = action.X_1
                H_1_new = [c for c in H_1 if c != card]
                P_new = P + [card]
                P_t_new = card
                if card[0] == BLACK:
                    chosen_color = action.wild_color if action.wild_color else RED
                    P_t_new = (chosen_color, card[1])

                # Estimate value using rollout
                value = simulate_rollout(
                    H_1_new,
                    particle.H_2,
                    particle.D_g,
                    P_new,
                    P_t_new,
                    "Active",
                    gamma,
                    0,
                    max_depth,
                )

                total_value += particle.weight * value
                total_weight += particle.weight
            else:
                # Draw action
                if len(particle.D_g) > 0:
                    drawn = random.choice(particle.D_g)
                    H_1_new = H_1 + [drawn]
                    D_g_new = [c for c in particle.D_g if c != drawn]
                    value = simulate_rollout(
                        H_1_new,
                        particle.H_2,
                        D_g_new,
                        P,
                        P_t,
                        "Active",
                        gamma,
                        0,
                        max_depth,
                    )
                    total_value += particle.weight * value
                    total_weight += particle.weight

        if total_weight > 0:
            action_values[action] = total_value / total_weight
        else:
            action_values[action] = 0.0

    # Return best action
    best_action = max(action_values, key=action_values.get)
    return best_action


def generate_policy_2(
    num_particles: int = 1000,
    mcts_iterations: int = 1000,
    planning_horizon: int = 5,
    gamma: float = 0.95,
    num_observations: int = 1000,
) -> Dict[str, Dict[str, Any]]:
    """
    Generate policy lookup table using particle filter + MCTS approach.

    Args:
        num_particles: Number of particles for belief approximation
        mcts_iterations: Number of MCTS iterations (simplified to particle evaluation)
        planning_horizon: Maximum lookahead depth
        gamma: Discount factor
        num_observations: Number of observations to generate

    Returns:
        Dictionary mapping observation keys to {"action": ..., "value": ...}
    """
    policy = {}
    full_deck = build_full_deck()

    print(f"Generating Policy 2 with {num_observations} observations...")

    for i in range(num_observations):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{num_observations} observations...")

        # Generate random observation
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

        # Initialize particles
        particles = initialize_particles(
            H_1, opponent_size, deck_size, P, num_particles
        )

        if len(particles) == 0:
            continue

        # Resample if needed
        particles = resample_particles(particles, num_particles)

        # Run MCTS to find best action
        best_action = mcts_search(
            H_1, particles, P, P_t, "Active", gamma, mcts_iterations, planning_horizon
        )

        # Estimate value (simplified)
        current_color = P_t[0] if P_t and P_t[0] != BLACK else None
        legal_actions = get_legal_actions(H_1, P_t, current_color)
        if best_action in legal_actions:
            # Compute approximate value
            value = 0.0
            for particle in particles[: min(10, len(particles))]:  # Sample subset
                if best_action.is_play():
                    card = best_action.X_1
                    H_1_new = [c for c in H_1 if c != card]
                    value += particle.weight * simulate_rollout(
                        H_1_new,
                        particle.H_2,
                        particle.D_g,
                        P,
                        P_t,
                        "Active",
                        gamma,
                        0,
                        planning_horizon,
                    )
                else:
                    if len(particle.D_g) > 0:
                        H_1_new = H_1 + [random.choice(particle.D_g)]
                        value += particle.weight * simulate_rollout(
                            H_1_new,
                            particle.H_2,
                            particle.D_g,
                            P,
                            P_t,
                            "Active",
                            gamma,
                            0,
                            planning_horizon,
                        )

            # Canonicalize observation
            obs_key = canonicalize_observation(
                H_1, opponent_size, deck_size, P_t, "Active"
            )
            policy[obs_key] = {"action": serialize_action(best_action), "value": value}

    print(f"Generated {len(policy)} policy entries.")
    return policy
