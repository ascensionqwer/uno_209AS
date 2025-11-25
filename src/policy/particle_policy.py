"""
Particle Policy: Runtime decision-making using dynamic particle filter + MCTS.

Uses particle filtering for belief approximation and MCTS for action selection.
Generates particles dynamically at runtime with caching.

**Player 1 Perspective:**
This module implements decision-making from Player 1's perspective:
- Player 1 observes game state (H_1, |H_2|, |D_g|, P, P_t, G_o)
- Player 1 makes decisions using get_action()
- System dynamics handle Player 2's turn
- Player 1 observes new state and updates particles based on Player 2's actions
- Particles represent beliefs about Player 2's hidden hand (H_2)
"""

import random
import math
from typing import List, Optional, Tuple

from ..uno.cards import (
    Card,
    RED,
    YELLOW,
    GREEN,
    BLUE,
    BLACK,
)
from ..uno.state import Action
from .observation_utils import canonicalize_game_state
from .particle_cache import ParticleCache
from .particle import Particle


class MCTSNode:
    """Node in MCTS tree with particle-based belief."""

    def __init__(
        self,
        H_1: List[Card],
        particles: List[Particle],
        P: List[Card],
        P_t: Optional[Card],
        G_o: str,
        action: Optional[Action] = None,
        parent=None,
    ):
        self.H_1 = H_1
        self.particles = (
            particles  # Particle set representing belief over opponent's hand
        )
        self.P = P
        self.P_t = P_t
        self.G_o = G_o
        self.action = action  # Action that led to this node
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_value = 0.0
        self.untried_actions = []

    @property
    def value(self) -> float:
        """Average value of this node."""
        return self.total_value / (self.visits + 1e-6)

    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried."""
        return len(self.untried_actions) == 0 and len(self.children) > 0

    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return self.G_o == "GameOver" or len(self.H_1) == 0

    def best_child(self, c: float = 1.414) -> "MCTSNode":
        """Select best child using UCB1 formula."""
        if len(self.children) == 0:
            return None
        return max(
            self.children,
            key=lambda child: child.value
            + c * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6)),
        )


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


def update_particles_opponent_play(
    particles: List[Particle], played_card: Card, P_t: Optional[Card]
) -> List[Particle]:
    """
    Update particles when opponent plays a card.
    Case 1: Opponent plays card z
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
    Case 2: Opponent draws card
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


def update_particles_after_action(
    particles: List[Particle],
    action: Action,
    H_1: List[Card],
    P: List[Card],
    P_t: Optional[Card],
) -> Tuple[List[Card], List[Card], Optional[Card], List[Particle]]:
    """
    Update particles after taking an action.
    Returns (H_1_new, P_new, P_t_new, particles_new).
    """
    if action.is_play():
        card = action.X_1
        H_1_new = [c for c in H_1 if c != card]
        P_new = P + [card]
        P_t_new = card
        if card[0] == BLACK:
            chosen_color = action.wild_color if action.wild_color else RED
            P_t_new = (chosen_color, card[1])
        # Particles unchanged (opponent hasn't acted yet)
        particles_new = particles
    else:
        # Draw action - update particles by sampling drawn card
        # For draw action, H_1_new is the same for all particles (we draw one card)
        particles_new = []
        if len(particles) > 0 and len(particles[0].D_g) > 0:
            # Sample a card from first particle's deck (all particles have same deck structure)
            drawn = random.choice(particles[0].D_g)
            H_1_new = H_1 + [drawn]
            for particle in particles:
                if len(particle.D_g) > 0 and drawn in particle.D_g:
                    D_g_new = [c for c in particle.D_g if c != drawn]
                    particles_new.append(
                        Particle(particle.H_2, D_g_new, particle.weight)
                    )
                else:
                    particles_new.append(particle)
        else:
            H_1_new = H_1
            particles_new = particles
        P_new = P
        P_t_new = P_t

    return H_1_new, P_new, P_t_new, particles_new


def mcts_search(
    H_1: List[Card],
    particles: List[Particle],
    P: List[Card],
    P_t: Optional[Card],
    G_o: str,
    gamma: float,
    num_iterations: int = 1000,
    max_depth: int = 5,
    ucb_c: float = 1.414,
    rollout_particle_sample_size: int = 10,
) -> Action:
    """
    Monte Carlo Tree Search with particle filter.
    Implements full MCTS: Selection -> Expansion -> Simulation -> Backpropagation.
    Returns best action.
    """
    # Get legal actions
    current_color = P_t[0] if P_t and P_t[0] != BLACK else None
    legal_actions = get_legal_actions(H_1, P_t, current_color)

    if len(legal_actions) == 0:
        return Action(n=1)

    if len(legal_actions) == 1:
        return legal_actions[0]

    # Initialize root node
    root = MCTSNode(H_1, particles, P, P_t, G_o)
    root.untried_actions = legal_actions.copy()

    # MCTS iterations
    for _ in range(num_iterations):
        # Selection: traverse tree using UCB1
        node = root
        depth = 0

        while not node.is_terminal() and node.is_fully_expanded() and depth < max_depth:
            node = node.best_child(ucb_c)
            depth += 1

        # Expansion: add new node for unexplored action
        if not node.is_terminal() and len(node.untried_actions) > 0:
            action = node.untried_actions.pop()
            H_1_new, P_new, P_t_new, particles_new = update_particles_after_action(
                node.particles, action, node.H_1, node.P, node.P_t
            )

            # Check if game over
            G_o_new = "GameOver" if len(H_1_new) == 0 else "Active"

            child = MCTSNode(
                H_1_new, particles_new, P_new, P_t_new, G_o_new, action, node
            )
            current_color_new = P_t_new[0] if P_t_new and P_t_new[0] != BLACK else None
            child.untried_actions = get_legal_actions(
                H_1_new, P_t_new, current_color_new
            )
            node.children.append(child)
            node = child
            depth += 1

        # Simulation: roll out using particle samples to estimate value
        if node.is_terminal():
            value = compute_reward(
                node.H_1, node.particles[0].H_2 if node.particles else []
            )
        else:
            # Use particle-based rollout
            total_value = 0.0
            total_weight = 0.0
            particle_subset = node.particles[
                : min(rollout_particle_sample_size, len(node.particles))
            ]
            for particle in particle_subset:
                rollout_value = simulate_rollout(
                    node.H_1,
                    particle.H_2,
                    particle.D_g,
                    node.P,
                    node.P_t,
                    node.G_o,
                    gamma,
                    depth,
                    max_depth,
                )
                total_value += particle.weight * rollout_value
                total_weight += particle.weight
            value = total_value / (total_weight + 1e-6)

        # Backpropagation: update node values
        while node is not None:
            node.visits += 1
            node.total_value += value
            value = gamma * value  # Discount for parent nodes
            node = node.parent

    # Return best action (most visited child)
    if len(root.children) == 0:
        # Fallback: return action with highest value
        return max(
            legal_actions,
            key=lambda a: simulate_rollout(
                H_1,
                particles[0].H_2 if particles else [],
                particles[0].D_g if particles else [],
                P,
                P_t,
                G_o,
                gamma,
                0,
                max_depth,
            )
            if particles
            else 0.0,
        )

    best_child = max(root.children, key=lambda c: c.visits)
    return best_child.action


class ParticlePolicy:
    """
    Runtime particle filter policy using MCTS for decision-making.
    Generates particles dynamically at runtime with caching.

    **Perspective: Player 1**
    This policy operates from Player 1's perspective:
    1. Player 1 observes game state (H_1, |H_2|, |D_g|, P, P_t, G_o)
    2. Player 1 makes a move using get_action()
    3. System dynamics occur (including Player 2's turn)
    4. Player 1 observes new game state
    5. Update particles based on observations of Player 2's actions
    6. Repeat

    Particles represent beliefs about Player 2's hidden hand (H_2).
    """

    def __init__(
        self,
        num_particles: int = 1000,
        mcts_iterations: int = 1000,
        planning_horizon: int = 5,
        gamma: float = 0.95,
        ucb_c: float = 1.414,
        rollout_particle_sample_size: int = 10,
        resample_threshold: float = 0.5,
    ):
        """
        Initialize particle policy (Player 1's perspective).

        Args:
            num_particles: Number of particles for belief approximation over Player 2's hand
            mcts_iterations: Number of MCTS iterations per decision
            planning_horizon: Maximum lookahead depth for rollouts
            gamma: Discount factor
            ucb_c: UCB1 exploration constant
            rollout_particle_sample_size: Number of particles to sample for rollouts
            resample_threshold: ESS threshold for resampling (fraction of num_particles)
        """
        self.num_particles = num_particles
        self.mcts_iterations = mcts_iterations
        self.planning_horizon = planning_horizon
        self.gamma = gamma
        self.ucb_c = ucb_c
        self.rollout_particle_sample_size = rollout_particle_sample_size
        self.resample_threshold = resample_threshold
        self.cache = ParticleCache()
        self.action_history = []  # Track action history for cache keys

    def get_action(
        self,
        H_1: List[Card],
        opponent_size: int,
        deck_size: int,
        P: List[Card],
        P_t: Optional[Card],
        G_o: str = "Active",
    ) -> Action:
        """
        Get optimal action for Player 1 from current game state using particle filter + MCTS.

        This is called when it's Player 1's turn. After this action is executed,
        system dynamics will handle Player 2's turn, and you should call update_after_opponent_action()
        when you observe the new game state.

        Args:
            H_1: Player 1's hand (observable)
            opponent_size: Size of Player 2's hand (observable, but not the actual cards)
            deck_size: Size of deck (will be auto-corrected to |L| - opponent_size)
            P: Played cards (observable)
            P_t: Top card of pile (observable)
            G_o: Game over status

        Returns:
            Optimal action for Player 1
        """
        # Create game state key for caching
        game_state_key = canonicalize_game_state(
            H_1, opponent_size, deck_size, P, P_t, G_o, self.action_history
        )

        # Get particles from cache (generates if not cached)
        particles = self.cache.get_particles(
            game_state_key, H_1, opponent_size, deck_size, P, self.num_particles
        )

        if len(particles) == 0:
            # Fallback: return draw action
            return Action(n=1)

        # Resample if needed
        particles = resample_particles(particles, self.num_particles)

        # Run MCTS to find best action
        best_action = mcts_search(
            H_1,
            particles,
            P,
            P_t,
            G_o,
            self.gamma,
            self.mcts_iterations,
            self.planning_horizon,
            self.ucb_c,
            self.rollout_particle_sample_size,
        )

        return best_action

    def update_after_action(self, action: Action):
        """
        Update action history after Player 1 takes an action.
        Call this after executing the action returned by get_action().
        """
        self.action_history.append(action)
        # Limit history size to prevent unbounded growth
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-100:]

    def update_after_opponent_action(
        self,
        particles: List[Particle],
        new_opponent_size: int,
        new_deck_size: int,
        new_P: List[Card],
        new_P_t: Optional[Card],
        opponent_played_card: Optional[Card] = None,
        opponent_drew: bool = False,
    ) -> List[Particle]:
        """
        Update particles after observing Player 2's action (system dynamics).

        Call this when you observe the new game state after Player 2's turn.
        This updates the particle filter based on what Player 2 did.

        Args:
            particles: Current particle set (from previous state)
            new_opponent_size: New size of Player 2's hand after their action
            new_deck_size: New size of deck after their action
            new_P: New played cards pile
            new_P_t: New top card
            opponent_played_card: Card Player 2 played (if they played), None otherwise
            opponent_drew: True if Player 2 drew a card (no legal move)

        Returns:
            Updated particle set
        """
        if opponent_played_card is not None:
            # Case 1: Player 2 played a card
            updated = update_particles_opponent_play(
                particles, opponent_played_card, new_P_t
            )
        elif opponent_drew:
            # Case 2: Player 2 drew a card (no legal move)
            updated = update_particles_opponent_draw(particles, new_P_t)
        else:
            # Case 3: Player 2's turn was skipped or other case
            # Particles remain unchanged
            updated = particles

        # Resample if needed
        updated = resample_particles(updated, self.num_particles)

        return updated

    def reset(self):
        """Reset policy state (clear action history)."""
        self.action_history = []

    def clear_cache(self):
        """Clear particle cache."""
        self.cache.clear()
