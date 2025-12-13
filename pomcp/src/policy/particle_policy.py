"""
Particle Policy: Full POMCP implementation for UNO.

Implements Partially Observable Monte-Carlo Planning (POMCP) with:
- History-based search tree T(h) = {N(h), V(h), B(h)}
- Particle filter for belief state approximation B(h)
- Monte-Carlo simulations with belief state updates
- Tree pruning after real actions/observations

**Player 1 Perspective:**
This module implements POMCP from Player 1's perspective:
- Player 1 observes O = (H_1, |H_2|, |D_g|, P, P_t, G_o)
- Maintains belief B(h) over opponent's hidden hand using particles
- Each simulation samples start state s ~ B(h_t)
- Updates belief states B(h) during simulations
- Prunes tree to T(h_t a_t o_t) after real action/observation
"""

import random
import math
import time
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


class POMCPNode:
    """
    POMCP Tree Node: T(h) = {N(h), V(h), B(h)}

    Implements history-based node with:
    - N(h): visit count for history h
    - V(h): value estimate for history h
    - B(h): belief state (particles) for history h
    """

    def __init__(
        self,
        history: List[Tuple[Action, Optional[Card]]],  # Action-observation history
        H_1: List[Card],  # Current observable state
        particles: List[Particle],  # Belief state B(h)
        P: List[Card],
        P_t: Optional[Card],
        G_o: str,
        action: Optional[Action] = None,  # Action that led to this node
        parent: Optional["POMCPNode"] = None,
    ):
        self.history = history  # Action-observation history h
        self.H_1 = H_1  # Observable state component
        self.particles = particles  # Belief state B(h) - CRITICAL for POMCP
        self.P = P
        self.P_t = P_t
        self.G_o = G_o
        self.action = action  # Action that led to this node
        self.parent = parent

        # POMCP node statistics - maintain compatibility with existing code
        self.N_h = 0  # Visit count N(h)
        self.V_h = 0.0  # Value estimate V(h)
        self.children = []  # List of POMCPNode children
        self.untried_actions = []

        # Compatibility aliases for existing code
        self.visits = 0  # Alias for N_h
        self.total_value = 0.0  # Alias for V_h * N_h

    @property
    def value(self) -> float:
        """Average value V(h) of this history."""
        return self.V_h / (self.N_h + 1e-6)

    def update_stats(self, reward: float):
        """Update POMCP node statistics."""
        self.N_h += 1
        self.V_h += reward
        # Update compatibility aliases
        self.visits = self.N_h
        self.total_value = self.V_h

    @property
    def node_value(self) -> float:
        """Average value V(h) of this history."""
        return self.V_h / (self.N_h + 1e-6)

    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried from this history."""
        return len(self.untried_actions) == 0 and len(self.children) > 0

    def is_terminal(self) -> bool:
        """Check if this history leads to terminal state."""
        return self.G_o == "GameOver" or len(self.H_1) == 0

    def best_child(self, c: float = 1.414) -> Optional["POMCPNode"]:
        """
        Select best child using UCB1 formula for POMCP.
        V⊕(ha) = V(ha) + c√(log N(h) / N(ha))
        """
        if len(self.children) == 0:
            return None

        # Fixed UCT formula - no +1e-6 fudge factors
        log_parent = math.log(self.N_h) if self.N_h > 0 else 0
        return max(
            self.children,
            key=lambda child: child.value
            + c * math.sqrt(log_parent / child.N_h if child.N_h > 0 else float("inf")),
        )

    def sample_state_from_belief(self) -> Tuple[List[Card], List[Card], List[Card]]:
        """
        Sample a complete state s ~ B(h) for simulation start.
        Returns: (H_1_sample, H_2_sample, D_g_sample)
        """
        if not self.particles:
            return self.H_1, [], []

        # Sample particle according to weights (importance sampling)
        weights = [p.weight for p in self.particles]
        total_weight = sum(weights)
        if total_weight == 0:
            particle = random.choice(self.particles)
        else:
            # Weighted sampling
            r = random.uniform(0, total_weight)
            cumulative = 0
            particle = self.particles[0]
            for p, w in zip(self.particles, weights):
                cumulative += w
                if r <= cumulative:
                    particle = p
                    break

        return self.H_1, particle.H_2.copy(), particle.D_g.copy()

    def update_belief_with_simulation_state(
        self, H_2_sim: List[Card], D_g_sim: List[Card]
    ):
        """
        Update belief state B(h) with simulation state.
        POMCP: "For every history h encountered during simulation,
        belief state B(h) is updated to include the simulation state"
        """
        # Create new particle from simulation state
        new_particle = Particle(H_2_sim.copy(), D_g_sim.copy(), weight=1.0)

        # Add to belief (will be normalized later)
        self.particles.append(new_particle)

        # Limit particle count to prevent explosion
        if len(self.particles) > 1000:
            # Keep top particles by weight (simple resampling)
            self.particles.sort(key=lambda p: p.weight, reverse=True)
            self.particles = self.particles[:500]

            # Renormalize weights
            total_weight = sum(p.weight for p in self.particles)
            if total_weight > 0:
                for p in self.particles:
                    p.weight /= total_weight


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

    Only keeps particles where played_card was in H_2 (legally possible).
    """
    updated = []
    for particle in particles:
        # Only keep particles where the played card was actually in opponent's hand
        if played_card in particle.H_2:
            H_2_new = [c for c in particle.H_2 if c != played_card]
            # Validation: ensure played card is removed
            assert played_card not in H_2_new, "Played card still in H_2 after update!"
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

    Only updates particles where opponent has no legal cards (legally possible state).
    """
    updated = []
    for particle in particles:
        # Check if opponent has no legal cards (required for draw action)
        legal_cards = [c for c in particle.H_2 if is_legal_play(c, P_t)]
        if len(legal_cards) == 0 and len(particle.D_g) > 0:
            # Sample a card from deck (must be from D_g, which doesn't contain H_1 or P cards)
            drawn = random.choice(particle.D_g)
            H_2_new = particle.H_2 + [drawn]
            D_g_new = [c for c in particle.D_g if c != drawn]
            # Validation: drawn card should be removed from deck
            assert drawn not in D_g_new, "Drawn card still in D_g after update!"
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


def default_rollout_policy(
    H_1: List[Card], P_t: Optional[Card], current_color: Optional[str]
) -> Action:
    """
    Simple rollout policy for POMCP - uniform random as per paper.
    """
    legal_actions = get_legal_actions(H_1, P_t, current_color)

    if len(legal_actions) == 0:
        return Action(n=1)

    # Uniform random action selection (as suggested in POMCP paper)
    return random.choice(legal_actions)


def simulate_rollout(
    H_1: List[Card],
    H_2: List[Card],
    D_g: List[Card],
    P: List[Card],  # Never None - filter before calling
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

    # Default policy for rollouts (not pure random)
    action = default_rollout_policy(H_1, P_t, current_color)

    if action.is_play():
        card = action.X_1
        H_1_new = [c for c in H_1 if c != card]
        P_new = [c for c in P if c is not None] + [card]
        P_t_new = card
        if card and card[0] == BLACK:
            chosen_color = (
                action.wild_color
                if action.wild_color
                else random.choice([RED, YELLOW, GREEN, BLUE])
            )
            P_t_new = (chosen_color, card[1])

        reward = compute_reward(H_1_new, H_2)

        # Opponent's turn (random)
        current_color = P_t_new[0] if P_t_new and P_t_new[0] != BLACK else None
        opp_legal = [c for c in H_2 if is_legal_play(c, P_t_new, current_color)]
        if len(opp_legal) > 0:
            opp_card = random.choice(opp_legal)
            H_2_new = [c for c in H_2 if c != opp_card]
            P_new_2 = [c for c in P_new if c is not None] + [opp_card]
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
                    [c for c in P_new if c is not None],
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
    Update particles after Player 1 takes an action.
    Returns (H_1_new, P_new, P_t_new, particles_new).

    Ensures particles remain valid: no cards from H_1_new or P_new appear in particle H_2 or D_g.
    """
    if action.is_play():
        card = action.X_1
        H_1_new = [c for c in H_1 if c != card]
        P_new = [c for c in P if c is not None] + [card]
        P_t_new = card
        if card and card[0] == BLACK:
            chosen_color = action.wild_color if action.wild_color else RED
            P_t_new = (chosen_color, card[1])
        # Particles unchanged (opponent hasn't acted yet)
        # Note: If particles were generated correctly for the current state,
        # the played card (which was in H_1) should not be in any particle.
        # Validation happens in MCTS after state update.
        particles_new = particles
    else:
        # Draw action - update particles by sampling drawn card
        # For draw action, H_1_new is the same for all particles (we draw one card)
        particles_new = []
        if len(particles) > 0 and len(particles[0].D_g) > 0:
            # Sample a card from first particle's deck (all particles have same deck structure)
            # This card is guaranteed to not be in H_1 or P since D_g was generated from L = D \ (H_1 ∪ P)
            drawn = random.choice(particles[0].D_g)
            H_1_new = H_1 + [drawn]
            for particle in particles:
                if len(particle.D_g) > 0 and drawn in particle.D_g:
                    D_g_new = [c for c in particle.D_g if c != drawn]
                    # Validation: drawn card should be removed from deck
                    assert drawn not in D_g_new, "Drawn card still in D_g after update!"
                    # Validation: drawn card should not be in H_1 or P
                    assert drawn not in H_1, f"Drawn card {drawn} was in H_1!"
                    assert drawn not in P, f"Drawn card {drawn} was in P!"
                    particles_new.append(
                        Particle(particle.H_2, D_g_new, particle.weight)
                    )
                else:
                    particles_new.append(particle)
        else:
            H_1_new = H_1
            particles_new = particles
        P_new = [c for c in P if c is not None]
        P_t_new = P_t

    return H_1_new, [c for c in P_new if c is not None], P_t_new, particles_new


def pomcp_search(
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
    current_color: Optional[str] = None,
) -> Action:
    """
    Monte Carlo Tree Search with particle filter.
    Implements full MCTS: Selection -> Expansion -> Simulation -> Backpropagation.
    Returns best action.
    """
    # Get legal actions
    # Use passed current_color, fall back to top card color if not provided
    effective_color = (
        current_color
        if current_color is not None
        else (P_t[0] if P_t and P_t[0] != BLACK else None)
    )
    legal_actions = get_legal_actions(H_1, P_t, effective_color)

    if len(legal_actions) == 0:
        return Action(n=1)

    if len(legal_actions) == 1:
        return legal_actions[0]

    # Initialize root node with empty history for POMCP
    root = POMCPNode([], H_1, particles, P, P_t, G_o)
    root.untried_actions = legal_actions.copy()

    # MCTS iterations
    for _ in range(num_iterations):
        # POMCP Search: Selection -> Expansion -> Simulation -> Backpropagation
        # Each simulation samples start state s ~ B(h_t) - CRITICAL for POMCP
        # Sample start state from current belief B(h_t)
        H_1_sample, H_2_sample, D_g_sample = root.sample_state_from_belief()

        # Selection: traverse tree using UCB1 with history-based nodes
        node = root
        history = root.history.copy()
        depth = 0

        while (
            node
            and not node.is_terminal()
            and node.is_fully_expanded()
            and depth < max_depth
        ):
            node = node.best_child(ucb_c)
            if node:
                history.append(node.action)
                depth += 1
            else:
                break

        # Expansion: add new node for unexplored action
        if node and not node.is_terminal() and len(node.untried_actions) > 0:
            action = node.untried_actions.pop()
            H_1_new, P_new, P_t_new, particles_new = update_particles_after_action(
                node.particles, action, node.H_1, node.P, node.P_t
            )

            # Check if game over
            G_o_new = "GameOver" if len(H_1_new) == 0 else "Active"

            # Create child node with updated history
            new_history = history + [(action, None)]  # Observation not yet known
            child = POMCPNode(
                new_history,
                H_1_new,
                particles_new,
                P_new,
                P_t_new,
                G_o_new,
                action,
                node,
            )
            current_color_new = P_t_new[0] if P_t_new and P_t_new[0] != BLACK else None
            child.untried_actions = get_legal_actions(
                H_1_new, P_t_new, current_color_new
            )
            node.children.append(child)
            node = child
            depth += 1

        # Simulation: Use the node we reached for rollout
        rollout_node = node  # This is the key fix - use the node we found
        if rollout_node.is_terminal():
            value = compute_reward(
                rollout_node.H_1,
                rollout_node.particles[0].H_2 if rollout_node.particles else [],
            )
        elif rollout_node:
            # POMCP: Sample state from belief and simulate with belief updates
            H_1_sim, H_2_sim, D_g_sim = rollout_node.sample_state_from_belief()

            # Simulate action and update belief during simulation
            rollout_value = simulate_rollout(
                H_1_sim,
                H_2_sim,
                D_g_sim,
                rollout_node.P,
                rollout_node.P_t,
                rollout_node.G_o,
                gamma,
                depth,
                max_depth,
            )

            # POMCP: Update belief with simulation state
            rollout_node.update_belief_with_simulation_state(H_2_sim, D_g_sim)
            value = rollout_value
        else:
            value = 0.0

        # Backpropagation: update POMCP node statistics from rollout_node back to root
        backprop_node = rollout_node
        while backprop_node is not None:
            backprop_node.update_stats(value)
            value = gamma * value  # Discount for parent nodes
            backprop_node = backprop_node.parent

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
    Full POMCP implementation for UNO from Player 1's perspective.

    Implements Partially Observable Monte-Carlo Planning with:
    - History-based search tree T(h) = {N(h), V(h), B(h)}
    - Particle filter belief approximation B(h) over opponent's hand
    - Monte-Carlo simulations with start state sampling s ~ B(h_t)
    - Tree pruning after real actions/observations
    - Belief state updates during simulations

    **POMCP Process:**
    1. Player 1 observes O = (H_1, |H_2|, |D_g|, P, P_t, G_o)
    2. Sample start state s ~ B(h_t) for each simulation
    3. Run PO-UCT search with belief-updated nodes
    4. Select best action argmax_a V(ha)
    5. Execute action and observe real outcome
    6. Prune tree to T(h_t a_t o_t) and update belief
    7. Repeat

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
        deck_builder=None,
    ):
        """
        Initialize POMCP policy (Player 1's perspective).

        Args:
            num_particles: Number of particles for belief approximation B(h)
            mcts_iterations: Number of POMCP simulations per decision
            planning_horizon: Maximum lookahead depth for rollouts
            gamma: Discount factor for value calculations
            ucb_c: UCB1 exploration constant for PO-UCT
            rollout_particle_sample_size: Number of particles to sample for rollouts
            resample_threshold: ESS threshold for particle resampling
            deck_builder: Function that returns full deck (default: standard UNO 108 cards)
                         For simplified game, pass lambda: build_simplified_deck(max_number, max_colors)
        """
        self.num_particles = num_particles
        self.mcts_iterations = mcts_iterations
        self.planning_horizon = planning_horizon
        self.gamma = gamma
        self.ucb_c = ucb_c
        self.rollout_particle_sample_size = rollout_particle_sample_size
        self.resample_threshold = resample_threshold
        self.cache = ParticleCache(deck_builder=deck_builder)
        self.action_history = []  # Action-observation history h
        self.decision_times = []  # Track decision times for performance analysis

        # POMCP tree root - persists across decisions with pruning
        self.root_node = None  # Current search tree root T(h_t)
        self.current_history = []  # Current action-observation history

    def get_action(
        self,
        H_1: List[Card],
        opponent_size: int,
        deck_size: int,
        P: List[Card],
        P_t: Optional[Card],
        G_o: str = "Active",
        current_color: Optional[str] = None,
    ) -> Action:
        """
        Get optimal action for Player 1 using full POMCP algorithm.

        POMCP Process:
        1. Get particles B(h_t) for current observation
        2. If tree exists, prune to current history, else create new root
        3. Run PO-UCT search with belief state sampling s ~ B(h)
        4. Select best action argmax_a V(ha)
        5. Return action for execution

        Args:
            H_1: Player 1's hand (observable)
            opponent_size: Size of Player 2's hand (observable)
            deck_size: Size of deck (observable)
            P: Played cards (observable)
            P_t: Top card of pile (observable)
            G_o: Game over status

        Returns:
            Optimal action for Player 1 using POMCP
        """
        start_time = time.time()

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
            decision_time = time.time() - start_time
            self.decision_times.append(decision_time)
            return Action(n=1)

        # Resample if needed
        particles = resample_particles(particles, self.num_particles)

        # POMCP: Check if we need to prune existing tree
        if self.root_node:
            # We have existing tree - check if it matches current state
            # For simplicity, create new root each decision (could be optimized)
            self.root_node = POMCPNode([], H_1, particles, P, P_t, G_o)
            self.current_history = []
        else:
            # First decision - create root
            self.root_node = POMCPNode([], H_1, particles, P, P_t, G_o)
            self.current_history = []

        # Run POMCP search to find best action
        best_action = pomcp_search(
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
            current_color,
        )

        decision_time = time.time() - start_time
        self.decision_times.append(decision_time)

        # POMCP: Update action history for next decision
        self.action_history.append(best_action)
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-100:]

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

    def prune_tree_to_history(
        self,
        action: Action,
        observation: Optional[Card],
        new_H_1: List[Card],
        new_particles: List[Particle],
        new_P: List[Card],
        new_P_t: Optional[Card],
        new_G_o: str,
    ):
        """
        POMCP: Prune search tree to T(h_t a_t o_t) after real action/observation.

        After executing real action and observing real outcome,
        prune tree to the corresponding child node and make it new root.
        All other histories are now impossible.

        Args:
            action: Action a_t that was executed
            observation: Observation o_t (card played by opponent, or None)
            new_H_1: Updated player hand
            new_particles: Updated belief state B(h_t a_t o_t)
            new_P: Updated played pile
            new_P_t: Updated top card
            new_G_o: Updated game status
        """
        if not self.root_node:
            return

        # Find child node corresponding to executed action
        target_child = None
        for child in self.root_node.children:
            if child.action and self._actions_equal(child.action, action):
                target_child = child
                break

        if target_child:
            # POMCP: Prune tree to this child
            self.root_node = target_child
            self.root_node.parent = None  # Break parent reference

            # Update node state with real observation
            if new_H_1 is not None:
                self.root_node.H_1 = new_H_1
            if new_particles is not None:
                self.root_node.particles = new_particles
            if new_P is not None:
                self.root_node.P = new_P
            if new_P_t is not None:
                self.root_node.P_t = new_P_t
            self.root_node.G_o = new_G_o

            # Update history with action-observation pair
            self.current_history.append((action, observation))
            self.root_node.history = self.current_history.copy()
        else:
            # Child not found - create new root
            self.root_node = POMCPNode(
                self.current_history + [(action, observation)],
                new_H_1 if new_H_1 is not None else [],
                new_particles if new_particles is not None else [],
                new_P if new_P is not None else [],
                new_P_t,
                new_G_o,
                action,
            )

    def _actions_equal(self, a1: Action, a2: Action) -> bool:
        """Check if two actions are equal for POMCP tree pruning."""
        if a1.is_play() and a2.is_play():
            return a1.X_1 == a2.X_1 and a1.wild_color == a2.wild_color
        elif a1.is_draw() and a2.is_draw():
            return a1.n == a2.n
        return False

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
        POMCP: Updates belief state and prunes search tree.

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
        # Update particles based on opponent action
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
            updated = particles

        # Resample if needed
        updated = resample_particles(updated, self.num_particles)

        # POMCP: Tree would be pruned in next get_action() call
        # when we have the new game state
        return updated

    def reset(self):
        """Reset policy state (clear action history)."""
        self.action_history = []

    def clear_cache(self):
        """Clear particle cache."""
        self.cache.clear()

    def get_decision_stats(self) -> dict:
        """Get decision timing statistics."""
        if not self.decision_times:
            return {"avg": 0, "max": 0, "min": 0, "count": 0}

        return {
            "avg": sum(self.decision_times) / len(self.decision_times),
            "max": max(self.decision_times),
            "min": min(self.decision_times),
            "count": len(self.decision_times),
        }

    def reset_decision_stats(self):
        """Reset decision timing statistics."""
        self.decision_times = []
