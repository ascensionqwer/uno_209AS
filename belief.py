from typing import List, Dict, Tuple, Set, Optional
import random
from collections import Counter
from itertools import combinations
import math
from cards import Card, RED, YELLOW, GREEN, BLUE
from pomdp import State
from uno import Uno


class Belief:
    """
    Belief state representation for UNO POMDP with proper Bayesian updates.

    Mathematical formulation:
    - L = D \ (H_1 ∪ P) - unknown card set
    - bel⁻(H_2) = 1/C(|L|, k) - prior belief (uniform over k-subsets)
    - Posterior updates based on opponent actions:
        * Case 1: Opponent plays card z
        * Case 2: Opponent draws (no legal cards in hand)

    O = (H_1, |H_2|, |D_g|, P, P_t, G_o) - Player 1's observation
    """

    def __init__(self, observation: Tuple):
        """
        Initialize belief from an observation.

        Args:
            observation: O = (H_1, |H_2|, |D_g|, P, P_t, G_o)
        """
        self.H_1 = observation[0]  # Player 1's hand (known)
        self.h2_size = observation[1]  # k = |H_2| (known)
        self.dg_size = observation[2]  # |D_g| (known)
        self.P = observation[3]  # Played cards (known)
        self.P_t = observation[4]  # Top card (known)
        self.G_o = observation[5]  # Game status (known)

        # L = D \ (H_1 ∪ P) - unknown cards
        self.L = self._compute_L()

        # N(P_t) = LEGAL(P_t) ∩ L - legal cards in unknown set
        self.N_Pt = self._compute_legal_unknown()

        # Valid H_2 configurations (for exact small state spaces)
        # For large spaces, we use sampling with proper distribution
        self.valid_h2_configs = None
        self.use_sampling = len(self.L) > 20  # Use sampling for large spaces

        # Posterior mode: None, "play", "draw"
        self.posterior_mode = None
        self.played_card = None  # Card played by opponent (Case 1)

    def _compute_L(self) -> List[Card]:
        """
        Computes L = D \ (H_1 ∪ P) - cards unknown to Player 1.

        Returns:
            List of cards that could be in H_2 or D_g
        """
        # Build full deck
        full_deck = []
        colors = [RED, YELLOW, GREEN, BLUE]
        for color in colors:
            full_deck.append((color, 0))
            for number in range(1, 10):
                full_deck.append((color, number))
                full_deck.append((color, number))

        # Count known cards: H_1 ∪ P
        known_cards = list(self.H_1) + list(self.P)
        known_counter = Counter(known_cards)

        # L = D \ (H_1 ∪ P)
        L = []
        full_counter = Counter(full_deck)

        for card, count in full_counter.items():
            unknown_count = count - known_counter.get(card, 0)
            L.extend([card] * unknown_count)

        return L

    def _compute_legal_unknown(self) -> Set[Card]:
        """
        Computes N(P_t) = LEGAL(P_t) ∩ L

        LEGAL(P_t) = {c ∈ D : c.COLOR = P_t.COLOR ∨ c.VALUE = P_t.VALUE}
        (Simplified: no wild cards in this implementation)

        Returns:
            Set of legal cards within unknown set L
        """
        if self.P_t is None:
            return set()

        P_t_color, P_t_value = self.P_t
        legal_unknown = set()

        for card in self.L:
            card_color, card_value = card
            if card_color == P_t_color or card_value == P_t_value:
                legal_unknown.add(card)

        return legal_unknown

    def reset_prior(self):
        """
        Resets to prior belief: bel⁻(H_2) = 1/C(|L|, k)
        Uniform distribution over all k-subsets of L.
        """
        self.posterior_mode = None
        self.played_card = None
        self.valid_h2_configs = None

    def update_opponent_played(self, played_card: Card):
        """
        Case 1: Opponent played card z.

        Update:
        - k' = k - 1 (opponent hand size decreases)
        - P' = P ∪ {z}
        - P_t' = z
        - L' = D \ (H_1 ∪ P')
        - z ∈ H_2 (condition: played card was in opponent hand)

        After update, reset to prior with new L and k.

        Args:
            played_card: Card that opponent played
        """
        # Verify card was in L (sanity check)
        if played_card not in self.L:
            print(f"Warning: Opponent played {played_card} not in L!")

        # Update game state knowledge
        self.h2_size -= 1  # k' = k - 1
        self.P = list(self.P) + [played_card]  # P' = P ∪ {z}
        self.P_t = played_card  # P_t' = z

        # Recompute L' and N(P_t')
        self.L = self._compute_L()
        self.N_Pt = self._compute_legal_unknown()

        # Reset to new prior with updated L and k
        self.reset_prior()

    def update_opponent_drew(self):
        """
        Case 2: Opponent drew a card (no legal play available).

        This observation tells us: H_2 ∩ N(P_t) = ∅
        (opponent had no legal cards before drawing)

        Posterior belief:
        - Before draw: bel⁺(H_2 | no_legal) = 1/C(|L|-|N(P_t)|, k)
          where H_2 ⊆ L \ N(P_t)

        - After draw: k' = k + 1, and we don't know which card was drawn
          bel⁻(H_2' | no_legal+draw) with |H_2'| = k+1

        Update:
        - k' = k + 1
        - |D_g|' = |D_g| - 1
        - L unchanged (we don't know what was drawn)
        - Valid H_2 configs must not contain any card from N(P_t)
        """
        # Record that we're in posterior mode (opponent had no legal cards)
        self.posterior_mode = "draw"

        # Update sizes
        self.h2_size += 1  # k' = k + 1
        self.dg_size -= 1  # |D_g|' = |D_g| - 1

        # L is unchanged, but valid H_2 must exclude N(P_t)
        # This constraint is applied during sampling

    def _prob_no_legal(self) -> float:
        """
        Computes Pr(no_legal) = C(|L|-|N(P_t)|, k) / C(|L|, k)

        Probability that opponent has no legal cards.
        """
        L_size = len(self.L)
        N_size = len(self.N_Pt)
        k = self.h2_size

        if k > L_size - N_size:
            return 0.0  # Impossible to have k cards without any legal ones

        try:
            numerator = math.comb(L_size - N_size, k)
            denominator = math.comb(L_size, k)
            return numerator / denominator if denominator > 0 else 0.0
        except:
            return 0.0

    def sample_state(self, seed: int = None) -> State:
        """
        Samples a possible state consistent with current belief.

        Sampling respects posterior constraints:
        - If posterior_mode = "draw": H_2 ⊆ L \ N(P_t)
        - Otherwise: H_2 ⊆ L (uniform prior)

        Args:
            seed: Random seed for reproducibility

        Returns:
            A possible State consistent with belief
        """
        rng = random.Random(seed)

        # Determine valid cards for H_2 based on belief state
        if self.posterior_mode == "draw":
            # Opponent had no legal cards: H_2 ⊆ L \ N(P_t)
            valid_for_h2 = [c for c in self.L if c not in self.N_Pt]
        else:
            # Prior belief: H_2 can be any k-subset of L
            valid_for_h2 = self.L.copy()

        # Check if sampling is possible
        if len(valid_for_h2) < self.h2_size:
            print(
                f"Warning: Cannot sample - need {self.h2_size} cards but only {len(valid_for_h2)} valid"
            )
            valid_for_h2 = self.L.copy()  # Fall back to prior

        # Sample H_2: randomly choose k cards from valid set
        rng.shuffle(valid_for_h2)
        H_2_sample = valid_for_h2[: self.h2_size]

        # Remaining cards go to D_g (up to observed size)
        remaining = [c for c in self.L if c not in H_2_sample]
        rng.shuffle(remaining)
        D_g_sample = remaining[: self.dg_size]

        # Construct state
        state = (
            list(self.H_1),  # H_1 (known)
            H_2_sample,  # H_2 (sampled from belief)
            D_g_sample,  # D_g (sampled)
            list(self.P),  # P (known)
            self.P_t,  # P_t (known)
            self.G_o,  # G_o (known)
        )

        return state

    def sample_states(self, n_samples: int = 100) -> List[State]:
        """
        Generates multiple state samples for belief approximation.

        Args:
            n_samples: Number of states to sample

        Returns:
            List of possible states from belief distribution
        """
        return [self.sample_state(seed=i) for i in range(n_samples)]

    def get_card_probabilities(self, location: str) -> Dict[Card, float]:
        """
        Estimates probability of each card being in a specific location.

        Uses proper belief distribution:
        - If posterior_mode = "draw": conditions on H_2 ⊆ L \ N(P_t)
        - Otherwise: uniform over k-subsets

        Args:
            location: Either "H_2" or "D_g"

        Returns:
            Dictionary mapping cards to probabilities
        """
        if location not in ["H_2", "D_g"]:
            raise ValueError("location must be 'H_2' or 'D_g'")

        # Determine valid cards based on belief
        if self.posterior_mode == "draw" and location == "H_2":
            valid_cards = [c for c in self.L if c not in self.N_Pt]
        else:
            valid_cards = self.L

        if len(valid_cards) == 0:
            return {}

        # Compute probabilities
        probabilities = {}

        if location == "H_2":
            # P(card in H_2) = k / |valid_cards| for uniform sampling
            for card in set(valid_cards):
                count = valid_cards.count(card)
                # Probability this card is in H_2
                probabilities[card] = (count * self.h2_size) / len(valid_cards)
        else:  # D_g
            # P(card in D_g) = |D_g| / |valid_cards|
            for card in set(valid_cards):
                count = valid_cards.count(card)
                probabilities[card] = (count * self.dg_size) / len(valid_cards)

        return probabilities

    def entropy(self) -> float:
        """
        Computes entropy of the belief state as measure of uncertainty.

        For prior: H = log₂(C(|L|, k))
        For posterior (draw): H = log₂(C(|L|-|N(P_t)|, k))

        Returns:
            Entropy value in bits
        """
        if self.posterior_mode == "draw":
            # Entropy over configurations with no legal cards
            L_size = len(self.L) - len(self.N_Pt)
        else:
            # Entropy over all configurations
            L_size = len(self.L)

        k = self.h2_size

        if L_size < k or k <= 0:
            return 0.0

        try:
            n_configs = math.comb(L_size, k)
            return math.log2(n_configs) if n_configs > 0 else 0.0
        except:
            # Approximation for very large numbers
            return L_size * 0.5

    def update(self, opponent_action, new_observation: Tuple):
        """
        Updates belief given opponent action and new observation.

        Args:
            opponent_action: Action opponent took (play or draw)
            new_observation: New observation O = (H_1, |H_2|, |D_g|, P, P_t, G_o)
        """
        if opponent_action is not None:
            if opponent_action.is_play():
                # Case 1: Opponent played a card
                self.update_opponent_played(opponent_action.X_1)
            elif opponent_action.is_draw():
                # Case 2: Opponent drew card(s)
                self.update_opponent_drew()

        # Update with new observation (in case we missed something)
        self.H_1 = new_observation[0]
        self.h2_size = new_observation[1]
        self.dg_size = new_observation[2]
        self.P = new_observation[3]
        self.P_t = new_observation[4]
        self.G_o = new_observation[5]

        # Recompute derived quantities
        self.L = self._compute_L()
        self.N_Pt = self._compute_legal_unknown()

    def __repr__(self):
        mode_str = f", Mode={self.posterior_mode}" if self.posterior_mode else ""
        prob_no_legal = self._prob_no_legal()
        return (
            f"Belief(H_1={len(self.H_1)} cards, "
            f"H_2={self.h2_size} cards, "
            f"D_g={self.dg_size} cards, "
            f"|L|={len(self.L)}, |N(P_t)|={len(self.N_Pt)}, "
            f"P(no_legal)={prob_no_legal:.3f}, "
            f"Entropy={self.entropy():.2f}{mode_str})"
        )


class BeliefUpdater:
    """
    Helper class for updating beliefs through game play.
    Properly tracks opponent actions for Bayesian updates.
    """

    def __init__(self, initial_observation: Tuple):
        """
        Initialize belief updater with initial observation.

        Args:
            initial_observation: Initial O = (H_1, |H_2|, |D_g|, P, P_t, G_o)
        """
        self.belief = Belief(initial_observation)
        self.observation_history = [initial_observation]
        self.action_history = []

    def update(self, opponent_action, new_observation: Tuple):
        """
        Updates belief given opponent action and new observation.

        Args:
            opponent_action: Action opponent took
            new_observation: Resulting observation
        """
        self.belief.update(opponent_action, new_observation)
        self.observation_history.append(new_observation)
        self.action_history.append(opponent_action)

    def get_belief(self) -> Belief:
        """Returns current belief state"""
        return self.belief

    def get_history(self) -> Tuple[List[Tuple], List]:
        """Returns observation and action history"""
        return self.observation_history, self.action_history


# Example usage demonstrating Bayesian updates
if __name__ == "__main__":
    from pomdp import Action

    # Create a game
    uno = Uno()
    uno.new_game(seed=42)
    uno.create_S()

    print("=" * 60)
    print("INITIAL STATE")
    print("=" * 60)

    # Get Player 1's observation
    observation = uno.get_O_space()
    print(f"H_1: {observation[0]}")
    print(f"|H_2|: {observation[1]}")
    print(f"P_t: {observation[4]}")

    # Create belief
    belief = Belief(observation)
    print(f"\n{belief}")
    print(f"Legal unknown cards N(P_t): {belief.N_Pt}")

    # Sample states under prior
    print("\n" + "=" * 60)
    print("SAMPLING UNDER PRIOR BELIEF")
    print("=" * 60)
    samples = belief.sample_states(3)
    for i, state in enumerate(samples):
        print(f"\nSample {i + 1} - Opponent hand H_2: {state[1]}")

    # Simulate Case 2: Opponent draws (no legal play)
    print("\n" + "=" * 60)
    print("CASE 2: OPPONENT DRAWS (NO LEGAL CARDS)")
    print("=" * 60)
    print(f"Before draw: P(no_legal) = {belief._prob_no_legal():.4f}")

    belief.update_opponent_drew()
    print(f"\nAfter observing draw:")
    print(belief)
    print(f"Opponent hand size now: {belief.h2_size}")

    # Sample states under posterior (should exclude legal cards)
    print("\nSampling under posterior (H_2 must exclude legal cards):")
    samples = belief.sample_states(3)
    for i, state in enumerate(samples):
        h2 = state[1]
        print(f"\nSample {i + 1} - H_2: {h2}")
        has_legal = any(c in belief.N_Pt for c in h2)
        print(f"  Contains legal card: {has_legal} (should be False)")

    # Simulate Case 1: Opponent plays a card
    print("\n" + "=" * 60)
    print("CASE 1: OPPONENT PLAYS A CARD")
    print("=" * 60)

    # Pick a card from opponent's likely hand
    if len(belief.L) > 0:
        played_card = belief.L[0]
        print(f"Opponent plays: {played_card}")

        belief.update_opponent_played(played_card)
        print(f"\nAfter opponent play:")
        print(belief)
        print(f"New P_t: {belief.P_t}")
        print(f"New |L|: {len(belief.L)}")
