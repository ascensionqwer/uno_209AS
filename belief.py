from typing import List, Dict, Tuple, Set
import random
from collections import Counter
from cards import Card, RED, YELLOW, GREEN, BLUE
from pomdp import State
from uno import Uno


class Belief:
    """
    Belief state representation for UNO POMDP.

    B: Belief over possible states given observations
    O = (H_1, |H_2|, |D_g|, P, P_t, G_o) - Player 1's observation

    The belief maintains:
    - Known information (H_1, P, P_t, G_o) - directly observable
    - Uncertain information (H_2, D_g) - must infer from observations

    We represent belief as a set of possible states consistent with observations.
    """

    def __init__(self, observation: Tuple):
        """
        Initialize belief from an observation.

        Args:
            observation: O = (H_1, |H_2|, |D_g|, P, P_t, G_o)
        """
        self.H_1 = observation[0]  # Player 1's hand (known)
        self.h2_size = observation[1]  # Size of Player 2's hand (known)
        self.dg_size = observation[2]  # Size of draw deck (known)
        self.P = observation[3]  # Played cards (known)
        self.P_t = observation[4]  # Top card (known)
        self.G_o = observation[5]  # Game status (known)

        # Unknown cards: those not in H_1 or P
        self.unknown_cards = self._compute_unknown_cards()

    def _compute_unknown_cards(self) -> List[Card]:
        """
        Computes the set of cards that are unknown to Player 1.
        Unknown = Full_Deck - H_1 - P

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

        # Count known cards
        known_cards = list(self.H_1) + list(self.P)
        known_counter = Counter(known_cards)

        # Subtract known from full deck
        unknown = []
        full_counter = Counter(full_deck)

        for card, count in full_counter.items():
            unknown_count = count - known_counter.get(card, 0)
            unknown.extend([card] * unknown_count)

        return unknown

    def sample_state(self, seed: int = None) -> State:
        """
        Samples a possible state consistent with current observations.

        Randomly partitions unknown_cards into H_2 and D_g according to
        observed sizes.

        Args:
            seed: Random seed for reproducibility

        Returns:
            A possible State consistent with observations
        """
        rng = random.Random(seed)

        # Shuffle unknown cards
        shuffled_unknown = self.unknown_cards.copy()
        rng.shuffle(shuffled_unknown)

        # Partition into H_2 and D_g based on observed sizes
        H_2_sample = shuffled_unknown[: self.h2_size]
        D_g_sample = shuffled_unknown[self.h2_size : self.h2_size + self.dg_size]

        # Construct state
        state = (
            list(self.H_1),  # H_1 (known)
            H_2_sample,  # H_2 (sampled)
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
            List of possible states
        """
        return [self.sample_state(seed=i) for i in range(n_samples)]

    def update(self, action, observation: Tuple):
        """
        Updates belief after taking action and receiving new observation.

        This is a simplified update that replaces the belief with the new observation.
        A more sophisticated implementation would use Bayesian filtering.

        Args:
            action: Action taken
            observation: New observation O = (H_1, |H_2|, |D_g|, P, P_t, G_o)
        """
        self.H_1 = observation[0]
        self.h2_size = observation[1]
        self.dg_size = observation[2]
        self.P = observation[3]
        self.P_t = observation[4]
        self.G_o = observation[5]

        # Recompute unknown cards
        self.unknown_cards = self._compute_unknown_cards()

    def get_card_probabilities(self, location: str) -> Dict[Card, float]:
        """
        Estimates probability of each card being in a specific location.

        Args:
            location: Either "H_2" or "D_g"

        Returns:
            Dictionary mapping cards to probabilities
        """
        if location not in ["H_2", "D_g"]:
            raise ValueError("location must be 'H_2' or 'D_g'")

        # Count unknown cards
        unknown_counter = Counter(self.unknown_cards)
        total_unknown = len(self.unknown_cards)

        if total_unknown == 0:
            return {}

        # Probability = (count of card in unknown) * (size of location / total unknown)
        location_size = self.h2_size if location == "H_2" else self.dg_size

        probabilities = {}
        for card, count in unknown_counter.items():
            # Hypergeometric-like probability
            probabilities[card] = (
                (count / total_unknown) * location_size / total_unknown * count
            )

        return probabilities

    def entropy(self) -> float:
        """
        Computes entropy of the belief state as a measure of uncertainty.

        Higher entropy = more uncertainty about hidden information.

        Returns:
            Entropy value
        """
        import math

        if len(self.unknown_cards) == 0:
            return 0.0

        # Count unique arrangements
        # This is a simplified entropy based on number of possible partitions
        total_unknown = len(self.unknown_cards)

        if total_unknown == 0:
            return 0.0

        # Combinatorial entropy: log of number of ways to partition
        # C(total_unknown, h2_size) ways to choose H_2 from unknown
        try:
            from math import comb

            n_partitions = comb(total_unknown, self.h2_size)
            entropy = math.log2(n_partitions) if n_partitions > 0 else 0.0
        except:
            # Approximation if comb not available
            entropy = total_unknown * 0.1

        return entropy

    def __repr__(self):
        return (
            f"Belief(H_1={len(self.H_1)} cards, "
            f"H_2={self.h2_size} cards, "
            f"D_g={self.dg_size} cards, "
            f"Unknown={len(self.unknown_cards)} cards, "
            f"Entropy={self.entropy():.2f})"
        )


class BeliefUpdater:
    """
    Helper class for updating beliefs through game play.
    Handles action-observation sequences.
    """

    def __init__(self, initial_observation: Tuple):
        """
        Initialize belief updater with initial observation.

        Args:
            initial_observation: Initial O = (H_1, |H_2|, |D_g|, P, P_t, G_o)
        """
        self.belief = Belief(initial_observation)
        self.history = [initial_observation]

    def update(self, action, new_observation: Tuple):
        """
        Updates belief given action and new observation.

        Args:
            action: Action taken
            new_observation: Resulting observation
        """
        self.belief.update(action, new_observation)
        self.history.append(new_observation)

    def get_belief(self) -> Belief:
        """Returns current belief state"""
        return self.belief

    def get_history(self) -> List[Tuple]:
        """Returns observation history"""
        return self.history


# Example usage
if __name__ == "__main__":
    # Create a game
    uno = Uno()
    uno.new_game(seed=42)
    uno.create_S()

    # Get Player 1's observation
    observation = uno.get_O_space()
    print("Initial Observation:")
    print(f"  H_1: {observation[0]}")
    print(f"  |H_2|: {observation[1]}")
    print(f"  |D_g|: {observation[2]}")
    print(f"  P: {observation[3]}")
    print(f"  P_t: {observation[4]}")
    print(f"  G_o: {observation[5]}")

    # Create belief
    belief = Belief(observation)
    print(f"\n{belief}")

    # Sample possible states
    print("\nSampling 3 possible states:")
    for i, state in enumerate(belief.sample_states(3)):
        print(f"\nSample {i + 1}:")
        print(f"  H_2: {state[1]}")
        print(f"  D_g (first 5): {state[2][:5]}")

    # Show card probabilities
    print("\nCard probabilities in H_2 (top 5):")
    h2_probs = belief.get_card_probabilities("H_2")
    for card, prob in sorted(h2_probs.items(), key=lambda x: -x[1])[:5]:
        print(f"  {card}: {prob:.4f}")

    # Demonstrate belief update
    print("\n--- Simulating action and belief update ---")
    legal_actions = uno.get_legal_actions(player=1)
    if legal_actions:
        action = legal_actions[0]
        print(f"Taking action: {action}")
        uno.execute_action(action, player=1)

        # Get new observation
        new_observation = uno.get_O_space()
        belief.update(action, new_observation)
        print(f"\nUpdated belief: {belief}")
