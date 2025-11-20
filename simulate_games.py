import random
import copy
from typing import List, Tuple, Optional
import numpy as np
from pomdp import State, Action
from belief import Belief
from uno import Uno


class Uno_AI:
    """
    Optimal AI player using belief states and expectiminimax search.

    Algorithm:
    1. Maintains belief state over hidden information (opponent hand, deck)
    2. Samples possible states from belief distribution
    3. Uses expectiminimax to evaluate actions
    4. Chooses action with highest expected value from Player 1 perspective
    """

    def __init__(self, player_id: int = 1, num_samples: int = 50, lookahead: int = 3):
        """
        Initialize AI player.

        Args:
            player_id: Player number (1 or 2)
            num_samples: Number of belief samples for action evaluation
            lookahead: Search depth for expectiminimax
        """
        self.player_id = player_id
        self.num_samples = num_samples
        self.lookahead = lookahead
        self.belief = None
        self.game = None

    def init_belief(self, game: 'Uno'):
        """Initialize belief state from game observation."""
        self.game = game
        observation = game.get_O_space()
        self.belief = Belief(observation)

    def update_belief(self, opponent_action: Action):
        """Update belief after observing opponent action."""
        if self.belief and opponent_action:
            new_observation = self.game.get_O_space()
            self.belief.update(opponent_action, new_observation)

    def evaluate_state(self, state: State) -> float:
        """
        Evaluate state value from AI player's perspective.

        Terminal: +100 if AI wins, -100 if opponent wins
        Non-terminal: difference in hand sizes (opponent_size - ai_size)
        """
        H_1, H_2, D_g, P, P_t, G_o = state

        if G_o == 'GameOver':
            ai_hand = H_1 if self.player_id == 1 else H_2
            opp_hand = H_2 if self.player_id == 1 else H_1

            if len(ai_hand) == 0:
                return 100.0
            elif len(opp_hand) == 0:
                return -100.0

        # Non-terminal: prefer fewer cards in hand
        ai_hand = H_1 if self.player_id == 1 else H_2
        opp_hand = H_2 if self.player_id == 1 else H_1
        return len(opp_hand) - len(ai_hand)

    def simulate_action(self, state: State, action: Action, player: int) -> State:
        """
        Simulate action on a state copy.

        Args:
            state: Current state
            action: Action to simulate
            player: Which player (1 or 2)

        Returns:
            New state after action
        """
        H_1, H_2, D_g, P, P_t, G_o = state
        # Deep copy to avoid mutations
        H_1, H_2, D_g, P = list(H_1), list(H_2), list(D_g), list(P)

        hand = H_1 if player == 1 else H_2

        if action.is_play():
            # Play card
            if action.X_1 in hand:
                hand.remove(action.X_1)
                P.append(action.X_1)
                P_t = action.X_1

                # Check win condition
                if len(hand) == 0:
                    G_o = 'GameOver'

        elif action.is_draw():
            # Draw cards
            n = action.n
            if len(D_g) >= n:
                drawn = D_g[:n]
                D_g = D_g[n:]
                hand.extend(drawn)
            P_t = P_t  # Unchanged

        return (H_1, H_2, D_g, P, P_t, G_o)

    def expectiminimax(self, state: State, depth: int, player: int) -> float:
        """
        Expectiminimax search from AI perspective.

        Args:
            state: Current game state
            depth: Remaining search depth
            player: Current player (1 or 2)

        Returns:
            Expected value for AI player
        """
        H_1, H_2, D_g, P, P_t, G_o = state

        # Terminal conditions
        if depth == 0 or G_o == 'GameOver':
            return self.evaluate_state(state)

        # Get legal actions for current player
        temp_game = Uno(H_1=H_1, H_2=H_2, D_g=D_g, P=P)
        temp_game.create_S()
        actions = temp_game.get_legal_actions(player)

        if len(actions) == 0:
            return self.evaluate_state(state)

        # Determine if maximizing or minimizing
        is_maximizing = (player == self.player_id)

        if is_maximizing:
            max_value = float('-inf')

            for action in actions:
                next_state = self.simulate_action(state, action, player)

                # If draw action, sample possible outcomes
                if action.is_draw() and len(D_g) > 0:
                    n_samples = min(5, len(D_g))
                    total_value = 0.0

                    for _ in range(n_samples):
                        sampled_state = self.simulate_action(state, action, player)
                        value = self.expectiminimax(sampled_state, depth - 1, 3 - player)
                        total_value += value

                    avg_value = total_value / n_samples
                    max_value = max(max_value, avg_value)
                else:
                    value = self.expectiminimax(next_state, depth - 1, 3 - player)
                    max_value = max(max_value, value)

            return max_value

        else:  # Minimizing
            min_value = float('inf')

            for action in actions:
                next_state = self.simulate_action(state, action, player)

                if action.is_draw() and len(D_g) > 0:
                    n_samples = min(5, len(D_g))
                    total_value = 0.0

                    for _ in range(n_samples):
                        sampled_state = self.simulate_action(state, action, player)
                        value = self.expectiminimax(sampled_state, depth - 1, 3 - player)
                        total_value += value

                    avg_value = total_value / n_samples
                    min_value = min(min_value, avg_value)
                else:
                    value = self.expectiminimax(next_state, depth - 1, 3 - player)
                    min_value = min(min_value, value)

            return min_value

    def choose_action(self) -> Optional[Action]:
        """
        Choose best action using belief-based expectiminimax.

        Returns:
            Best action according to expectiminimax over belief samples
        """
        if not self.belief:
            raise ValueError("Belief not initialized - call init_belief() first")

        # Get legal actions
        actions = self.game.get_legal_actions(self.player_id)

        if len(actions) == 0:
            return None

        if len(actions) == 1:
            return actions[0]

        # Evaluate each action over sampled states
        action_values = {}

        for action in actions:
            total_value = 0.0

            # Sample states from belief and evaluate
            for _ in range(self.num_samples):
                sampled_state = self.belief.sample_state()
                next_state = self.simulate_action(sampled_state, action, self.player_id)
                value = self.expectiminimax(next_state, self.lookahead, 3 - self.player_id)
                total_value += value

            avg_value = total_value / self.num_samples
            action_values[id(action)] = (action, avg_value)

        # Return action with highest expected value
        best_action, best_value = max(action_values.values(), key=lambda x: x[1])
        return best_action

class Uno_Heuristic:
    """
    Simple heuristic player: plays first legal card, otherwise draws.

    This implements the baseline strategy:
    - Check hand for legal cards
    - Play first legal card found
    - If no legal cards, draw from position [0]
    """

    def __init__(self, player_id: int = 2):
        """
        Initialize heuristic player.

        Args:
            player_id: Player number (1 or 2)
        """
        self.player_id = player_id
        self.game = None

    def set_game(self, game: 'Uno'):
        """Set game reference."""
        self.game = game

    def choose_action(self) -> Optional[Action]:
        """
        Choose action using simple heuristic.

        Returns:
            First legal card to play, or draw action if no legal plays
        """
        if not self.game:
            raise ValueError("Game not set - call set_game() first")

        # Get legal actions
        actions = self.game.get_legal_actions(self.player_id)

        if len(actions) == 0:
            return None

        # Return first action (greedy: play first legal card or draw)
        return actions[0]

class GameController:
    """
    Controller for managing games between two players.

    Handles:
    - Game initialization
    - Turn management
    - Belief updates for AI players
    - Game statistics
    """

    def __init__(self, player1, player2, verbose: bool = True):
        """
        Initialize game controller.

        Args:
            player1: Player 1 (Uno_AI or Uno_Heuristic)
            player2: Player 2 (Uno_AI or Uno_Heuristic)
            verbose: Whether to print game progress
        """
        self.player1 = player1
        self.player2 = player2
        self.verbose = verbose
        self.game = None
        self.current_player = 1
        self.turn_count = 0

        # Configure players
        self.player1.player_id = 1
        self.player2.player_id = 2

    def init_game(self, seed: Optional[int] = None):
        """Initialize new game."""
        self.game = Uno()
        self.game.new_game(seed=seed)

        # Set game reference for players
        if hasattr(self.player1, 'init_belief'):
            self.player1.init_belief(self.game)
        if hasattr(self.player1, 'set_game'):
            self.player1.set_game(self.game)

        if hasattr(self.player2, 'init_belief'):
            self.player2.init_belief(self.game)
        if hasattr(self.player2, 'set_game'):
            self.player2.set_game(self.game)

        self.current_player = 1
        self.turn_count = 0

        if self.verbose:
            print("=" * 60)
            print("NEW GAME STARTED")
            print("=" * 60)
            print(f"Player 1 hand: {self.game.H_1}")
            print(f"Player 2 hand size: {len(self.game.H_2)}")
            print(f"Top card: {self.game.P_t}")
            print("=" * 60)

    def play_turn(self) -> bool:
        """
        Play one turn.

        Returns:
            True if game continues, False if game over
        """
        if self.game.G_o == 'GameOver':
            return False

        self.turn_count += 1

        # Get current player
        player = self.player1 if self.current_player == 1 else self.player2
        player_name = "Player 1" if self.current_player == 1 else "Player 2"

        # Choose action
        action = player.choose_action()

        if action is None:
            if self.verbose:
                print(f"Turn {self.turn_count}: {player_name} has no valid actions")
            return False

        # Execute action
        success = self.game.execute_action(action, self.current_player)

        if not success:
            if self.verbose:
                print(f"Turn {self.turn_count}: {player_name} action failed")
            return False

        # Log action
        if self.verbose:
            if action.is_play():
                print(f"Turn {self.turn_count}: {player_name} played {action.X_1}")
            else:
                print(f"Turn {self.turn_count}: {player_name} drew {action.n} card(s)")

        # Update opponent's belief if they are AI
        opponent = self.player2 if self.current_player == 1 else self.player1
        if hasattr(opponent, 'update_belief'):
            opponent.update_belief(action)

        # Check game over
        if self.game.G_o == 'GameOver':
            winner = 1 if len(self.game.H_1) == 0 else 2
            if self.verbose:
                print(f"\n🎉 Player {winner} wins in {self.turn_count} turns!")
            return False

        # Switch player
        self.current_player = 3 - self.current_player
        return True

    def play_game(self, seed: Optional[int] = None, max_turns: int = 500) -> Tuple[int, int]:
        """
        Play complete game.

        Args:
            seed: Random seed for game initialization
            max_turns: Maximum turns before declaring draw

        Returns:
            Tuple of (winner, num_turns) where winner is 1, 2, or 0 for draw
        """
        self.init_game(seed=seed)

        while self.turn_count < max_turns:
            continue_game = self.play_turn()
            if not continue_game:
                break

        # Determine winner
        if self.game.G_o == 'GameOver':
            if len(self.game.H_1) == 0:
                return 1, self.turn_count
            elif len(self.game.H_2) == 0:
                return 2, self.turn_count

        # Draw (max turns reached)
        if self.verbose:
            print(f"\nGame ended in draw after {max_turns} turns")
        return 0, self.turn_count