"""
Full UNO game simulation with belief tracking.

This demonstrates:
- Complete game play between two players
- Player 1 with belief state tracking
- Simple heuristic policies for both players
- Turn-by-turn game progression
"""

from uno import Uno
from pomdp import Action
from belief import Belief, BeliefUpdater
import random


def simple_policy_player1(legal_actions, hand, belief=None):
    """
    Policy for Player 1: Play highest value card.

    Args:
        legal_actions: List of legal actions
        hand: Current hand
        belief: Current belief state (optional)

    Returns:
        Selected action
    """
    if len(legal_actions) == 0:
        return None

    # Separate play actions from draw actions
    play_actions = [a for a in legal_actions if a.is_play()]
    draw_actions = [a for a in legal_actions if a.is_draw()]

    if play_actions:
        # Play the card with highest value
        best_action = max(play_actions, key=lambda a: a.X_1[1])
        return best_action
    elif draw_actions:
        # Must draw
        return draw_actions[0]

    return legal_actions[0]


def simple_policy_player2(legal_actions, hand):
    """
    Policy for Player 2: Play first legal card in hand (no optimization).

    Args:
        legal_actions: List of legal actions
        hand: Current hand

    Returns:
        Selected action
    """
    if len(legal_actions) == 0:
        return None

    # Separate play actions from draw actions
    play_actions = [a for a in legal_actions if a.is_play()]
    draw_actions = [a for a in legal_actions if a.is_draw()]

    if play_actions:
        # Play first legal card found in hand
        # Find which card in hand is playable first
        for card in hand:
            for action in play_actions:
                if action.X_1 == card:
                    return action
        # Fallback: just return first play action
        return play_actions[0]
    elif draw_actions:
        # Must draw
        return draw_actions[0]

    return legal_actions[0]


def play_full_game(seed=42, verbose=True, max_turns=200):
    """
    Plays a complete game of UNO with belief tracking for Player 1.

    Args:
        seed: Random seed for reproducibility
        verbose: Print turn-by-turn information
        max_turns: Maximum turns before declaring draw

    Returns:
        Winner ("Player1", "Player2", or "Draw"), number of turns
    """
    # Initialize game
    uno = Uno()
    uno.new_game(seed=seed, deal=7)
    uno.create_S()

    # Initialize belief for Player 1
    initial_obs = uno.get_O_space()
    belief_updater = BeliefUpdater(initial_obs)

    if verbose:
        print("=" * 70)
        print("UNO GAME START")
        print("=" * 70)
        print(f"Starting hand Player 1: {uno.H_1}")
        print(f"Starting hand Player 2 size: {len(uno.H_2)}")
        print(f"Top card: {uno.P_t}")
        print(f"Deck size: {len(uno.D_g)}")
        print()

    turn = 0
    current_player = 1  # Start with Player 1

    while uno.G_o == "Active" and turn < max_turns:
        turn += 1

        if verbose:
            print("-" * 70)
            print(f"TURN {turn} - Player {current_player}'s turn")
            print("-" * 70)

        # Get legal actions for current player
        legal_actions = uno.get_legal_actions(player=current_player)

        if verbose:
            print(f"Top card: {uno.P_t}")
            if current_player == 1:
                print(f"Player 1 hand: {uno.H_1}")
            else:
                print(f"Player 2 hand size: {len(uno.H_2)}")
            print(
                f"Legal actions ({len(legal_actions)}): {legal_actions[:3]}{'...' if len(legal_actions) > 3 else ''}"
            )

        # Select action based on policy
        if current_player == 1:
            # Player 1 uses belief and optimizes for highest value
            belief = belief_updater.get_belief()
            if verbose:
                print(
                    f"Player 1 belief: |L|={len(belief.L)}, |N(P_t)|={len(belief.N_Pt)}, "
                    f"P(opp_no_legal)={belief._prob_no_legal():.3f}"
                )
            action = simple_policy_player1(legal_actions, uno.H_1, belief)
        else:
            # Player 2 plays first legal card (simple)
            action = simple_policy_player2(legal_actions, uno.H_2)

        if action is None:
            if verbose:
                print(f"ERROR: No legal actions available!")
            break

        if verbose:
            if action.is_play():
                print(f"Player {current_player} plays: {action.X_1}")
            else:
                print(f"Player {current_player} draws {action.n} card(s)")

        # Execute action
        success = uno.execute_action(action, player=current_player)

        if not success:
            if verbose:
                print(f"ERROR: Action execution failed!")
            break

        if verbose and action.is_draw():
            print(f"  Drew: {action.Y_n}")

        # Update belief if Player 2 just acted
        if current_player == 2:
            new_obs = uno.get_O_space()
            belief_updater.update(action, new_obs)

            if verbose:
                belief = belief_updater.get_belief()
                if action.is_draw():
                    print(f"  → Belief updated: Player 2 had no legal cards!")
                print(
                    f"  → New belief: |L|={len(belief.L)}, Entropy={belief.entropy():.2f}"
                )

        # Check win condition
        if len(uno.H_1) == 0:
            if verbose:
                print("\n" + "=" * 70)
                print("PLAYER 1 WINS!")
                print("=" * 70)
            return "Player1", turn
        elif len(uno.H_2) == 0:
            if verbose:
                print("\n" + "=" * 70)
                print("PLAYER 2 WINS!")
                print("=" * 70)
            return "Player2", turn

        # Switch player
        current_player = 2 if current_player == 1 else 1

        if verbose:
            print(
                f"Cards remaining - P1: {len(uno.H_1)}, P2: {len(uno.H_2)}, Deck: {len(uno.D_g)}"
            )

    if verbose:
        print("\n" + "=" * 70)
        print(f"GAME ENDED IN DRAW (reached {max_turns} turns)")
        print("=" * 70)

    return "Draw", turn


def run_multiple_games(n_games=5, verbose_first=True):
    """
    Runs multiple games and reports statistics.

    Args:
        n_games: Number of games to simulate
        verbose_first: Print details of first game only
    """
    results = {"Player1": 0, "Player2": 0, "Draw": 0}
    turn_counts = []

    for i in range(n_games):
        print(f"\n{'=' * 70}")
        print(f"GAME {i + 1}/{n_games}")
        print(f"{'=' * 70}")

        winner, turns = play_full_game(
            seed=i, verbose=(i == 0 and verbose_first), max_turns=1000
        )

        results[winner] += 1
        turn_counts.append(turns)

        if not (i == 0 and verbose_first):
            print(f"Result: {winner} won in {turns} turns")

    # Print statistics
    print("\n" + "=" * 70)
    print("GAME STATISTICS")
    print("=" * 70)
    print(f"Total games: {n_games}")
    print(
        f"Player 1 wins: {results['Player1']} ({results['Player1'] / n_games * 100:.1f}%)"
    )
    print(
        f"Player 2 wins: {results['Player2']} ({results['Player2'] / n_games * 100:.1f}%)"
    )
    print(f"Draws: {results['Draw']} ({results['Draw'] / n_games * 100:.1f}%)")
    print(f"Average turns per game: {sum(turn_counts) / len(turn_counts):.1f}")
    print(f"Min turns: {min(turn_counts)}, Max turns: {max(turn_counts)}")


if __name__ == "__main__":
    # Option 1: Play a single game with detailed output
    print("OPTION 1: Single detailed game")
    print()
    winner, turns = play_full_game(seed=42, verbose=True, max_turns=1000)
    print(f"\nFinal result: {winner} won in {turns} turns")

    # Option 2: Run multiple games for statistics
    print("\n\n" + "=" * 70)
    print("OPTION 2: Multiple games for statistics")
    print("=" * 70)
    run_multiple_games(n_games=5, verbose_first=False)
