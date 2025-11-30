"""Shared game runner utilities for UNO simulations."""

import time
import random
from typing import Optional
from src.uno import Uno, Action, card_to_string
from src.uno.cards import RED, YELLOW, GREEN, BLUE
from src.policy.particle_policy import ParticlePolicy
from src.utils.config_loader import get_particle_policy_config


def display_game_state(game: Uno, current_player: int, policy=None):
    """Display current game state (full visibility - God mode)."""
    print("\n" + "=" * 60)
    print(f"Player {current_player}'s Turn")
    print("=" * 60)
    game.create_S()
    state = game.State

    print(f"\nTop Card: {card_to_string(state[4]) if state[4] is not None else 'None'}")
    if game.current_color:
        print(f"Current Color: {game.current_color}")
    print(f"Deck: {len(state[2])} cards remaining")
    if policy:
        print(f"ParticlePolicy Cache Size: {policy.cache.size()}")
    print(f"\nPlayer 1 Hand ({len(state[0])} cards):")
    for i, card in enumerate(state[0], 1):
        print(f"  {i}. {card_to_string(card)}")
    print(f"\nPlayer 2 Hand ({len(state[1])} cards):")
    for i, card in enumerate(state[1], 1):
        print(f"  {i}. {card_to_string(card)}")
    print()


def choose_action_simple(game: Uno, player: int) -> Optional[Action]:
    """
    Simple automated action selection:
    - Play first legal card found (first index)
    - If Wild, choose most common color in hand
    - If no legal play, draw 1
    """
    legal_actions = game.get_legal_actions(player)

    if not legal_actions:
        return None

    # Prefer playing over drawing
    play_actions = [a for a in legal_actions if a.is_play()]
    if play_actions:
        action = play_actions[0]
        # If Wild card, set color
        if action.X_1 and game._is_wild(action.X_1):
            hand = game.H_1 if player == 1 else game.H_2
            action.wild_color = game._choose_wild_color(hand)
        return action

    # Must draw
    return legal_actions[0]


# Global variable to store naive decision times
_naive_decision_times = []


def choose_action_naive(game: Uno, player: int) -> Optional[Action]:
    """
    Naive action selection with timing.
    - Play first legal card found (if multiple, randomly pick one)
    - If Wild, choose most frequent color in hand (random if tie)
    - If no legal play, draw 1
    """
    start_time = time.time()

    legal_actions = game.get_legal_actions(player)

    if not legal_actions:
        return None

    # Prefer playing over drawing
    play_actions = [a for a in legal_actions if a.is_play()]
    if play_actions:
        action = play_actions[0]  # Play first legal card found

        # If Wild card, choose most frequent color in hand
        if action.X_1 and game._is_wild(action.X_1):
            hand = game.H_1 if player == 1 else game.H_2
            color_counts = {}
            for card in hand:
                if card[0] != "Black":  # Don't count wild cards
                    color_counts[card[0]] = color_counts.get(card[0], 0) + 1

            if color_counts:
                max_count = max(color_counts.values())
                most_frequent_colors = [
                    color for color, count in color_counts.items() if count == max_count
                ]
                chosen_color = random.choice(most_frequent_colors)
            else:
                # No colored cards in hand, pick randomly from 4 viable colors
                colors = [RED, YELLOW, GREEN, BLUE]
                chosen_color = random.choice(colors)

            action.wild_color = chosen_color
    else:
        # Must draw
        action = legal_actions[0]

    decision_time = time.time() - start_time
    _naive_decision_times.append(decision_time)

    return action


def get_naive_decision_stats() -> dict:
    """Get naive decision timing statistics."""
    if not _naive_decision_times:
        return {"avg": 0, "max": 0, "min": 0, "count": 0}

    return {
        "avg": sum(_naive_decision_times) / len(_naive_decision_times),
        "max": max(_naive_decision_times),
        "min": min(_naive_decision_times),
        "count": len(_naive_decision_times),
    }


def reset_naive_decision_stats():
    """Reset naive decision timing statistics."""
    global _naive_decision_times
    _naive_decision_times = []


def run_single_game(
    seed: Optional[int] = None, show_config: bool = True
) -> tuple[int, int]:
    """
    Run a single UNO game with full verbose logging (God mode view).

    Args:
        seed: Random seed for game initialization (None for random)
        show_config: Whether to show ParticlePolicy configuration at start

    Returns:
        Tuple of (turn_count, winner) where winner is 1, 2, or 0 (no winner/safety limit)
    """
    # Initialize game
    game = Uno()
    game.new_game(seed=seed)

    # Initialize ParticlePolicy for Player 1
    config = get_particle_policy_config()
    policy = ParticlePolicy(**config)

    if show_config:
        print("\nParticlePolicy Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print()

    current_player = 1
    turn_count = 0
    max_turns = 1000  # Safety limit
    consecutive_no_progress = 0
    max_no_progress = 50  # Safety check for infinite loops
    prev_state = None

    while game.G_o == "Active" and turn_count < max_turns:
        turn_count += 1

        # Check if we need to skip this player
        if game.skip_next:
            print(f"\n>>> Player {current_player} is skipped!")
            game.skip_next = False
            current_player = 3 - current_player  # Switch: 1->2, 2->1
            # Track state after skip
            game.create_S()
            current_state = (len(game.H_1), len(game.H_2), game.G_o, current_player)
            if prev_state == current_state:
                consecutive_no_progress += 1
                if consecutive_no_progress >= max_no_progress:
                    print(
                        f"\n>>> INFINITE LOOP DETECTED: No progress for {max_no_progress} consecutive turns!"
                    )
                    print(f">>> Breaking at turn {turn_count}")
                    break
            else:
                consecutive_no_progress = 0
            prev_state = current_state
            continue

        # Check if player needs to draw cards (from Draw 2 or Wild Draw 4)
        if game.draw_pending > 0:
            print(f"\n>>> Player {current_player} must draw {game.draw_pending} cards!")
            action = Action(n=game.draw_pending)
            success = game.execute_action(action, current_player)
            if not success:
                print(f">>> Failed to execute draw action for player {current_player}")
                break
            # After drawing, skip this player's turn
            game.skip_next = True
            display_game_state(game, current_player)
            current_player = 3 - current_player
            # Track state after draw
            game.create_S()
            current_state = (len(game.H_1), len(game.H_2), game.G_o, current_player)
            if prev_state == current_state:
                consecutive_no_progress += 1
                if consecutive_no_progress >= max_no_progress:
                    print(
                        f"\n>>> INFINITE LOOP DETECTED: No progress for {max_no_progress} consecutive turns!"
                    )
                    print(f">>> Breaking at turn {turn_count}")
                    break
            else:
                consecutive_no_progress = 0
            prev_state = current_state
            continue

        display_game_state(
            game, current_player, policy if current_player == 1 else None
        )

        # Get current state for Player 1's policy
        game.create_S()
        state = game.State

        # Choose action based on player
        if current_player == 1:
            # Player 1: Use ParticlePolicy
            print("Player 1 (ParticlePolicy) computing action...")
            action = policy.get_action(
                state[0], len(state[1]), len(state[2]), state[3], state[4], state[5]
            )
            policy.update_after_action(action)
        else:
            # Player 2: Use simple policy
            action = choose_action_simple(game, current_player)

        if action is None:
            print(f"Player {current_player} has no actions available!")
            break

        if action.is_play():
            card_str = card_to_string(action.X_1)
            if action.wild_color:
                print(
                    f"Player {current_player} plays {card_str} and chooses color {action.wild_color}"
                )
            else:
                print(f"Player {current_player} plays {card_str}")
        else:
            print(f"Player {current_player} draws {action.n} card(s)")

        success = game.execute_action(action, current_player)
        if not success:
            print(f"Action failed for player {current_player}")
            break

        # Track state after action for infinite loop detection
        game.create_S()
        current_state = (len(game.H_1), len(game.H_2), game.G_o, current_player)
        if prev_state == current_state:
            consecutive_no_progress += 1
            if consecutive_no_progress >= max_no_progress:
                print(
                    f"\n>>> INFINITE LOOP DETECTED: No progress for {max_no_progress} consecutive turns!"
                )
                print(f">>> Breaking at turn {turn_count}")
                break
        else:
            consecutive_no_progress = 0
        prev_state = current_state

        # Check game over
        if game.G_o == "GameOver":
            break

        # Switch players (unless current player plays again)
        if not game.player_plays_again and not game.skip_next:
            current_player = 3 - current_player
        elif game.player_plays_again:
            # Reset flag - player will continue their turn
            game.player_plays_again = False

    # Game over
    print("\n" + "=" * 60)
    print("GAME OVER")
    print("=" * 60)
    game.create_S()
    state = game.State

    print(f"Total turns: {turn_count}")
    print(f"Player 1 final hand size: {len(state[0])}")
    print(f"Player 2 final hand size: {len(state[1])}")
    print(f"ParticlePolicy cache size: {policy.cache.size()}")

    # Determine winner
    if len(state[0]) == 0:
        print("\n🎉 Player 1 (ParticlePolicy) WINS!")
        winner = 1
    elif len(state[1]) == 0:
        print("\n🎉 Player 2 (Simple Policy) WINS!")
        winner = 2
    else:
        print("\nGame ended without a winner (safety limit reached)")
        winner = 0

    return turn_count, winner


def run_naive_vs_naive_game(
    seed: Optional[int] = None, show_output: bool = False
) -> tuple[int, int]:
    """
    Run a single UNO game with both players using naive policies.

    Args:
        seed: Random seed for game initialization (None for random)
        show_output: Whether to show game state output (default False for batch runs)

    Returns:
        Tuple of (turn_count, winner) where winner is 1, 2, or 0 (no winner/safety limit)
    """
    # Initialize game
    game = Uno()
    if seed is not None:
        game.new_game(seed=seed)
    else:
        game.new_game()

    current_player = 1
    turn_count = 0
    max_turns = 1000  # Safety limit
    consecutive_no_progress = 0
    max_no_progress = 50  # Safety check for infinite loops
    prev_state = None

    while game.G_o == "Active" and turn_count < max_turns:
        turn_count += 1

        # Check if we need to skip this player
        if game.skip_next:
            if show_output:
                print(f"\n>>> Player {current_player} is skipped!")
            game.skip_next = False
            current_player = 3 - current_player  # Switch: 1->2, 2->1
            # Track state after skip
            game.create_S()
            current_state = (len(game.H_1), len(game.H_2), game.G_o, current_player)
            if prev_state == current_state:
                consecutive_no_progress += 1
                if consecutive_no_progress >= max_no_progress:
                    if show_output:
                        print(
                            f"\n>>> INFINITE LOOP DETECTED: No progress for {max_no_progress} consecutive turns!"
                        )
                        print(f">>> Breaking at turn {turn_count}")
                    break
            else:
                consecutive_no_progress = 0
            prev_state = current_state
            continue

        # Check if player needs to draw cards (from Draw 2 or Wild Draw 4)
        if game.draw_pending > 0:
            if show_output:
                print(
                    f"\n>>> Player {current_player} must draw {game.draw_pending} cards!"
                )
            action = Action(n=game.draw_pending)
            success = game.execute_action(action, current_player)
            if not success:
                if show_output:
                    print(
                        f">>> Failed to execute draw action for player {current_player}"
                    )
                break
            # After drawing, skip this player's turn
            game.skip_next = True
            if show_output:
                display_game_state(game, current_player)
            current_player = 3 - current_player
            # Track state after draw
            game.create_S()
            current_state = (len(game.H_1), len(game.H_2), game.G_o, current_player)
            if prev_state == current_state:
                consecutive_no_progress += 1
                if consecutive_no_progress >= max_no_progress:
                    if show_output:
                        print(
                            f"\n>>> INFINITE LOOP DETECTED: No progress for {max_no_progress} consecutive turns!"
                        )
                        print(f">>> Breaking at turn {turn_count}")
                    break
            else:
                consecutive_no_progress = 0
            prev_state = current_state
            continue

        if show_output:
            display_game_state(game, current_player)

        # Choose action for both players using naive policy
        action = choose_action_naive(game, current_player)

        if action is None:
            if show_output:
                print(f"Player {current_player} has no actions available!")
            break

        if show_output:
            if action.is_play():
                card_str = (
                    card_to_string(action.X_1) if action.X_1 is not None else "Unknown"
                )
                if action.wild_color:
                    print(
                        f"Player {current_player} plays {card_str} and chooses color {action.wild_color}"
                    )
                else:
                    print(f"Player {current_player} plays {card_str}")
            else:
                print(f"Player {current_player} draws {action.n} card(s)")

        success = game.execute_action(action, current_player)
        if not success:
            if show_output:
                print(f"Action failed for player {current_player}")
            break

        # Track state after action for infinite loop detection
        game.create_S()
        current_state = (len(game.H_1), len(game.H_2), game.G_o, current_player)
        if prev_state == current_state:
            consecutive_no_progress += 1
            if consecutive_no_progress >= max_no_progress:
                if show_output:
                    print(
                        f"\n>>> INFINITE LOOP DETECTED: No progress for {max_no_progress} consecutive turns!"
                    )
                    print(f">>> Breaking at turn {turn_count}")
                break
        else:
            consecutive_no_progress = 0
        prev_state = current_state

        # Check game over
        if game.G_o == "GameOver":
            break

        # Switch players (unless current player plays again)
        if not game.player_plays_again and not game.skip_next:
            current_player = 3 - current_player
        elif game.player_plays_again:
            # Reset flag - player will continue their turn
            game.player_plays_again = False

    # Game over
    if show_output:
        print("\n" + "=" * 60)
        print("GAME OVER")
        print("=" * 60)
        game.create_S()
        state = game.State
        print(f"Total turns: {turn_count}")
        print(f"Player 1 final hand size: {len(state[0])}")
        print(f"Player 2 final hand size: {len(state[1])}")

    # Determine winner
    game.create_S()
    state = game.State
    if len(state[0]) == 0:
        if show_output:
            print("\n🎉 Player 1 (Naive) WINS!")
        winner = 1
    elif len(state[1]) == 0:
        if show_output:
            print("\n🎉 Player 2 (Naive) WINS!")
        winner = 2
    else:
        if show_output:
            print("\nGame ended without a winner (safety limit reached)")
        winner = 0

    return turn_count, winner
