"""Script to test ParticlePolicy runtime decision-making and naive agent simulations."""

import random
from src.policy.particle_policy import ParticlePolicy
from src.utils.config_loader import get_particle_policy_config
from src.uno.game import Uno
from src.uno.cards import card_to_string
from src.uno.state import Action


def test_particle_policy():
    """Test ParticlePolicy with a sample game."""
    config = get_particle_policy_config()

    print("=" * 60)
    print("Particle Policy: Runtime Particle Filter + MCTS")
    print("=" * 60)
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Initialize policy
    policy = ParticlePolicy(**config)

    # Create a test game
    game = Uno()
    game.new_game()
    game.create_S()
    state = game.State

    # Get observation
    if state is None:
        print("Error: Game state is None")
        return
    
    H_1 = state[0] if len(state) > 0 else []
    H_2 = state[1] if len(state) > 1 else []
    D_g = state[2] if len(state) > 2 else []
    P = state[3] if len(state) > 3 else []
    P_t = state[4] if len(state) > 4 else None
    G_o = state[5] if len(state) > 5 else "Active"

    print("Test Game State (God's View - Full System State):")
    print("=" * 60)
    print(f"H_1 (Player 1 hand): {[card_to_string(c) for c in H_1]} ({len(H_1)} cards)")
    print(f"H_2 (Player 2 hand): {[card_to_string(c) for c in H_2]} ({len(H_2)} cards)")
    print(f"D_g (Draw pile/deck): {len(D_g)} cards")
    print(f"P (Played pile): {[card_to_string(c) for c in P]} ({len(P)} cards)")
    print(f"P_t (Top card): {card_to_string(P_t) if P_t else 'None'}")
    print(f"G_o (Game status): {G_o}")
    print()

    # Get action from policy
    print("Computing optimal action...")
    action = policy.get_action(H_1, len(H_2), len(D_g), P, P_t, G_o)

    print(f"Selected action: {action}")
    print()
    print("Policy test completed successfully!")


def choose_action_naive(game: Uno, player: int):
    """
    Naive agent action selection:
    - Play first legal card found
    - If multiple legal cards, randomly pick one
    - If Wild, choose most frequent color in hand (random if tie)
    - If no legal play, draw 1
    """
    legal_actions = game.get_legal_actions(player)

    if not legal_actions:
        return None

    # Prefer playing over drawing
    play_actions = [a for a in legal_actions if a.is_play()]
    if play_actions:
        action = random.choice(play_actions)
        
        # If Wild card, choose most frequent color in hand
        if action.X_1 and game._is_wild(action.X_1):
            hand = game.H_1 if player == 1 else game.H_2
            color_counts = {}
            for card in hand:
                if card[0] != 'Black':  # Don't count wild cards
                    color_counts[card[0]] = color_counts.get(card[0], 0) + 1
            
            if color_counts:
                max_count = max(color_counts.values())
                most_frequent_colors = [color for color, count in color_counts.items() if count == max_count]
                chosen_color = random.choice(most_frequent_colors)
            else:
                colors = ['Red', 'Yellow', 'Green', 'Blue']
                chosen_color = random.choice(colors)
            
            action.wild_color = chosen_color
        return action

    # Must draw
    return legal_actions[0]


def run_naive_vs_naive_game(seed=None, show_output=True):
    """Run a single game between two naive agents."""
    game = Uno()
    if seed is not None:
        game.new_game(seed=seed)
    else:
        game.new_game()

    current_player = 1
    turn_count = 0
    max_turns = 10000

    if show_output:
        print(f"\n{'='*60}")
        print(f"GAME START - Seed: {seed}")
        print(f"{'='*60}")

    while game.G_o == "Active" and turn_count < max_turns:
        turn_count += 1

        if show_output:
            print(f"\nTurn {turn_count} - Player {current_player}'s turn")
            game.create_S()
            state = game.State
            if state is not None and len(state) > 4:
                print(f"Top Card: {card_to_string(state[4]) if state[4] else 'None'}")
                print(f"Player 1 Hand: {[card_to_string(c) for c in state[0]]}")
                print(f"Player 2 Hand: {[card_to_string(c) for c in state[1]]}")
            else:
                print("Error: Invalid game state")

        # Handle skip
        if game.skip_next:
            if show_output:
                print(f"Player {current_player} is skipped!")
            game.skip_next = False
            current_player = 3 - current_player
            continue

        # Handle draw pending
        if game.draw_pending > 0:
            if show_output:
                print(f"Player {current_player} must draw {game.draw_pending} cards!")
            action = Action(n=game.draw_pending)
            success = game.execute_action(action, current_player)
            if not success:
                if show_output:
                    print(f"Failed to execute draw action for player {current_player}")
                break
            game.skip_next = True
            current_player = 3 - current_player
            continue

        # Choose action using naive policy
        action = choose_action_naive(game, current_player)

        if action is None:
            if show_output:
                print(f"Player {current_player} has no actions available!")
            break

        if show_output:
            if action.is_play():
                card_str = card_to_string(action.X_1)
                if action.wild_color:
                    print(f"Player {current_player} plays {card_str} and chooses color {action.wild_color}")
                else:
                    print(f"Player {current_player} plays {card_str}")
            else:
                print(f"Player {current_player} draws {action.n} card(s)")

        success = game.execute_action(action, current_player)
        if not success:
            if show_output:
                print(f"Action failed for player {current_player}")
            break

        if game.G_o == "GameOver":
            break

        # Switch players
        if not game.player_plays_again and not game.skip_next:
            current_player = 3 - current_player
        elif game.player_plays_again:
            game.player_plays_again = False

    # Determine winner
    game.create_S()
    state = game.State
    
    if show_output:
        print(f"\n{'='*60}")
        print("GAME OVER")
        print(f"{'='*60}")
        print(f"Total turns: {turn_count}")
    
    if state is not None and len(state) >= 2:
        if show_output:
            print(f"Player 1 final hand size: {len(state[0])}")
            print(f"Player 2 final hand size: {len(state[1])}")

        if len(state[0]) == 0:
            if show_output:
                print("🎉 Player 1 (Naive) WINS!")
            return 1, 0
        elif len(state[1]) == 0:
            if show_output:
                print("🎉 Player 2 (Naive) WINS!")
            return 0, 1
        else:
            if show_output:
                print("Game ended without a winner (turn limit reached) - 0.5 wins each")
            return 0.5, 0.5
    else:
        if show_output:
            print("Error: Invalid final state")
        return 0.5, 0.5


def main():
    """Run 10 games of naive agent vs naive agent."""
    print("=" * 60)
    print("NAIVE AGENT VS NAIVE AGENT SIMULATION")
    print("=" * 60)
    
    player1_wins = 0
    player2_wins = 0
    
    for game_num in range(1, 11):
        print(f"\n{'#'*40}")
        print(f"GAME {game_num}/10")
        print(f"{'#'*40}")
        
        p1_wins, p2_wins = run_naive_vs_naive_game(seed=game_num, show_output=False)
        player1_wins += p1_wins
        player2_wins += p2_wins
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Player 1 total wins: {player1_wins}")
    print(f"Player 2 total wins: {player2_wins}")
    print(f"Player 1 win rate: {player1_wins/10:.1%}")
    print(f"Player 2 win rate: {player2_wins/10:.1%}")


if __name__ == "__main__":
    main()
