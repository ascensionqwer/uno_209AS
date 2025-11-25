"""Script to test ParticlePolicy runtime decision-making."""

from src.policy.particle_policy import ParticlePolicy
from src.utils.config_loader import get_particle_policy_config
from src.uno.game import Uno
from src.uno.cards import card_to_string


def main():
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
    H_1 = state[0]
    H_2 = state[1]
    D_g = state[2]
    P = state[3]
    P_t = state[4]
    G_o = state[5]

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


if __name__ == "__main__":
    main()
