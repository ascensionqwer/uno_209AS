"""Script to test ParticlePolicy runtime decision-making."""

from src.policy.particle_policy import ParticlePolicy
from src.utils.config_loader import get_particle_policy_config
from src.uno.game import Uno


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
    opponent_size = len(state[1])
    deck_size = len(state[2])
    P = state[3]
    P_t = state[4]
    G_o = state[5]

    print("Test Game State:")
    print(f"  Player 1 hand size: {len(H_1)}")
    print(f"  Opponent hand size: {opponent_size}")
    print(f"  Deck size: {deck_size}")
    print(f"  Top card: {P_t}")
    print()

    # Get action from policy
    print("Computing optimal action...")
    action = policy.get_action(H_1, opponent_size, deck_size, P, P_t, G_o)

    print(f"Selected action: {action}")
    print()
    print("Policy test completed successfully!")


if __name__ == "__main__":
    main()
