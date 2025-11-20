"""Script to generate Policy 2 based on MATH2.md particle filter + MCTS approach."""

import json
from pathlib import Path
from src.policy.policy_generator_2 import generate_policy_2
from src.utils.config_loader import get_policy2_config


def main():
    """Generate Policy 2 and save to policy/policy2.json"""
    config = get_policy2_config()
    num_particles = config.get("num_particles", 1000)
    mcts_iterations = config.get("mcts_iterations", 1000)
    planning_horizon = config.get("planning_horizon", 5)
    gamma = config.get("gamma", 0.95)

    print("=" * 60)
    print("Policy Generator 2: Particle Filter + MCTS (MATH2.md)")
    print("=" * 60)
    print("Configuration:")
    print(f"  num_particles: {num_particles}")
    print(f"  mcts_iterations: {mcts_iterations}")
    print(f"  planning_horizon: {planning_horizon}")
    print(f"  gamma: {gamma}")
    print()

    # Generate policy
    policy = generate_policy_2(
        num_particles=num_particles,
        mcts_iterations=mcts_iterations,
        planning_horizon=planning_horizon,
        gamma=gamma,
        num_observations=1000,
    )

    # Save to file
    output_dir = Path("policy")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "policy2.json"

    with open(output_path, "w") as f:
        json.dump(policy, f, indent=2)

    print()
    print(f"Policy saved to {output_path}")
    print(f"Total entries: {len(policy)}")


if __name__ == "__main__":
    main()
