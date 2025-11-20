"""Script to generate Policy 1 based on MATH.md value function approach."""

import json
from pathlib import Path
from src.policy.policy_generator_1 import generate_policy_1
from src.utils.config_loader import get_policy1_config


def main():
    """Generate Policy 1 and save to policy/policy1.json"""
    config = get_policy1_config()
    gamma = config.get("gamma", 0.95)
    num_belief_samples = config.get("num_belief_samples", 50)
    max_depth = config.get("max_depth", 3)

    print("=" * 60)
    print("Policy Generator 1: Value Function Approach (MATH.md)")
    print("=" * 60)
    print("Configuration:")
    print(f"  gamma: {gamma}")
    print(f"  num_belief_samples: {num_belief_samples}")
    print(f"  max_depth: {max_depth}")
    print("  num_observations: ALL (full enumeration)")
    print()

    # Generate policy - enumerates ALL possible states
    policy = generate_policy_1(
        gamma=gamma,
        num_belief_samples=num_belief_samples,
        max_depth=max_depth,
    )

    # Save to file
    output_dir = Path("policy")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "policy1.json"

    with open(output_path, "w") as f:
        json.dump(policy, f, indent=2)

    print()
    print(f"Policy saved to {output_path}")
    print(f"Total entries: {len(policy)}")


if __name__ == "__main__":
    main()
