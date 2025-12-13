#!/usr/bin/env python3
"""
Interactive UNO game: Human vs POMCP AI
Play UNO against the POMCP AI in the terminal.
"""

import sys
import argparse
from src.utils.game_runner import run_player_vs_pomcp_game


def main():
    """Run interactive UNO game with human vs POMCP AI."""
    parser = argparse.ArgumentParser(
        description="Play UNO against POMCP AI in the terminal"
    )
    parser.add_argument(
        "--seed", type=int, help="Random seed for game initialization (optional)"
    )
    parser.add_argument(
        "--player",
        type=int,
        choices=[1, 2],
        default=1,
        help="Which player number you want to be (1 or 2, default: 1)",
    )

    args = parser.parse_args()

    try:
        run_player_vs_pomcp_game(seed=args.seed, human_player=args.player)
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
