# Generate policy and run simulation game
_Exported on 11/25/2025 at 02:34:27 PST from Cursor (2.1.26)_

---

**User**

generate the policy and then run the full simulation game engine for one full game and lets see what happens

---

**Cursor**


Reviewing the codebase to understand policy generation and the game engine, then generating the policy and running a full game simulation.



Running the full game simulation. The policy is initialized automatically when the ParticlePolicy is created.



Trying `python3`:


---

**User**

remember, @AGENTS.md use uv for python

---

**Cursor**


Re-running the simulation with `uv`:



Ran the full game simulation using `uv`. Results:

**Game Summary:**
- **Winner**: Player 2 (Simple Policy) won with 0 cards remaining
- **Total turns**: 14
- **Player 1 (ParticlePolicy)**: Ended with 10 cards
- **ParticlePolicy cache**: 5 cached particle sets

**Notable events:**
1. Player 1 opened with a WILD card, choosing blue
2. Player 2 played a WILD_DRAW_4, forcing Player 1 to draw 4 cards
3. Player 2 used REVERSE cards to maintain control
4. Player 1 drew a card when Player 2 had 1 card left, allowing Player 2 to win

The ParticlePolicy initialized and made decisions using the particle filter + MCTS approach, with the cache growing as the game progressed. The simple policy won this game.

---

**User**

is the deck randomly distributed to each player?

---

**Cursor**


Checking how cards are distributed in the game initialization:



Yes. The deck is randomly shuffled before dealing.

Looking at the `new_game` method:

```60:67:src/uno/game.py
rng = random.Random(seed)
full_deck = self.build_full_deck()
rng.shuffle(full_deck)

# Deal to hands
self.H_1 = full_deck[:deal]
self.H_2 = full_deck[deal : 2 * deal]
remaining = full_deck[2 * deal :]
```

1. Builds the full 108-card deck
2. Shuffles it using `random.Random(seed)` (seed=42 in `main.py`)
3. Deals sequentially from the shuffled deck:
   - First 7 cards → Player 1 (H_1)
   - Next 7 cards → Player 2 (H_2)
   - Remaining cards → draw pile (D_g)

The distribution is random (via shuffle), but dealing is sequential from the shuffled deck. With a fixed seed, the same shuffle/deal order is reproducible.

```python
rng = random.Random(seed)
full_deck = self.build_full_deck()
rng.shuffle(full_deck)

# Deal to hands
self.H_1 = full_deck[:deal]
self.H_2 = full_deck[deal : 2 * deal]
remaining = full_deck[2 * deal :]
```

---

**User**

increase the number of particles cause player 1 is sucking really bad now idk what is going on, please elaborate then fix the issue

---

**Cursor**


Analyzing why Player 1 is underperforming and increasing particles. Checking the current configuration and particle policy implementation:



Analyzing why Player 1 is underperforming and increasing particles and related parameters:



Issue: parameters are too low, causing poor decisions.

- num_particles = 100: too few to approximate Player 2's hand distribution.
- mcts_iterations = 50: insufficient tree exploration.
- planning_horizon = 3: too short to see Player 2 winning soon.
- rollout_particle_sample_size = 10: noisy value estimates during rollouts.

Increasing these parameters:


```python
from src.uno import Uno, Action, card_to_string
from src.policy.particle_policy import ParticlePolicy
from src.utils.config_loader import get_particle_policy_config


def display_game_state(game: Uno, current_player: int, policy=None):
    """Display current game state (full visibility - God mode)."""
    print("\n" + "=" * 60)
    print(f"Player {current_player}'s Turn")
    print("=" * 60)
    game.create_S()
    state = game.State

    print(f"\nTop Card: {card_to_string(state[4]) if state[4] else 'None'}")
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


def choose_action_simple(game: Uno, player: int):
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


def main():
    """Run 2-player UNO game simulator.
    
    Player 1 uses ParticlePolicy, Player 2 uses simple policy.
    """
    print("=" * 60)
    print("UNO Simulation: ParticlePolicy (Player 1) vs Simple Policy (Player 2)")
    print("=" * 60)

    # Initialize game
    game = Uno()
    game.new_game(seed=42)  # Fixed seed for reproducibility

    # Initialize ParticlePolicy for Player 1
    config = get_particle_policy_config()
    # Increased parameters for better decision-making
    config["num_particles"] = 500  # Better approximation of opponent's hand
    config["mcts_iterations"] = 200  # More thorough tree exploration
    config["planning_horizon"] = 5  # Look further ahead to see winning threats
    config["rollout_particle_sample_size"] = 20  # More accurate value estimates
    policy = ParticlePolicy(**config)

    print(f"\nParticlePolicy Configuration:")
    print(f"  num_particles: {config['num_particles']}")
    print(f"  mcts_iterations: {config['mcts_iterations']}")
    print(f"  planning_horizon: {config['planning_horizon']}")
    print()

    current_player = 1
    turn_count = 0
    max_turns = 1000  # Safety limit

    while game.G_o == "Active" and turn_count < max_turns:
        turn_count += 1

        # Check if we need to skip this player
        if game.skip_next:
            print(f"\n>>> Player {current_player} is skipped!")
            game.skip_next = False
            current_player = 3 - current_player  # Switch: 1->2, 2->1
            continue

        # Check if player needs to draw cards (from Draw 2 or Wild Draw 4)
        if game.draw_pending > 0:
            print(f"\n>>> Player {current_player} must draw {game.draw_pending} cards!")
            action = Action(n=game.draw_pending)
            game.execute_action(action, current_player)
            # After drawing, skip this player's turn
            game.skip_next = True
            display_game_state(game, current_player)
            current_player = 3 - current_player
            continue

        display_game_state(game, current_player, policy if current_player == 1 else None)

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

    if len(state[0]) == 0:
        print("\n🎉 Player 1 (ParticlePolicy) WINS!")
    elif len(state[1]) == 0:
        print("\n🎉 Player 2 (Simple Policy) WINS!")
    else:
        print("\nGame ended without a winner (safety limit reached)")


if __name__ == "__main__":
    main()

```


Re-running the simulation with the improved parameters:


---

**User**

don't keep 'legacy' code or whatever, if it needs to get removed or refactored then do it i don't care about code history cause github handles that for me

---

**Cursor**


Searching for legacy code to remove:



Checking if the legacy code is used anywhere:


---

**User**

additionally, make sure all parameters are taken from @config.jsonc bro don't hardcode stuff that overrides it within the .py files

---

**Cursor**


Removing legacy code and ensuring all parameters come from `config.jsonc`. Checking what's hardcoded:



Removing legacy code and hardcoded overrides:


```jsonc
{
  "particle_policy": {
    // Number of particles for belief approximation
    // Each particle represents a possible opponent hand + deck configuration
    // More particles = better approximation but slower computation
    "num_particles": 500,
    // Number of MCTS iterations per decision (Selection -> Expansion -> Simulation -> Backpropagation)
    // More iterations = better action selection but slower
    "mcts_iterations": 200,
    // Maximum lookahead depth when simulating rollouts to estimate action values
    // Deeper = better estimates but slower
    "planning_horizon": 5,
    // Discount factor for value function (0-1, typically 0.95)
    "gamma": 0.95,
    // UCB1 exploration constant for MCTS tree search
    // Higher = more exploration, lower = more exploitation
    "ucb_c": 1.414,
    // Number of particles to sample for each rollout simulation
    // Fewer = faster but less accurate value estimates
    "rollout_particle_sample_size": 20,
    // Effective sample size threshold for resampling (fraction of num_particles)
    // Lower = resample more often (better diversity but slower)
    "resample_threshold": 0.5
  }
}

```

```python
from src.uno import Uno, Action, card_to_string
from src.policy.particle_policy import ParticlePolicy
from src.utils.config_loader import get_particle_policy_config


def display_game_state(game: Uno, current_player: int, policy=None):
    """Display current game state (full visibility - God mode)."""
    print("\n" + "=" * 60)
    print(f"Player {current_player}'s Turn")
    print("=" * 60)
    game.create_S()
    state = game.State

    print(f"\nTop Card: {card_to_string(state[4]) if state[4] else 'None'}")
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


def choose_action_simple(game: Uno, player: int):
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


def main():
    """Run 2-player UNO game simulator.
    
    Player 1 uses ParticlePolicy, Player 2 uses simple policy.
    """
    print("=" * 60)
    print("UNO Simulation: ParticlePolicy (Player 1) vs Simple Policy (Player 2)")
    print("=" * 60)

    # Initialize game
    game = Uno()
    game.new_game(seed=42)  # Fixed seed for reproducibility

    # Initialize ParticlePolicy for Player 1
    config = get_particle_policy_config()
    policy = ParticlePolicy(**config)

    print(f"\nParticlePolicy Configuration:")
    print(f"  num_particles: {config['num_particles']}")
    print(f"  mcts_iterations: {config['mcts_iterations']}")
    print(f"  planning_horizon: {config['planning_horizon']}")
    print()

    current_player = 1
    turn_count = 0
    max_turns = 1000  # Safety limit

    while game.G_o == "Active" and turn_count < max_turns:
        turn_count += 1

        # Check if we need to skip this player
        if game.skip_next:
            print(f"\n>>> Player {current_player} is skipped!")
            game.skip_next = False
            current_player = 3 - current_player  # Switch: 1->2, 2->1
            continue

        # Check if player needs to draw cards (from Draw 2 or Wild Draw 4)
        if game.draw_pending > 0:
            print(f"\n>>> Player {current_player} must draw {game.draw_pending} cards!")
            action = Action(n=game.draw_pending)
            game.execute_action(action, current_player)
            # After drawing, skip this player's turn
            game.skip_next = True
            display_game_state(game, current_player)
            current_player = 3 - current_player
            continue

        display_game_state(game, current_player, policy if current_player == 1 else None)

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

    if len(state[0]) == 0:
        print("\n🎉 Player 1 (ParticlePolicy) WINS!")
    elif len(state[1]) == 0:
        print("\n🎉 Player 2 (Simple Policy) WINS!")
    else:
        print("\nGame ended without a winner (safety limit reached)")


if __name__ == "__main__":
    main()

```

```python
"""Configuration loader for policy generators."""

import json
import re
from pathlib import Path
from typing import Dict, Any

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.jsonc"
_config_cache: Dict[str, Any] = None


def strip_jsonc_comments(text: str) -> str:
    """
    Strip comments from JSONC (JSON with Comments) format.
    Removes:
    - Single-line comments (// ...)
    - Multi-line comments (/* ... */)
    """
    # Remove single-line comments
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    
    # Remove multi-line comments
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    
    return text


def load_config() -> Dict[str, Any]:
    """Load configuration from config.jsonc (JSON with Comments)."""
    global _config_cache
    if _config_cache is None:
        if not _CONFIG_PATH.exists():
            raise FileNotFoundError(f"Config file not found: {_CONFIG_PATH}")
        with open(_CONFIG_PATH, "r") as f:
            content = f.read()
        # Strip comments before parsing
        json_content = strip_jsonc_comments(content)
        _config_cache = json.loads(json_content)
    return _config_cache


def get_particle_policy_config() -> Dict[str, Any]:
    """Get configuration for particle policy (runtime decision-making)."""
    config = load_config()
    return config.get(
        "particle_policy",
        {
            "num_particles": 500,
            "mcts_iterations": 200,
            "planning_horizon": 5,
            "gamma": 0.95,
            "ucb_c": 1.414,
            "rollout_particle_sample_size": 20,
            "resample_threshold": 0.5,
        },
    )

```

```python
from src.uno import Uno, Action, card_to_string
from src.policy.particle_policy import ParticlePolicy
from src.utils.config_loader import get_particle_policy_config


def display_game_state(game: Uno, current_player: int, policy=None):
    """Display current game state (full visibility - God mode)."""
    print("\n" + "=" * 60)
    print(f"Player {current_player}'s Turn")
    print("=" * 60)
    game.create_S()
    state = game.State

    print(f"\nTop Card: {card_to_string(state[4]) if state[4] else 'None'}")
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


def choose_action_simple(game: Uno, player: int):
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


def main():
    """Run 2-player UNO game simulator.
    
    Player 1 uses ParticlePolicy, Player 2 uses simple policy.
    """
    print("=" * 60)
    print("UNO Simulation: ParticlePolicy (Player 1) vs Simple Policy (Player 2)")
    print("=" * 60)

    # Initialize game
    game = Uno()
    game.new_game(seed=42)  # Fixed seed for reproducibility

    # Initialize ParticlePolicy for Player 1
    config = get_particle_policy_config()
    policy = ParticlePolicy(**config)

    print(f"\nParticlePolicy Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    current_player = 1
    turn_count = 0
    max_turns = 1000  # Safety limit

    while game.G_o == "Active" and turn_count < max_turns:
        turn_count += 1

        # Check if we need to skip this player
        if game.skip_next:
            print(f"\n>>> Player {current_player} is skipped!")
            game.skip_next = False
            current_player = 3 - current_player  # Switch: 1->2, 2->1
            continue

        # Check if player needs to draw cards (from Draw 2 or Wild Draw 4)
        if game.draw_pending > 0:
            print(f"\n>>> Player {current_player} must draw {game.draw_pending} cards!")
            action = Action(n=game.draw_pending)
            game.execute_action(action, current_player)
            # After drawing, skip this player's turn
            game.skip_next = True
            display_game_state(game, current_player)
            current_player = 3 - current_player
            continue

        display_game_state(game, current_player, policy if current_player == 1 else None)

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

    if len(state[0]) == 0:
        print("\n🎉 Player 1 (ParticlePolicy) WINS!")
    elif len(state[1]) == 0:
        print("\n🎉 Player 2 (Simple Policy) WINS!")
    else:
        print("\nGame ended without a winner (safety limit reached)")


if __name__ == "__main__":
    main()

```


Checking for lint errors and verifying everything works:



Verifying the final state:



Cleanup complete:

1. Removed legacy `policy2` section from `config.jsonc`
2. Removed `get_policy2_config()` function from `config_loader.py`
3. Removed hardcoded parameter overrides in `main.py` — now uses values from `config.jsonc`
4. Updated config printing to show all parameters

All parameters now come from `config.jsonc`. The current values are:
- `num_particles`: 1000
- `mcts_iterations`: 2000
- `planning_horizon`: 5
- `rollout_particle_sample_size`: 20

Adjust these in `config.jsonc` as needed; the code will use them automatically.

---

**User**

is there a set random seed? if so then remove it

---

**Cursor**


Searching for random seed usage:



Removing fixed seeds in `main.py` and `policy2.py`:



Removing fixed seeds from both files:


```python
from src.uno import Uno, Action, card_to_string
from src.policy.particle_policy import ParticlePolicy
from src.utils.config_loader import get_particle_policy_config


def display_game_state(game: Uno, current_player: int, policy=None):
    """Display current game state (full visibility - God mode)."""
    print("\n" + "=" * 60)
    print(f"Player {current_player}'s Turn")
    print("=" * 60)
    game.create_S()
    state = game.State

    print(f"\nTop Card: {card_to_string(state[4]) if state[4] else 'None'}")
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


def choose_action_simple(game: Uno, player: int):
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


def main():
    """Run 2-player UNO game simulator.
    
    Player 1 uses ParticlePolicy, Player 2 uses simple policy.
    """
    print("=" * 60)
    print("UNO Simulation: ParticlePolicy (Player 1) vs Simple Policy (Player 2)")
    print("=" * 60)

    # Initialize game
    game = Uno()
    game.new_game()

    # Initialize ParticlePolicy for Player 1
    config = get_particle_policy_config()
    policy = ParticlePolicy(**config)

    print(f"\nParticlePolicy Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    current_player = 1
    turn_count = 0
    max_turns = 1000  # Safety limit

    while game.G_o == "Active" and turn_count < max_turns:
        turn_count += 1

        # Check if we need to skip this player
        if game.skip_next:
            print(f"\n>>> Player {current_player} is skipped!")
            game.skip_next = False
            current_player = 3 - current_player  # Switch: 1->2, 2->1
            continue

        # Check if player needs to draw cards (from Draw 2 or Wild Draw 4)
        if game.draw_pending > 0:
            print(f"\n>>> Player {current_player} must draw {game.draw_pending} cards!")
            action = Action(n=game.draw_pending)
            game.execute_action(action, current_player)
            # After drawing, skip this player's turn
            game.skip_next = True
            display_game_state(game, current_player)
            current_player = 3 - current_player
            continue

        display_game_state(game, current_player, policy if current_player == 1 else None)

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

    if len(state[0]) == 0:
        print("\n🎉 Player 1 (ParticlePolicy) WINS!")
    elif len(state[1]) == 0:
        print("\n🎉 Player 2 (Simple Policy) WINS!")
    else:
        print("\nGame ended without a winner (safety limit reached)")


if __name__ == "__main__":
    main()

```

```python
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

```


Removed fixed seeds from both files. Games will now use random seeds each run. Verifying no issues:



Removed fixed seeds from `main.py` and `policy2.py`. Each run now uses a random seed, so games will vary.

---

**User**

run uvx ruff check and format respectively then commit all changes to this branch im on now

---

**Cursor**


Checking the current branch, then running ruff check and format, then committing:



Fixing issues automatically, then formatting:



Checking git status, then committing all changes:



Checking if `config.jsonc` needs to be committed:



Completed:

1. Ran `uvx ruff check` — found 7 fixable issues
2. Fixed them with `uvx ruff check --fix`
3. Formatted code with `uvx ruff format` — 4 files reformatted
4. Committed all changes to branch `sk/finalization`

All changes are committed. The working tree is clean.

---

**User**

add a test.py file along with github action ci.yml that tests basic game functionality and ensures the core game logic of uno rules works. you don't have to test the policy or whatever, these tests should focus purely on uno game logic implementaitons and whether the rules are correctly implemented or not

(for a 2 player uno game where if you have no legal moves you draw 1 card and turn ends)

actually nevermind, instead of test.py, use an actual python

---

**Cursor**


Reviewing the game implementation to understand the UNO rules, then drafting a test plan.



Checking project structure and dependencies:



Creating a plan for `test.py` and a GitHub Actions CI workflow:


---

**User**

use an actual python testing frameowrk like pytest, then you can create <thing>.test.py files inline next to the actual uno game logic, then run pytest in the ci.,yml to verify

---

**Cursor**


Updating the plan to use inline test files next to the game logic files:



Creating a new plan with the correct structure:


---

**User**

ensure you use uvx pytest or similar or whatever is best im not 100% sure about syntax

---

**Cursor**


Updating the plan to use `uvx pytest` for running tests:


---

**User**

double check the plan making sure that the new tests are specifically around core uno game logic for our 2 player uno game not around policy or anything else

---

**Cursor**


Reviewing the plan to confirm it focuses only on core UNO game logic:



Confirming the plan focuses only on core UNO game logic. Adding a clarifying note and updating CI to use `uvx pytest`:


---

**User**

continue

---

**Cursor**


Executing the plan. Creating inline test files for core UNO game logic and the CI workflow:


```python
"""Tests for UNO card building and deck construction."""

import pytest
from src.uno.cards import (
    build_full_deck,
    card_to_string,
    RED,
    YELLOW,
    GREEN,
    BLUE,
    SKIP,
    REVERSE,
    DRAW_2,
    WILD,
    WILD_DRAW_4,
)


def test_deck_size():
    """Verify full deck has exactly 108 cards."""
    deck = build_full_deck()
    assert len(deck) == 108


def test_deck_distribution():
    """Verify correct distribution of card types."""
    deck = build_full_deck()
    
    # Count card types
    number_cards = sum(1 for _, value in deck if isinstance(value, int))
    skip_cards = sum(1 for _, value in deck if value == SKIP)
    reverse_cards = sum(1 for _, value in deck if value == REVERSE)
    draw_2_cards = sum(1 for _, value in deck if value == DRAW_2)
    wild_cards = sum(1 for _, value in deck if value == WILD)
    wild_draw_4_cards = sum(1 for _, value in deck if value == WILD_DRAW_4)
    
    # Verify counts
    assert number_cards == 76  # 4 colors * (1 zero + 2 each of 1-9) = 4 * 19 = 76
    assert skip_cards == 8  # 2 per color * 4 colors = 8
    assert reverse_cards == 8  # 2 per color * 4 colors = 8
    assert draw_2_cards == 
```

---

**User**

Implement the plan as specified, it is attached for your reference. Do NOT edit the plan file itself.

To-do's from the plan have already been created. Do not create them again. Mark them as in_progress as you work, starting with the first one. Don't stop until you have completed all the to-dos.

---

**Cursor**


Implementing the plan. Starting with the first todo: creating the cards test file.


```python
"""Tests for UNO card and deck building logic."""

import pytest
from src.uno.cards import (
    build_full_deck,
    card_to_string,
    RED,
    YELLOW,
    GREEN,
    BLUE,
    SKIP,
    REVERSE,
    DRAW_2,
    WILD,
    WILD_DRAW_4,
    BLACK,
)


def test_deck_size():
    """Verify full deck has exactly 108 cards."""
    deck = build_full_deck()
    assert len(deck) == 108


def test_deck_distribution():
    """Verify correct distribution of card types."""
    deck = build_full_deck()
    
    number_cards = [c for c in deck if isinstance(c[1], int)]
    skip_cards = [c for c in deck if c[1] == SKIP]
    reverse_cards = [c for c in deck if c[1] == REVERSE]
    draw_2_cards = [c for c in deck if c[1] == DRAW_2]
    wild_cards = [c for c in deck if c[1] == WILD]
    wild_draw_4_cards = [c for c in deck if c[1] == WILD_DRAW_4]
    
    assert len(number_cards) == 76  # 4 colors * (1 zero + 2 each of 1-9)
    assert len(skip_cards) == 8  # 2 per color * 4 colors
    assert len(reverse_cards) == 8  # 2 per color * 4 colors
    assert len(draw_2_cards) == 8  # 2 per color * 4 colors
    assert len(wild_cards) == 4
    assert len(wild_draw_4_cards) == 4


def test_number_card_distribution():
    """Verify one 0 per color, two of each 1-9 per color."""
    deck = build_full_deck()
    colors = [RED, YELLOW, GREEN, BLUE]
    
    for color in colors:
        zeros = [c for c in deck if c == (color, 0)]
        assert len(zeros) == 1, f"Should have exactly 1 zero card in {color}"
        
        for number in range(1, 10):
            number_cards = [c for c in deck if c == (color, number)]
            assert len(number_cards) == 2, f"Should have exactly 2 {color}{number} cards"


def test_special_card_distribution():
    """Verify special cards (Skip, Reverse, Draw 2) have 2 per color."""
    deck = build_full_deck()
    colors = [RED, YELLOW, GREEN, BLUE]
    special_types = [SKIP, REVERSE, DRAW_2]
    
    for color in colors:
        for special_type in special_types:
            cards = [c for c in deck if c == (color, special_type)]
            assert len(cards) == 2, f"Should have exactly 2 {color} {special_type} cards"


def test_wild_card_distribution():
    """Verify Wild cards have no color and correct count."""
    deck = build_full_deck()
    
    wild_cards = [c for c in deck if c[1] == WILD]
    wild_draw_4_cards = [c for c in deck if c[1] == WILD_DRAW_4]
    
    assert len(wild_cards) == 4
    assert len(wild_draw_4_cards) == 4
    
    # All wild cards should have BLACK (None) color
    for card in wild_cards + wild_draw_4_cards:
        assert card[0] == BLACK, "Wild cards should have BLACK (None) color"


def test_card_to_string():
    """Test card_to_string() function for various card types."""
    # Number card
    assert card_to_string((RED, 5)) == "R5"
    assert card_to_string((YELLOW, 0)) == "Y0"
    
    # Special cards
    assert card_to_string((GREEN, SKIP)) == "G_SKIP"
    assert card_to_string((BLUE, REVERSE)) == "B_REVERSE"
    assert card_to_string((RED, DRAW_2)) == "R_DRAW_2"
    
    # Wild cards
    assert card_to_string((BLACK, WILD)) == "WILD"
    assert card_to_string((BLACK, WILD_DRAW_4)) == "WILD_DRAW_4"

```


Creating the game test file:


```python
"""Tests for UNO game logic and rules."""

import pytest
from src.uno.game import Uno
from src.uno.cards import (
    RED,
    YELLOW,
    GREEN,
    BLUE,
    SKIP,
    REVERSE,
    DRAW_2,
    WILD,
    WILD_DRAW_4,
    BLACK,
)
from src.uno.state import Action


# Helper functions for test setup
def create_game_with_state(H_1, H_2, top_card, deck):
    """Helper to create game with specific state."""
    game = Uno(H_1=H_1, H_2=H_2, D_g=deck, P=[top_card])
    game.current_color = top_card[0] if top_card[0] is not None else RED
    game.create_S()
    return game


# Game Initialization Tests
def test_game_initialization():
    """Verify 7 cards dealt to each player, initial top card is non-Wild, game state is Active."""
    game = Uno()
    game.new_game(seed=42)
    
    assert len(game.H_1) == 7, "Player 1 should have 7 cards"
    assert len(game.H_2) == 7, "Player 2 should have 7 cards"
    assert len(game.P) == 1, "Should have one card in played pile"
    assert game.P_t is not None, "Top card should exist"
    assert game.P_t[0] is not None, "Top card should not be Wild (should have color)"
    assert len(game.D_g) > 0, "Deck should have remaining cards"
    assert game.G_o == "Active", "Game should be Active"
    assert game.current_color is not None, "Current color should be set"


def test_game_initialization_state_creation():
    """Verify state creation works correctly."""
    game = Uno()
    game.new_game(seed=42)
    
    state = game.create_S()
    assert state is not None
    assert len(state[0]) == 7  # H_1
    assert len(state[1]) == 7  # H_2
    assert len(state[2]) > 0  # D_g
    assert len(state[3]) == 1  # P
    assert state[4] is not None  # P_t
    assert state[5] == "Active"  # G_o


# Legal Play Rules Tests
def test_legal_play_color_match():
    """Test color matching (same color as top card)."""
    game = create_game_with_state(
        H_1=[(RED, 5), (YELLOW, 3)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    assert game.is_legal_play((RED, 5)) is True, "Red card should match red top card"
    assert game.is_legal_play((YELLOW, 3)) is False, "Yellow card should not match red top card"


def test_legal_play_value_match():
    """Test value matching (same value as top card)."""
    game = create_game_with_state(
        H_1=[(RED, 5), (YELLOW, 5)],
        H_2=[],
        top_card=(GREEN, 5),
        deck=[]
    )
    
    assert game.is_legal_play((RED, 5)) is True, "Same value should match"
    assert game.is_legal_play((YELLOW, 5)) is True, "Same value should match"


def test_legal_play_wild_always_legal():
    """Test Wild cards are always legal."""
    game = create_game_with_state(
        H_1=[(BLACK, WILD), (BLACK, WILD_DRAW_4)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    assert game.is_legal_play((BLACK, WILD)) is True, "Wild should always be legal"
    assert game.is_legal_play((BLACK, WILD_DRAW_4)) is True, "Wild Draw 4 should always be legal"


def test_legal_play_illegal_rejected():
    """Test illegal plays are rejected."""
    game = create_game_with_state(
        H_1=[(YELLOW, 3), (GREEN, 8)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    assert game.is_legal_play((YELLOW, 3)) is False, "Different color and value should be illegal"
    assert game.is_legal_play((GREEN, 8)) is False, "Different color and value should be illegal"


def test_legal_play_wild_color_handling():
    """Test current_color handling for Wild cards."""
    game = create_game_with_state(
        H_1=[(RED, 5)],
        H_2=[],
        top_card=(BLACK, WILD),
        deck=[]
    )
    game.current_color = BLUE
    
    assert game.is_legal_play((RED, 5)) is False, "Should not match when current_color is BLUE"
    
    game.current_color = RED
    assert game.is_legal_play((RED, 5)) is True, "Should match when current_color is RED"


def test_get_legal_actions():
    """Test get_legal_actions returns correct actions."""
    game = create_game_with_state(
        H_1=[(RED, 5), (YELLOW, 3)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    legal_actions = game.get_legal_actions(1)
    assert len(legal_actions) == 1, "Should have one legal play (RED 5)"
    assert legal_actions[0].is_play() is True
    assert legal_actions[0].X_1 == (RED, 5)


def test_get_legal_actions_no_legal_plays():
    """Test get_legal_actions returns draw action when no legal plays."""
    game = create_game_with_state(
        H_1=[(YELLOW, 3), (GREEN, 8)],
        H_2=[],
        top_card=(RED, 7),
        deck=[(BLUE, 1)]
    )
    
    legal_actions = game.get_legal_actions(1)
    assert len(legal_actions) == 1, "Should have one action (draw)"
    assert legal_actions[0].is_draw() is True
    assert legal_actions[0].n == 1


# Special Card Effects Tests
def test_special_card_skip():
    """Test SKIP: Player plays again (2-player rule)."""
    game = create_game_with_state(
        H_1=[(RED, SKIP)],
        H_2=[],
        top_card=(RED, 5),
        deck=[]
    )
    
    action = Action(X_1=(RED, SKIP))
    result = game.execute_action(action, 1)
    
    assert result is True
    assert game.player_plays_again is True, "Player should play again after Skip"
    assert len(game.H_1) == 0, "Card should be removed from hand"


def test_special_card_reverse():
    """Test REVERSE: Player plays again (2-player rule, acts like Skip)."""
    game = create_game_with_state(
        H_1=[(RED, REVERSE)],
        H_2=[],
        top_card=(RED, 5),
        deck=[]
    )
    
    action = Action(X_1=(RED, REVERSE))
    result = game.execute_action(action, 1)
    
    assert result is True
    assert game.player_plays_again is True, "Player should play again after Reverse (2-player rule)"


def test_special_card_draw_2():
    """Test DRAW_2: Next player draws 2 and skips turn."""
    game = create_game_with_state(
        H_1=[(RED, DRAW_2)],
        H_2=[(YELLOW, 1)],
        top_card=(RED, 5),
        deck=[(BLUE, 2), (GREEN, 3)]
    )
    
    action = Action(X_1=(RED, DRAW_2))
    result = game.execute_action(action, 1)
    
    assert result is True
    assert game.draw_pending == 2, "Next player should draw 2 cards"


def test_special_card_wild():
    """Test WILD: Choose color, current_color updated."""
    game = create_game_with_state(
        H_1=[(BLACK, WILD)],
        H_2=[],
        top_card=(RED, 5),
        deck=[]
    )
    
    action = Action(X_1=(BLACK, WILD), wild_color=BLUE)
    result = game.execute_action(action, 1)
    
    assert result is True
    assert game.current_color == BLUE, "Current color should be set to chosen color"
    assert len(game.H_1) == 0, "Card should be removed from hand"


def test_special_card_wild_draw_4():
    """Test WILD_DRAW_4: Next player draws 4 and skips turn."""
    game = create_game_with_state(
        H_1=[(BLACK, WILD_DRAW_4)],
        H_2=[(YELLOW, 1)],
        top_card=(RED, 5),
        deck=[(BLUE, 2), (GREEN, 3), (RED, 4), (YELLOW, 5)]
    )
    
    action = Action(X_1=(BLACK, WILD_DRAW_4), wild_color=GREEN)
    result = game.execute_action(action, 1)
    
    assert result is True
    assert game.current_color == GREEN, "Current color should be set"
    assert game.draw_pending == 4, "Next player should draw 4 cards"


# Draw Rules Tests
def test_draw_rules_no_legal_plays():
    """Test when no legal plays, must draw 1 card."""
    game = create_game_with_state(
        H_1=[(YELLOW, 3)],
        H_2=[],
        top_card=(RED, 7),
        deck=[(BLUE, 1)]
    )
    
    legal_actions = game.get_legal_actions(1)
    assert len(legal_actions) == 1
    assert legal_actions[0].is_draw() is True
    assert legal_actions[0].n == 1


def test_draw_action_adds_cards():
    """Test draw action properly adds cards to hand."""
    game = create_game_with_state(
        H_1=[(YELLOW, 3)],
        H_2=[],
        top_card=(RED, 7),
        deck=[(BLUE, 1), (GREEN, 2)]
    )
    
    initial_hand_size = len(game.H_1)
    action = Action(n=1)
    result = game.execute_action(action, 1)
    
    assert result is True
    assert len(game.H_1) == initial_hand_size + 1, "Hand should have one more card"
    assert len(action.Y_n) == 1, "Action should record drawn cards"


def test_draw_pending_cards():
    """Test draw pending cards (from DRAW_2/WILD_DRAW_4)."""
    game = create_game_with_state(
        H_1=[],
        H_2=[(YELLOW, 1)],
        top_card=(RED, DRAW_2),
        deck=[(BLUE, 1), (GREEN, 2)]
    )
    game.draw_pending = 2
    
    action = Action(n=1)  # Player tries to draw 1, but should draw 2
    result = game.execute_action(action, 2)
    
    assert result is True
    assert len(game.H_2) == 3, "Should have drawn 2 cards (1 original + 2 pending)"
    assert game.draw_pending == 0, "Draw pending should be cleared"


# Game End Tests
def test_game_end_player_has_zero_cards():
    """Test game ends when player has 0 cards."""
    game = create_game_with_state(
        H_1=[(RED, 5)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    action = Action(X_1=(RED, 5))
    game.execute_action(action, 1)
    
    assert len(game.H_1) == 0, "Player 1 should have 0 cards"
    assert game.G_o == "GameOver", "Game should be over"


def test_game_end_no_actions_after_game_over():
    """Test no actions allowed after game over."""
    game = create_game_with_state(
        H_1=[],
        H_2=[(YELLOW, 1)],
        top_card=(RED, 7),
        deck=[]
    )
    game.G_o = "GameOver"
    
    action = Action(X_1=(YELLOW, 1))
    result = game.execute_action(action, 2)
    
    assert result is False, "Should not allow actions after game over"


# Deck Reshuffling Tests
def test_deck_reshuffle_when_empty():
    """Test when draw pile empty, reshuffle played pile (except top card)."""
    game = create_game_with_state(
        H_1=[(YELLOW, 3)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]  # Empty deck
    )
    # Add some cards to played pile
    game.P = [(RED, 7), (BLUE, 1), (GREEN, 2), (YELLOW, 3)]
    
    action = Action(n=1)
    game.execute_action(action, 1)
    
    assert len(game.D_g) > 0, "Deck should have cards after reshuffle"
    assert len(game.P) == 1, "Played pile should only have top card"
    assert game.P[0] == (RED, 7), "Top card should remain"


def test_draw_after_reshuffle():
    """Test cards can still be drawn after reshuffle."""
    game = create_game_with_state(
        H_1=[(YELLOW, 3)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]  # Empty deck
    )
    game.P = [(RED, 7), (BLUE, 1), (GREEN, 2)]
    
    initial_hand_size = len(game.H_1)
    action = Action(n=1)
    result = game.execute_action(action, 1)
    
    assert result is True
    assert len(game.H_1) == initial_hand_size + 1, "Should have drawn a card after reshuffle"


# Action Execution Tests
def test_action_execution_valid_play():
    """Test valid actions execute successfully."""
    game = create_game_with_state(
        H_1=[(RED, 5)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    initial_hand_size = len(game.H_1)
    initial_pile_size = len(game.P)
    
    action = Action(X_1=(RED, 5))
    result = game.execute_action(action, 1)
    
    assert result is True
    assert len(game.H_1) == initial_hand_size - 1, "Card should be removed from hand"
    assert len(game.P) == initial_pile_size + 1, "Card should be added to played pile"


def test_action_execution_invalid_card_not_in_hand():
    """Test invalid actions are rejected (card not in hand)."""
    game = create_game_with_state(
        H_1=[(RED, 5)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    action = Action(X_1=(YELLOW, 3))  # Not in hand
    result = game.execute_action(action, 1)
    
    assert result is False, "Should reject card not in hand"


def test_action_execution_invalid_illegal_play():
    """Test invalid actions are rejected (illegal play)."""
    game = create_game_with_state(
        H_1=[(YELLOW, 3)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    action = Action(X_1=(YELLOW, 3))  # Illegal play
    result = game.execute_action(action, 1)
    
    assert result is False, "Should reject illegal play"


def test_action_execution_player_2():
    """Test action execution for Player 2."""
    game = create_game_with_state(
        H_1=[],
        H_2=[(RED, 5)],
        top_card=(RED, 7),
        deck=[]
    )
    
    action = Action(X_1=(RED, 5))
    result = game.execute_action(action, 2)
    
    assert result is True
    assert len(game.H_2) == 0, "Card should be removed from Player 2's hand"
    assert len(game.P) == 2, "Card should be added to played pile"

```

```python
"""Tests for UNO state and Action class."""

import pytest
from src.uno.state import Action
from src.uno.cards import RED, YELLOW, BLACK, WILD


def test_action_validation_must_specify_one():
    """Test Action validation: must specify X_1 OR n, not both."""
    # Valid: only X_1
    action1 = Action(X_1=(RED, 5))
    assert action1.is_play() is True
    assert action1.is_draw() is False
    
    # Valid: only n
    action2 = Action(n=1)
    assert action2.is_play() is False
    assert action2.is_draw() is True
    
    # Invalid: both specified
    with pytest.raises(ValueError, match="Must specify EITHER"):
        Action(X_1=(RED, 5), n=1)
    
    # Invalid: neither specified
    with pytest.raises(ValueError, match="Must specify EITHER"):
        Action()


def test_action_validation_draw_n_values():
    """Test draw action validation: n must be in {1, 2, 4}."""
    # Valid values
    Action(n=1)
    Action(n=2)
    Action(n=4)
    
    # Invalid values
    with pytest.raises(ValueError, match="n must be in"):
        Action(n=0)
    
    with pytest.raises(ValueError, match="n must be in"):
        Action(n=3)
    
    with pytest.raises(ValueError, match="n must be in"):
        Action(n=5)


def test_action_type_checking():
    """Test action type checking (is_play, is_draw)."""
    play_action = Action(X_1=(RED, 5))
    assert play_action.is_play() is True
    assert play_action.is_draw() is False
    
    draw_action = Action(n=1)
    assert draw_action.is_play() is False
    assert draw_action.is_draw() is True


def test_action_representation_play():
    """Test Action representation for play actions."""
    action = Action(X_1=(RED, 5))
    repr_str = repr(action)
    assert "X_1" in repr_str
    assert "RED" in repr_str or "5" in repr_str
    
    # With wild color
    wild_action = Action(X_1=(BLACK, WILD), wild_color=YELLOW)
    repr_str = repr(wild_action)
    assert "wild_color" in repr_str
    assert "YELLOW" in repr_str or "Y" in repr_str


def test_action_representation_draw():
    """Test Action representation for draw actions."""
    action = Action(n=1)
    repr_str = repr(action)
    assert "Y_1" in repr_str or "n=1" in repr_str
    
    # After drawing (Y_n filled)
    action.Y_n = [(RED, 5)]
    repr_str = repr(action)
    assert "Y_1" in repr_str or "Y_n" in repr_str


def test_action_wild_color():
    """Test Action with wild_color parameter."""
    action = Action(X_1=(BLACK, WILD), wild_color=RED)
    assert action.wild_color == RED
    assert action.X_1 == (BLACK, WILD)

```

```yaml
name: CI

on:
  push:
    branches: ["**"]
  pull_request:
    branches: ["**"]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: "latest"
      
      - name: Set up Python
        run: uv python install 3.13
      
      - name: Run tests
        run: uvx pytest -v

```


Pytest didn't discover the tests. Checking pytest configuration and renaming test files to the standard `test_*.py` format:



Fixing imports to use relative imports:


```python
"""Tests for UNO card and deck building logic."""

import pytest
from .cards import (
    build_full_deck,
    card_to_string,
    RED,
    YELLOW,
    GREEN,
    BLUE,
    SKIP,
    REVERSE,
    DRAW_2,
    WILD,
    WILD_DRAW_4,
    BLACK,
)


def test_deck_size():
    """Verify full deck has exactly 108 cards."""
    deck = build_full_deck()
    assert len(deck) == 108


def test_deck_distribution():
    """Verify correct distribution of card types."""
    deck = build_full_deck()
    
    number_cards = [c for c in deck if isinstance(c[1], int)]
    skip_cards = [c for c in deck if c[1] == SKIP]
    reverse_cards = [c for c in deck if c[1] == REVERSE]
    draw_2_cards = [c for c in deck if c[1] == DRAW_2]
    wild_cards = [c for c in deck if c[1] == WILD]
    wild_draw_4_cards = [c for c in deck if c[1] == WILD_DRAW_4]
    
    assert len(number_cards) == 76  # 4 colors * (1 zero + 2 each of 1-9)
    assert len(skip_cards) == 8  # 2 per color * 4 colors
    assert len(reverse_cards) == 8  # 2 per color * 4 colors
    assert len(draw_2_cards) == 8  # 2 per color * 4 colors
    assert len(wild_cards) == 4
    assert len(wild_draw_4_cards) == 4


def test_number_card_distribution():
    """Verify one 0 per color, two of each 1-9 per color."""
    deck = build_full_deck()
    colors = [RED, YELLOW, GREEN, BLUE]
    
    for color in colors:
        zeros = [c for c in deck if c == (color, 0)]
        assert len(zeros) == 1, f"Should have exactly 1 zero card in {color}"
        
        for number in range(1, 10):
            number_cards = [c for c in deck if c == (color, number)]
            assert len(number_cards) == 2, f"Should have exactly 2 {color}{number} cards"


def test_special_card_distribution():
    """Verify special cards (Skip, Reverse, Draw 2) have 2 per color."""
    deck = build_full_deck()
    colors = [RED, YELLOW, GREEN, BLUE]
    special_types = [SKIP, REVERSE, DRAW_2]
    
    for color in colors:
        for special_type in special_types:
            cards = [c for c in deck if c == (color, special_type)]
            assert len(cards) == 2, f"Should have exactly 2 {color} {special_type} cards"


def test_wild_card_distribution():
    """Verify Wild cards have no color and correct count."""
    deck = build_full_deck()
    
    wild_cards = [c for c in deck if c[1] == WILD]
    wild_draw_4_cards = [c for c in deck if c[1] == WILD_DRAW_4]
    
    assert len(wild_cards) == 4
    assert len(wild_draw_4_cards) == 4
    
    # All wild cards should have BLACK (None) color
    for card in wild_cards + wild_draw_4_cards:
        assert card[0] == BLACK, "Wild cards should have BLACK (None) color"


def test_card_to_string():
    """Test card_to_string() function for various card types."""
    # Number card
    assert card_to_string((RED, 5)) == "R5"
    assert card_to_string((YELLOW, 0)) == "Y0"
    
    # Special cards
    assert card_to_string((GREEN, SKIP)) == "G_SKIP"
    assert card_to_string((BLUE, REVERSE)) == "B_REVERSE"
    assert card_to_string((RED, DRAW_2)) == "R_DRAW_2"
    
    # Wild cards
    assert card_to_string((BLACK, WILD)) == "WILD"
    assert card_to_string((BLACK, WILD_DRAW_4)) == "WILD_DRAW_4"


```

```python
"""Tests for UNO game logic and rules."""

import pytest
from .game import Uno
from .cards import (
    RED,
    YELLOW,
    GREEN,
    BLUE,
    SKIP,
    REVERSE,
    DRAW_2,
    WILD,
    WILD_DRAW_4,
    BLACK,
)
from src.uno.state import Action


# Helper functions for test setup
def create_game_with_state(H_1, H_2, top_card, deck):
    """Helper to create game with specific state."""
    game = Uno(H_1=H_1, H_2=H_2, D_g=deck, P=[top_card])
    game.current_color = top_card[0] if top_card[0] is not None else RED
    game.create_S()
    return game


# Game Initialization Tests
def test_game_initialization():
    """Verify 7 cards dealt to each player, initial top card is non-Wild, game state is Active."""
    game = Uno()
    game.new_game(seed=42)
    
    assert len(game.H_1) == 7, "Player 1 should have 7 cards"
    assert len(game.H_2) == 7, "Player 2 should have 7 cards"
    assert len(game.P) == 1, "Should have one card in played pile"
    assert game.P_t is not None, "Top card should exist"
    assert game.P_t[0] is not None, "Top card should not be Wild (should have color)"
    assert len(game.D_g) > 0, "Deck should have remaining cards"
    assert game.G_o == "Active", "Game should be Active"
    assert game.current_color is not None, "Current color should be set"


def test_game_initialization_state_creation():
    """Verify state creation works correctly."""
    game = Uno()
    game.new_game(seed=42)
    
    state = game.create_S()
    assert state is not None
    assert len(state[0]) == 7  # H_1
    assert len(state[1]) == 7  # H_2
    assert len(state[2]) > 0  # D_g
    assert len(state[3]) == 1  # P
    assert state[4] is not None  # P_t
    assert state[5] == "Active"  # G_o


# Legal Play Rules Tests
def test_legal_play_color_match():
    """Test color matching (same color as top card)."""
    game = create_game_with_state(
        H_1=[(RED, 5), (YELLOW, 3)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    assert game.is_legal_play((RED, 5)) is True, "Red card should match red top card"
    assert game.is_legal_play((YELLOW, 3)) is False, "Yellow card should not match red top card"


def test_legal_play_value_match():
    """Test value matching (same value as top card)."""
    game = create_game_with_state(
        H_1=[(RED, 5), (YELLOW, 5)],
        H_2=[],
        top_card=(GREEN, 5),
        deck=[]
    )
    
    assert game.is_legal_play((RED, 5)) is True, "Same value should match"
    assert game.is_legal_play((YELLOW, 5)) is True, "Same value should match"


def test_legal_play_wild_always_legal():
    """Test Wild cards are always legal."""
    game = create_game_with_state(
        H_1=[(BLACK, WILD), (BLACK, WILD_DRAW_4)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    assert game.is_legal_play((BLACK, WILD)) is True, "Wild should always be legal"
    assert game.is_legal_play((BLACK, WILD_DRAW_4)) is True, "Wild Draw 4 should always be legal"


def test_legal_play_illegal_rejected():
    """Test illegal plays are rejected."""
    game = create_game_with_state(
        H_1=[(YELLOW, 3), (GREEN, 8)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    assert game.is_legal_play((YELLOW, 3)) is False, "Different color and value should be illegal"
    assert game.is_legal_play((GREEN, 8)) is False, "Different color and value should be illegal"


def test_legal_play_wild_color_handling():
    """Test current_color handling for Wild cards."""
    game = create_game_with_state(
        H_1=[(RED, 5)],
        H_2=[],
        top_card=(BLACK, WILD),
        deck=[]
    )
    game.current_color = BLUE
    
    assert game.is_legal_play((RED, 5)) is False, "Should not match when current_color is BLUE"
    
    game.current_color = RED
    assert game.is_legal_play((RED, 5)) is True, "Should match when current_color is RED"


def test_get_legal_actions():
    """Test get_legal_actions returns correct actions."""
    game = create_game_with_state(
        H_1=[(RED, 5), (YELLOW, 3)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    legal_actions = game.get_legal_actions(1)
    assert len(legal_actions) == 1, "Should have one legal play (RED 5)"
    assert legal_actions[0].is_play() is True
    assert legal_actions[0].X_1 == (RED, 5)


def test_get_legal_actions_no_legal_plays():
    """Test get_legal_actions returns draw action when no legal plays."""
    game = create_game_with_state(
        H_1=[(YELLOW, 3), (GREEN, 8)],
        H_2=[],
        top_card=(RED, 7),
        deck=[(BLUE, 1)]
    )
    
    legal_actions = game.get_legal_actions(1)
    assert len(legal_actions) == 1, "Should have one action (draw)"
    assert legal_actions[0].is_draw() is True
    assert legal_actions[0].n == 1


# Special Card Effects Tests
def test_special_card_skip():
    """Test SKIP: Player plays again (2-player rule)."""
    game = create_game_with_state(
        H_1=[(RED, SKIP)],
        H_2=[],
        top_card=(RED, 5),
        deck=[]
    )
    
    action = Action(X_1=(RED, SKIP))
    result = game.execute_action(action, 1)
    
    assert result is True
    assert game.player_plays_again is True, "Player should play again after Skip"
    assert len(game.H_1) == 0, "Card should be removed from hand"


def test_special_card_reverse():
    """Test REVERSE: Player plays again (2-player rule, acts like Skip)."""
    game = create_game_with_state(
        H_1=[(RED, REVERSE)],
        H_2=[],
        top_card=(RED, 5),
        deck=[]
    )
    
    action = Action(X_1=(RED, REVERSE))
    result = game.execute_action(action, 1)
    
    assert result is True
    assert game.player_plays_again is True, "Player should play again after Reverse (2-player rule)"


def test_special_card_draw_2():
    """Test DRAW_2: Next player draws 2 and skips turn."""
    game = create_game_with_state(
        H_1=[(RED, DRAW_2)],
        H_2=[(YELLOW, 1)],
        top_card=(RED, 5),
        deck=[(BLUE, 2), (GREEN, 3)]
    )
    
    action = Action(X_1=(RED, DRAW_2))
    result = game.execute_action(action, 1)
    
    assert result is True
    assert game.draw_pending == 2, "Next player should draw 2 cards"


def test_special_card_wild():
    """Test WILD: Choose color, current_color updated."""
    game = create_game_with_state(
        H_1=[(BLACK, WILD)],
        H_2=[],
        top_card=(RED, 5),
        deck=[]
    )
    
    action = Action(X_1=(BLACK, WILD), wild_color=BLUE)
    result = game.execute_action(action, 1)
    
    assert result is True
    assert game.current_color == BLUE, "Current color should be set to chosen color"
    assert len(game.H_1) == 0, "Card should be removed from hand"


def test_special_card_wild_draw_4():
    """Test WILD_DRAW_4: Next player draws 4 and skips turn."""
    game = create_game_with_state(
        H_1=[(BLACK, WILD_DRAW_4)],
        H_2=[(YELLOW, 1)],
        top_card=(RED, 5),
        deck=[(BLUE, 2), (GREEN, 3), (RED, 4), (YELLOW, 5)]
    )
    
    action = Action(X_1=(BLACK, WILD_DRAW_4), wild_color=GREEN)
    result = game.execute_action(action, 1)
    
    assert result is True
    assert game.current_color == GREEN, "Current color should be set"
    assert game.draw_pending == 4, "Next player should draw 4 cards"


# Draw Rules Tests
def test_draw_rules_no_legal_plays():
    """Test when no legal plays, must draw 1 card."""
    game = create_game_with_state(
        H_1=[(YELLOW, 3)],
        H_2=[],
        top_card=(RED, 7),
        deck=[(BLUE, 1)]
    )
    
    legal_actions = game.get_legal_actions(1)
    assert len(legal_actions) == 1
    assert legal_actions[0].is_draw() is True
    assert legal_actions[0].n == 1


def test_draw_action_adds_cards():
    """Test draw action properly adds cards to hand."""
    game = create_game_with_state(
        H_1=[(YELLOW, 3)],
        H_2=[],
        top_card=(RED, 7),
        deck=[(BLUE, 1), (GREEN, 2)]
    )
    
    initial_hand_size = len(game.H_1)
    action = Action(n=1)
    result = game.execute_action(action, 1)
    
    assert result is True
    assert len(game.H_1) == initial_hand_size + 1, "Hand should have one more card"
    assert len(action.Y_n) == 1, "Action should record drawn cards"


def test_draw_pending_cards():
    """Test draw pending cards (from DRAW_2/WILD_DRAW_4)."""
    game = create_game_with_state(
        H_1=[],
        H_2=[(YELLOW, 1)],
        top_card=(RED, DRAW_2),
        deck=[(BLUE, 1), (GREEN, 2)]
    )
    game.draw_pending = 2
    
    action = Action(n=1)  # Player tries to draw 1, but should draw 2
    result = game.execute_action(action, 2)
    
    assert result is True
    assert len(game.H_2) == 3, "Should have drawn 2 cards (1 original + 2 pending)"
    assert game.draw_pending == 0, "Draw pending should be cleared"


# Game End Tests
def test_game_end_player_has_zero_cards():
    """Test game ends when player has 0 cards."""
    game = create_game_with_state(
        H_1=[(RED, 5)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    action = Action(X_1=(RED, 5))
    game.execute_action(action, 1)
    
    assert len(game.H_1) == 0, "Player 1 should have 0 cards"
    assert game.G_o == "GameOver", "Game should be over"


def test_game_end_no_actions_after_game_over():
    """Test no actions allowed after game over."""
    game = create_game_with_state(
        H_1=[],
        H_2=[(YELLOW, 1)],
        top_card=(RED, 7),
        deck=[]
    )
    game.G_o = "GameOver"
    
    action = Action(X_1=(YELLOW, 1))
    result = game.execute_action(action, 2)
    
    assert result is False, "Should not allow actions after game over"


# Deck Reshuffling Tests
def test_deck_reshuffle_when_empty():
    """Test when draw pile empty, reshuffle played pile (except top card)."""
    game = create_game_with_state(
        H_1=[(YELLOW, 3)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]  # Empty deck
    )
    # Add some cards to played pile
    game.P = [(RED, 7), (BLUE, 1), (GREEN, 2), (YELLOW, 3)]
    
    action = Action(n=1)
    game.execute_action(action, 1)
    
    assert len(game.D_g) > 0, "Deck should have cards after reshuffle"
    assert len(game.P) == 1, "Played pile should only have top card"
    assert game.P[0] == (RED, 7), "Top card should remain"


def test_draw_after_reshuffle():
    """Test cards can still be drawn after reshuffle."""
    game = create_game_with_state(
        H_1=[(YELLOW, 3)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]  # Empty deck
    )
    game.P = [(RED, 7), (BLUE, 1), (GREEN, 2)]
    
    initial_hand_size = len(game.H_1)
    action = Action(n=1)
    result = game.execute_action(action, 1)
    
    assert result is True
    assert len(game.H_1) == initial_hand_size + 1, "Should have drawn a card after reshuffle"


# Action Execution Tests
def test_action_execution_valid_play():
    """Test valid actions execute successfully."""
    game = create_game_with_state(
        H_1=[(RED, 5)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    initial_hand_size = len(game.H_1)
    initial_pile_size = len(game.P)
    
    action = Action(X_1=(RED, 5))
    result = game.execute_action(action, 1)
    
    assert result is True
    assert len(game.H_1) == initial_hand_size - 1, "Card should be removed from hand"
    assert len(game.P) == initial_pile_size + 1, "Card should be added to played pile"


def test_action_execution_invalid_card_not_in_hand():
    """Test invalid actions are rejected (card not in hand)."""
    game = create_game_with_state(
        H_1=[(RED, 5)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    action = Action(X_1=(YELLOW, 3))  # Not in hand
    result = game.execute_action(action, 1)
    
    assert result is False, "Should reject card not in hand"


def test_action_execution_invalid_illegal_play():
    """Test invalid actions are rejected (illegal play)."""
    game = create_game_with_state(
        H_1=[(YELLOW, 3)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    action = Action(X_1=(YELLOW, 3))  # Illegal play
    result = game.execute_action(action, 1)
    
    assert result is False, "Should reject illegal play"


def test_action_execution_player_2():
    """Test action execution for Player 2."""
    game = create_game_with_state(
        H_1=[],
        H_2=[(RED, 5)],
        top_card=(RED, 7),
        deck=[]
    )
    
    action = Action(X_1=(RED, 5))
    result = game.execute_action(action, 2)
    
    assert result is True
    assert len(game.H_2) == 0, "Card should be removed from Player 2's hand"
    assert len(game.P) == 2, "Card should be added to played pile"


```

```python
"""Tests for UNO game logic and rules."""

import pytest
from .game import Uno
from .cards import (
    RED,
    YELLOW,
    GREEN,
    BLUE,
    SKIP,
    REVERSE,
    DRAW_2,
    WILD,
    WILD_DRAW_4,
    BLACK,
)
from .state import Action


# Helper functions for test setup
def create_game_with_state(H_1, H_2, top_card, deck):
    """Helper to create game with specific state."""
    game = Uno(H_1=H_1, H_2=H_2, D_g=deck, P=[top_card])
    game.current_color = top_card[0] if top_card[0] is not None else RED
    game.create_S()
    return game


# Game Initialization Tests
def test_game_initialization():
    """Verify 7 cards dealt to each player, initial top card is non-Wild, game state is Active."""
    game = Uno()
    game.new_game(seed=42)
    
    assert len(game.H_1) == 7, "Player 1 should have 7 cards"
    assert len(game.H_2) == 7, "Player 2 should have 7 cards"
    assert len(game.P) == 1, "Should have one card in played pile"
    assert game.P_t is not None, "Top card should exist"
    assert game.P_t[0] is not None, "Top card should not be Wild (should have color)"
    assert len(game.D_g) > 0, "Deck should have remaining cards"
    assert game.G_o == "Active", "Game should be Active"
    assert game.current_color is not None, "Current color should be set"


def test_game_initialization_state_creation():
    """Verify state creation works correctly."""
    game = Uno()
    game.new_game(seed=42)
    
    state = game.create_S()
    assert state is not None
    assert len(state[0]) == 7  # H_1
    assert len(state[1]) == 7  # H_2
    assert len(state[2]) > 0  # D_g
    assert len(state[3]) == 1  # P
    assert state[4] is not None  # P_t
    assert state[5] == "Active"  # G_o


# Legal Play Rules Tests
def test_legal_play_color_match():
    """Test color matching (same color as top card)."""
    game = create_game_with_state(
        H_1=[(RED, 5), (YELLOW, 3)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    assert game.is_legal_play((RED, 5)) is True, "Red card should match red top card"
    assert game.is_legal_play((YELLOW, 3)) is False, "Yellow card should not match red top card"


def test_legal_play_value_match():
    """Test value matching (same value as top card)."""
    game = create_game_with_state(
        H_1=[(RED, 5), (YELLOW, 5)],
        H_2=[],
        top_card=(GREEN, 5),
        deck=[]
    )
    
    assert game.is_legal_play((RED, 5)) is True, "Same value should match"
    assert game.is_legal_play((YELLOW, 5)) is True, "Same value should match"


def test_legal_play_wild_always_legal():
    """Test Wild cards are always legal."""
    game = create_game_with_state(
        H_1=[(BLACK, WILD), (BLACK, WILD_DRAW_4)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    assert game.is_legal_play((BLACK, WILD)) is True, "Wild should always be legal"
    assert game.is_legal_play((BLACK, WILD_DRAW_4)) is True, "Wild Draw 4 should always be legal"


def test_legal_play_illegal_rejected():
    """Test illegal plays are rejected."""
    game = create_game_with_state(
        H_1=[(YELLOW, 3), (GREEN, 8)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    assert game.is_legal_play((YELLOW, 3)) is False, "Different color and value should be illegal"
    assert game.is_legal_play((GREEN, 8)) is False, "Different color and value should be illegal"


def test_legal_play_wild_color_handling():
    """Test current_color handling for Wild cards."""
    game = create_game_with_state(
        H_1=[(RED, 5)],
        H_2=[],
        top_card=(BLACK, WILD),
        deck=[]
    )
    game.current_color = BLUE
    
    assert game.is_legal_play((RED, 5)) is False, "Should not match when current_color is BLUE"
    
    game.current_color = RED
    assert game.is_legal_play((RED, 5)) is True, "Should match when current_color is RED"


def test_get_legal_actions():
    """Test get_legal_actions returns correct actions."""
    game = create_game_with_state(
        H_1=[(RED, 5), (YELLOW, 3)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    legal_actions = game.get_legal_actions(1)
    assert len(legal_actions) == 1, "Should have one legal play (RED 5)"
    assert legal_actions[0].is_play() is True
    assert legal_actions[0].X_1 == (RED, 5)


def test_get_legal_actions_no_legal_plays():
    """Test get_legal_actions returns draw action when no legal plays."""
    game = create_game_with_state(
        H_1=[(YELLOW, 3), (GREEN, 8)],
        H_2=[],
        top_card=(RED, 7),
        deck=[(BLUE, 1)]
    )
    
    legal_actions = game.get_legal_actions(1)
    assert len(legal_actions) == 1, "Should have one action (draw)"
    assert legal_actions[0].is_draw() is True
    assert legal_actions[0].n == 1


# Special Card Effects Tests
def test_special_card_skip():
    """Test SKIP: Player plays again (2-player rule)."""
    game = create_game_with_state(
        H_1=[(RED, SKIP)],
        H_2=[],
        top_card=(RED, 5),
        deck=[]
    )
    
    action = Action(X_1=(RED, SKIP))
    result = game.execute_action(action, 1)
    
    assert result is True
    assert game.player_plays_again is True, "Player should play again after Skip"
    assert len(game.H_1) == 0, "Card should be removed from hand"


def test_special_card_reverse():
    """Test REVERSE: Player plays again (2-player rule, acts like Skip)."""
    game = create_game_with_state(
        H_1=[(RED, REVERSE)],
        H_2=[],
        top_card=(RED, 5),
        deck=[]
    )
    
    action = Action(X_1=(RED, REVERSE))
    result = game.execute_action(action, 1)
    
    assert result is True
    assert game.player_plays_again is True, "Player should play again after Reverse (2-player rule)"


def test_special_card_draw_2():
    """Test DRAW_2: Next player draws 2 and skips turn."""
    game = create_game_with_state(
        H_1=[(RED, DRAW_2)],
        H_2=[(YELLOW, 1)],
        top_card=(RED, 5),
        deck=[(BLUE, 2), (GREEN, 3)]
    )
    
    action = Action(X_1=(RED, DRAW_2))
    result = game.execute_action(action, 1)
    
    assert result is True
    assert game.draw_pending == 2, "Next player should draw 2 cards"


def test_special_card_wild():
    """Test WILD: Choose color, current_color updated."""
    game = create_game_with_state(
        H_1=[(BLACK, WILD)],
        H_2=[],
        top_card=(RED, 5),
        deck=[]
    )
    
    action = Action(X_1=(BLACK, WILD), wild_color=BLUE)
    result = game.execute_action(action, 1)
    
    assert result is True
    assert game.current_color == BLUE, "Current color should be set to chosen color"
    assert len(game.H_1) == 0, "Card should be removed from hand"


def test_special_card_wild_draw_4():
    """Test WILD_DRAW_4: Next player draws 4 and skips turn."""
    game = create_game_with_state(
        H_1=[(BLACK, WILD_DRAW_4)],
        H_2=[(YELLOW, 1)],
        top_card=(RED, 5),
        deck=[(BLUE, 2), (GREEN, 3), (RED, 4), (YELLOW, 5)]
    )
    
    action = Action(X_1=(BLACK, WILD_DRAW_4), wild_color=GREEN)
    result = game.execute_action(action, 1)
    
    assert result is True
    assert game.current_color == GREEN, "Current color should be set"
    assert game.draw_pending == 4, "Next player should draw 4 cards"


# Draw Rules Tests
def test_draw_rules_no_legal_plays():
    """Test when no legal plays, must draw 1 card."""
    game = create_game_with_state(
        H_1=[(YELLOW, 3)],
        H_2=[],
        top_card=(RED, 7),
        deck=[(BLUE, 1)]
    )
    
    legal_actions = game.get_legal_actions(1)
    assert len(legal_actions) == 1
    assert legal_actions[0].is_draw() is True
    assert legal_actions[0].n == 1


def test_draw_action_adds_cards():
    """Test draw action properly adds cards to hand."""
    game = create_game_with_state(
        H_1=[(YELLOW, 3)],
        H_2=[],
        top_card=(RED, 7),
        deck=[(BLUE, 1), (GREEN, 2)]
    )
    
    initial_hand_size = len(game.H_1)
    action = Action(n=1)
    result = game.execute_action(action, 1)
    
    assert result is True
    assert len(game.H_1) == initial_hand_size + 1, "Hand should have one more card"
    assert len(action.Y_n) == 1, "Action should record drawn cards"


def test_draw_pending_cards():
    """Test draw pending cards (from DRAW_2/WILD_DRAW_4)."""
    game = create_game_with_state(
        H_1=[],
        H_2=[(YELLOW, 1)],
        top_card=(RED, DRAW_2),
        deck=[(BLUE, 1), (GREEN, 2)]
    )
    game.draw_pending = 2
    
    action = Action(n=1)  # Player tries to draw 1, but should draw 2
    result = game.execute_action(action, 2)
    
    assert result is True
    assert len(game.H_2) == 3, "Should have drawn 2 cards (1 original + 2 pending)"
    assert game.draw_pending == 0, "Draw pending should be cleared"


# Game End Tests
def test_game_end_player_has_zero_cards():
    """Test game ends when player has 0 cards."""
    game = create_game_with_state(
        H_1=[(RED, 5)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    action = Action(X_1=(RED, 5))
    game.execute_action(action, 1)
    
    assert len(game.H_1) == 0, "Player 1 should have 0 cards"
    assert game.G_o == "GameOver", "Game should be over"


def test_game_end_no_actions_after_game_over():
    """Test no actions allowed after game over."""
    game = create_game_with_state(
        H_1=[],
        H_2=[(YELLOW, 1)],
        top_card=(RED, 7),
        deck=[]
    )
    game.G_o = "GameOver"
    
    action = Action(X_1=(YELLOW, 1))
    result = game.execute_action(action, 2)
    
    assert result is False, "Should not allow actions after game over"


# Deck Reshuffling Tests
def test_deck_reshuffle_when_empty():
    """Test when draw pile empty, reshuffle played pile (except top card)."""
    game = create_game_with_state(
        H_1=[(YELLOW, 3)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]  # Empty deck
    )
    # Add some cards to played pile
    game.P = [(RED, 7), (BLUE, 1), (GREEN, 2), (YELLOW, 3)]
    
    action = Action(n=1)
    game.execute_action(action, 1)
    
    assert len(game.D_g) > 0, "Deck should have cards after reshuffle"
    assert len(game.P) == 1, "Played pile should only have top card"
    assert game.P[0] == (RED, 7), "Top card should remain"


def test_draw_after_reshuffle():
    """Test cards can still be drawn after reshuffle."""
    game = create_game_with_state(
        H_1=[(YELLOW, 3)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]  # Empty deck
    )
    game.P = [(RED, 7), (BLUE, 1), (GREEN, 2)]
    
    initial_hand_size = len(game.H_1)
    action = Action(n=1)
    result = game.execute_action(action, 1)
    
    assert result is True
    assert len(game.H_1) == initial_hand_size + 1, "Should have drawn a card after reshuffle"


# Action Execution Tests
def test_action_execution_valid_play():
    """Test valid actions execute successfully."""
    game = create_game_with_state(
        H_1=[(RED, 5)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    initial_hand_size = len(game.H_1)
    initial_pile_size = len(game.P)
    
    action = Action(X_1=(RED, 5))
    result = game.execute_action(action, 1)
    
    assert result is True
    assert len(game.H_1) == initial_hand_size - 1, "Card should be removed from hand"
    assert len(game.P) == initial_pile_size + 1, "Card should be added to played pile"


def test_action_execution_invalid_card_not_in_hand():
    """Test invalid actions are rejected (card not in hand)."""
    game = create_game_with_state(
        H_1=[(RED, 5)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    action = Action(X_1=(YELLOW, 3))  # Not in hand
    result = game.execute_action(action, 1)
    
    assert result is False, "Should reject card not in hand"


def test_action_execution_invalid_illegal_play():
    """Test invalid actions are rejected (illegal play)."""
    game = create_game_with_state(
        H_1=[(YELLOW, 3)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    action = Action(X_1=(YELLOW, 3))  # Illegal play
    result = game.execute_action(action, 1)
    
    assert result is False, "Should reject illegal play"


def test_action_execution_player_2():
    """Test action execution for Player 2."""
    game = create_game_with_state(
        H_1=[],
        H_2=[(RED, 5)],
        top_card=(RED, 7),
        deck=[]
    )
    
    action = Action(X_1=(RED, 5))
    result = game.execute_action(action, 2)
    
    assert result is True
    assert len(game.H_2) == 0, "Card should be removed from Player 2's hand"
    assert len(game.P) == 2, "Card should be added to played pile"


```

```python
"""Tests for UNO state and Action class."""

import pytest
from .state import Action
from .cards import RED, YELLOW, BLACK, WILD


def test_action_validation_must_specify_one():
    """Test Action validation: must specify X_1 OR n, not both."""
    # Valid: only X_1
    action1 = Action(X_1=(RED, 5))
    assert action1.is_play() is True
    assert action1.is_draw() is False
    
    # Valid: only n
    action2 = Action(n=1)
    assert action2.is_play() is False
    assert action2.is_draw() is True
    
    # Invalid: both specified
    with pytest.raises(ValueError, match="Must specify EITHER"):
        Action(X_1=(RED, 5), n=1)
    
    # Invalid: neither specified
    with pytest.raises(ValueError, match="Must specify EITHER"):
        Action()


def test_action_validation_draw_n_values():
    """Test draw action validation: n must be in {1, 2, 4}."""
    # Valid values
    Action(n=1)
    Action(n=2)
    Action(n=4)
    
    # Invalid values
    with pytest.raises(ValueError, match="n must be in"):
        Action(n=0)
    
    with pytest.raises(ValueError, match="n must be in"):
        Action(n=3)
    
    with pytest.raises(ValueError, match="n must be in"):
        Action(n=5)


def test_action_type_checking():
    """Test action type checking (is_play, is_draw)."""
    play_action = Action(X_1=(RED, 5))
    assert play_action.is_play() is True
    assert play_action.is_draw() is False
    
    draw_action = Action(n=1)
    assert draw_action.is_play() is False
    assert draw_action.is_draw() is True


def test_action_representation_play():
    """Test Action representation for play actions."""
    action = Action(X_1=(RED, 5))
    repr_str = repr(action)
    assert "X_1" in repr_str
    assert "RED" in repr_str or "5" in repr_str
    
    # With wild color
    wild_action = Action(X_1=(BLACK, WILD), wild_color=YELLOW)
    repr_str = repr(wild_action)
    assert "wild_color" in repr_str
    assert "YELLOW" in repr_str or "Y" in repr_str


def test_action_representation_draw():
    """Test Action representation for draw actions."""
    action = Action(n=1)
    repr_str = repr(action)
    assert "Y_1" in repr_str or "n=1" in repr_str
    
    # After drawing (Y_n filled)
    action.Y_n = [(RED, 5)]
    repr_str = repr(action)
    assert "Y_1" in repr_str or "Y_n" in repr_str


def test_action_wild_color():
    """Test Action with wild_color parameter."""
    action = Action(X_1=(BLACK, WILD), wild_color=RED)
    assert action.wild_color == RED
    assert action.X_1 == (BLACK, WILD)


```


Fixing the failing test: the top card is the last card in the pile, not the first:


```python
"""Tests for UNO game logic and rules."""

import pytest
from .game import Uno
from .cards import (
    RED,
    YELLOW,
    GREEN,
    BLUE,
    SKIP,
    REVERSE,
    DRAW_2,
    WILD,
    WILD_DRAW_4,
    BLACK,
)
from .state import Action


# Helper functions for test setup
def create_game_with_state(H_1, H_2, top_card, deck):
    """Helper to create game with specific state."""
    game = Uno(H_1=H_1, H_2=H_2, D_g=deck, P=[top_card])
    game.current_color = top_card[0] if top_card[0] is not None else RED
    game.create_S()
    return game


# Game Initialization Tests
def test_game_initialization():
    """Verify 7 cards dealt to each player, initial top card is non-Wild, game state is Active."""
    game = Uno()
    game.new_game(seed=42)
    
    assert len(game.H_1) == 7, "Player 1 should have 7 cards"
    assert len(game.H_2) == 7, "Player 2 should have 7 cards"
    assert len(game.P) == 1, "Should have one card in played pile"
    assert game.P_t is not None, "Top card should exist"
    assert game.P_t[0] is not None, "Top card should not be Wild (should have color)"
    assert len(game.D_g) > 0, "Deck should have remaining cards"
    assert game.G_o == "Active", "Game should be Active"
    assert game.current_color is not None, "Current color should be set"


def test_game_initialization_state_creation():
    """Verify state creation works correctly."""
    game = Uno()
    game.new_game(seed=42)
    
    state = game.create_S()
    assert state is not None
    assert len(state[0]) == 7  # H_1
    assert len(state[1]) == 7  # H_2
    assert len(state[2]) > 0  # D_g
    assert len(state[3]) == 1  # P
    assert state[4] is not None  # P_t
    assert state[5] == "Active"  # G_o


# Legal Play Rules Tests
def test_legal_play_color_match():
    """Test color matching (same color as top card)."""
    game = create_game_with_state(
        H_1=[(RED, 5), (YELLOW, 3)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    assert game.is_legal_play((RED, 5)) is True, "Red card should match red top card"
    assert game.is_legal_play((YELLOW, 3)) is False, "Yellow card should not match red top card"


def test_legal_play_value_match():
    """Test value matching (same value as top card)."""
    game = create_game_with_state(
        H_1=[(RED, 5), (YELLOW, 5)],
        H_2=[],
        top_card=(GREEN, 5),
        deck=[]
    )
    
    assert game.is_legal_play((RED, 5)) is True, "Same value should match"
    assert game.is_legal_play((YELLOW, 5)) is True, "Same value should match"


def test_legal_play_wild_always_legal():
    """Test Wild cards are always legal."""
    game = create_game_with_state(
        H_1=[(BLACK, WILD), (BLACK, WILD_DRAW_4)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    assert game.is_legal_play((BLACK, WILD)) is True, "Wild should always be legal"
    assert game.is_legal_play((BLACK, WILD_DRAW_4)) is True, "Wild Draw 4 should always be legal"


def test_legal_play_illegal_rejected():
    """Test illegal plays are rejected."""
    game = create_game_with_state(
        H_1=[(YELLOW, 3), (GREEN, 8)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    assert game.is_legal_play((YELLOW, 3)) is False, "Different color and value should be illegal"
    assert game.is_legal_play((GREEN, 8)) is False, "Different color and value should be illegal"


def test_legal_play_wild_color_handling():
    """Test current_color handling for Wild cards."""
    game = create_game_with_state(
        H_1=[(RED, 5)],
        H_2=[],
        top_card=(BLACK, WILD),
        deck=[]
    )
    game.current_color = BLUE
    
    assert game.is_legal_play((RED, 5)) is False, "Should not match when current_color is BLUE"
    
    game.current_color = RED
    assert game.is_legal_play((RED, 5)) is True, "Should match when current_color is RED"


def test_get_legal_actions():
    """Test get_legal_actions returns correct actions."""
    game = create_game_with_state(
        H_1=[(RED, 5), (YELLOW, 3)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    legal_actions = game.get_legal_actions(1)
    assert len(legal_actions) == 1, "Should have one legal play (RED 5)"
    assert legal_actions[0].is_play() is True
    assert legal_actions[0].X_1 == (RED, 5)


def test_get_legal_actions_no_legal_plays():
    """Test get_legal_actions returns draw action when no legal plays."""
    game = create_game_with_state(
        H_1=[(YELLOW, 3), (GREEN, 8)],
        H_2=[],
        top_card=(RED, 7),
        deck=[(BLUE, 1)]
    )
    
    legal_actions = game.get_legal_actions(1)
    assert len(legal_actions) == 1, "Should have one action (draw)"
    assert legal_actions[0].is_draw() is True
    assert legal_actions[0].n == 1


# Special Card Effects Tests
def test_special_card_skip():
    """Test SKIP: Player plays again (2-player rule)."""
    game = create_game_with_state(
        H_1=[(RED, SKIP)],
        H_2=[],
        top_card=(RED, 5),
        deck=[]
    )
    
    action = Action(X_1=(RED, SKIP))
    result = game.execute_action(action, 1)
    
    assert result is True
    assert game.player_plays_again is True, "Player should play again after Skip"
    assert len(game.H_1) == 0, "Card should be removed from hand"


def test_special_card_reverse():
    """Test REVERSE: Player plays again (2-player rule, acts like Skip)."""
    game = create_game_with_state(
        H_1=[(RED, REVERSE)],
        H_2=[],
        top_card=(RED, 5),
        deck=[]
    )
    
    action = Action(X_1=(RED, REVERSE))
    result = game.execute_action(action, 1)
    
    assert result is True
    assert game.player_plays_again is True, "Player should play again after Reverse (2-player rule)"


def test_special_card_draw_2():
    """Test DRAW_2: Next player draws 2 and skips turn."""
    game = create_game_with_state(
        H_1=[(RED, DRAW_2)],
        H_2=[(YELLOW, 1)],
        top_card=(RED, 5),
        deck=[(BLUE, 2), (GREEN, 3)]
    )
    
    action = Action(X_1=(RED, DRAW_2))
    result = game.execute_action(action, 1)
    
    assert result is True
    assert game.draw_pending == 2, "Next player should draw 2 cards"


def test_special_card_wild():
    """Test WILD: Choose color, current_color updated."""
    game = create_game_with_state(
        H_1=[(BLACK, WILD)],
        H_2=[],
        top_card=(RED, 5),
        deck=[]
    )
    
    action = Action(X_1=(BLACK, WILD), wild_color=BLUE)
    result = game.execute_action(action, 1)
    
    assert result is True
    assert game.current_color == BLUE, "Current color should be set to chosen color"
    assert len(game.H_1) == 0, "Card should be removed from hand"


def test_special_card_wild_draw_4():
    """Test WILD_DRAW_4: Next player draws 4 and skips turn."""
    game = create_game_with_state(
        H_1=[(BLACK, WILD_DRAW_4)],
        H_2=[(YELLOW, 1)],
        top_card=(RED, 5),
        deck=[(BLUE, 2), (GREEN, 3), (RED, 4), (YELLOW, 5)]
    )
    
    action = Action(X_1=(BLACK, WILD_DRAW_4), wild_color=GREEN)
    result = game.execute_action(action, 1)
    
    assert result is True
    assert game.current_color == GREEN, "Current color should be set"
    assert game.draw_pending == 4, "Next player should draw 4 cards"


# Draw Rules Tests
def test_draw_rules_no_legal_plays():
    """Test when no legal plays, must draw 1 card."""
    game = create_game_with_state(
        H_1=[(YELLOW, 3)],
        H_2=[],
        top_card=(RED, 7),
        deck=[(BLUE, 1)]
    )
    
    legal_actions = game.get_legal_actions(1)
    assert len(legal_actions) == 1
    assert legal_actions[0].is_draw() is True
    assert legal_actions[0].n == 1


def test_draw_action_adds_cards():
    """Test draw action properly adds cards to hand."""
    game = create_game_with_state(
        H_1=[(YELLOW, 3)],
        H_2=[],
        top_card=(RED, 7),
        deck=[(BLUE, 1), (GREEN, 2)]
    )
    
    initial_hand_size = len(game.H_1)
    action = Action(n=1)
    result = game.execute_action(action, 1)
    
    assert result is True
    assert len(game.H_1) == initial_hand_size + 1, "Hand should have one more card"
    assert len(action.Y_n) == 1, "Action should record drawn cards"


def test_draw_pending_cards():
    """Test draw pending cards (from DRAW_2/WILD_DRAW_4)."""
    game = create_game_with_state(
        H_1=[],
        H_2=[(YELLOW, 1)],
        top_card=(RED, DRAW_2),
        deck=[(BLUE, 1), (GREEN, 2)]
    )
    game.draw_pending = 2
    
    action = Action(n=1)  # Player tries to draw 1, but should draw 2
    result = game.execute_action(action, 2)
    
    assert result is True
    assert len(game.H_2) == 3, "Should have drawn 2 cards (1 original + 2 pending)"
    assert game.draw_pending == 0, "Draw pending should be cleared"


# Game End Tests
def test_game_end_player_has_zero_cards():
    """Test game ends when player has 0 cards."""
    game = create_game_with_state(
        H_1=[(RED, 5)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    action = Action(X_1=(RED, 5))
    game.execute_action(action, 1)
    
    assert len(game.H_1) == 0, "Player 1 should have 0 cards"
    assert game.G_o == "GameOver", "Game should be over"


def test_game_end_no_actions_after_game_over():
    """Test no actions allowed after game over."""
    game = create_game_with_state(
        H_1=[],
        H_2=[(YELLOW, 1)],
        top_card=(RED, 7),
        deck=[]
    )
    game.G_o = "GameOver"
    
    action = Action(X_1=(YELLOW, 1))
    result = game.execute_action(action, 2)
    
    assert result is False, "Should not allow actions after game over"


# Deck Reshuffling Tests
def test_deck_reshuffle_when_empty():
    """Test when draw pile empty, reshuffle played pile (except top card)."""
    game = create_game_with_state(
        H_1=[(YELLOW, 3)],
        H_2=[],
        top_card=(YELLOW, 3),  # Top card is last in pile
        deck=[]  # Empty deck
    )
    # Add some cards to played pile (top card is last)
    game.P = [(RED, 7), (BLUE, 1), (GREEN, 2), (YELLOW, 3)]
    game.create_S()  # Update P_t
    
    top_card_before = game.P_t
    
    action = Action(n=1)
    game.execute_action(action, 1)
    
    assert len(game.D_g) > 0, "Deck should have cards after reshuffle"
    assert len(game.P) == 1, "Played pile should only have top card"
    assert game.P[0] == top_card_before, "Top card should remain"


def test_draw_after_reshuffle():
    """Test cards can still be drawn after reshuffle."""
    game = create_game_with_state(
        H_1=[(YELLOW, 3)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]  # Empty deck
    )
    game.P = [(RED, 7), (BLUE, 1), (GREEN, 2)]
    
    initial_hand_size = len(game.H_1)
    action = Action(n=1)
    result = game.execute_action(action, 1)
    
    assert result is True
    assert len(game.H_1) == initial_hand_size + 1, "Should have drawn a card after reshuffle"


# Action Execution Tests
def test_action_execution_valid_play():
    """Test valid actions execute successfully."""
    game = create_game_with_state(
        H_1=[(RED, 5)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    initial_hand_size = len(game.H_1)
    initial_pile_size = len(game.P)
    
    action = Action(X_1=(RED, 5))
    result = game.execute_action(action, 1)
    
    assert result is True
    assert len(game.H_1) == initial_hand_size - 1, "Card should be removed from hand"
    assert len(game.P) == initial_pile_size + 1, "Card should be added to played pile"


def test_action_execution_invalid_card_not_in_hand():
    """Test invalid actions are rejected (card not in hand)."""
    game = create_game_with_state(
        H_1=[(RED, 5)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    action = Action(X_1=(YELLOW, 3))  # Not in hand
    result = game.execute_action(action, 1)
    
    assert result is False, "Should reject card not in hand"


def test_action_execution_invalid_illegal_play():
    """Test invalid actions are rejected (illegal play)."""
    game = create_game_with_state(
        H_1=[(YELLOW, 3)],
        H_2=[],
        top_card=(RED, 7),
        deck=[]
    )
    
    action = Action(X_1=(YELLOW, 3))  # Illegal play
    result = game.execute_action(action, 1)
    
    assert result is False, "Should reject illegal play"


def test_action_execution_player_2():
    """Test action execution for Player 2."""
    game = create_game_with_state(
        H_1=[],
        H_2=[(RED, 5)],
        top_card=(RED, 7),
        deck=[]
    )
    
    action = Action(X_1=(RED, 5))
    result = game.execute_action(action, 2)
    
    assert result is True
    assert len(game.H_2) == 0, "Card should be removed from Player 2's hand"
    assert len(game.P) == 2, "Card should be added to played pile"


```


Running all tests to confirm everything passes:



All 37 tests pass. Verifying the CI workflow is correct:



## Summary

1. **Created `src/uno/test_cards.py`** — 6 tests covering:
   - Deck size (108 cards)
   - Card distribution (number cards, special cards, wild cards)
   - Number card distribution (one 0, two of each 1-9 per color)
   - Special card distribution
   - Wild card distribution
   - `card_to_string()` function

2. **Created `src/uno/test_game.py`** — 25 tests covering:
   - Game initialization (7 cards per player, non-Wild top card, Active state)
   - Legal play rules (color match, value match, wild cards, illegal plays)
   - Special card effects (Skip, Reverse, Draw 2, Wild, Wild Draw 4)
   - Draw rules (no legal plays → draw 1, turn ends)
   - Game end conditions
   - Deck reshuffling
   - Action execution (valid/invalid actions, both players)

3. **Created `src/uno/test_state.py`** — 6 tests covering:
   - Action validation (X_1 OR n, not both)
   - Draw action validation (n must be in {1, 2, 4})
   - Action type checking
   - Action representation

4. **Created `.github/workflows/ci.yml`** — GitHub Actions workflow that:
   - Runs on all pushes and pull requests
   - Uses `uvx pytest -v` to run tests
   - Uses Python 3.13 as specified

All 37 tests pass. Tests focus on core UNO game logic for 2-player games, not policy.

---

**User**

verify they all work using uvx 

---

**Cursor**


Running all tests with `uvx pytest`:



All 37 tests pass using `uvx pytest`.

- 6 card/deck tests
- 25 game logic tests
- 6 state/action tests

Tests cover core UNO game logic for 2-player games. The CI workflow will run these automatically on push/PR.

