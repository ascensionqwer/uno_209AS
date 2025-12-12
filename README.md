# POMDP Based Uno

This project explores the concept of "belief-based trapping" in the card game Uno. The core idea is to use a belief state (a representation of the possible cards an opponent might hold) to make optimal moves that "trap" the opponent, forcing them to draw a card.

## Scripts

This project contains the following scripts:

### `belief_based_trapping.py`

This script provides a concrete example of belief-based trapping. It sets up a specific, small-scale scenario with a micro-deck of 6 cards to demonstrate the concept. This script is used for the results section titled "Belief-Based Trapping: A Concrete Example."

### `generalization_believe_based_trapping.py`

This script runs an exhaustive analysis on the 6-card micro-deck to determine how often an optimal trapping play exists. It iterates through all possible game states and logs the results to `exhaustive_scenario_log.txt`. This script is used for the results section titled "Generalization: Exhaustive Analysis Reveals Rarity of Optimal Structure."

The output of this script is logged in `exhaustive_scenario_log.txt`.

### `belief_based_verification_on_extended_deck.py`

This script verifies the belief state update logic on a larger, more realistic Uno deck. It runs a series of random scenarios to test the correctness of the belief update mechanism when an opponent plays or draws a card. This script is used for the results section titled "Belief State Verification on Extended Deck."

### `belief_probability_verification.py`

This script verfies the belief state update logic with duplicate cards, and shows the probability of each card being present in the opponents hand after change in the game state (such as a draw or a play).

### `belief_probability_harness.py`

This script is an extended harness to test the robustness of our belief state update by testing against 900 random scenarios and recording its performance.

## Supporting Modules

*   `belief.py`: Contains the `Belief` class, which is used to represent the belief state of the agent.
*   `cards.py`: Defines the card constants (colors and values).
*   `uno.py`: Contains the core game logic for Uno.
*   `pomdp.py`: (Likely) contains the POMDP formulation for the Uno game.
*   `main.py`: The main entry point for the project.
*   `MATH.md`: Contains the mathematical formulation of the problem.

## How to Run

To run any of the scripts, simply execute them from the command line:

```bash
python belief_based_trapping.py
python generalization_believe_based_trapping.py
python belief_based_verification_on_extended_deck.py
```
