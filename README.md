# uno_209AS

by Samyak Kakatur & others

## Overview

UNO game engine with AI agent using dynamic particle filter and Monte Carlo Tree Search (MCTS) for decision-making.

**Important: Player 1 Perspective**
The policy operates from **Player 1's perspective**:

1. Player 1 observes game state (H_1, |H_2|, |D_g|, P, P_t, G_o)
2. Player 1 makes a move using `get_action()`
3. System dynamics occur (including Player 2's turn)
4. Player 1 observes new game state
5. Update particles based on observations of Player 2's actions using `update_after_opponent_action()`
6. Repeat

Particles represent beliefs about Player 2's hidden hand (H_2).

## Architecture

The system uses a **dynamic particle filter** approach that generates particles at runtime based on the current game state. Particles are cached per game state to avoid redundant generation.

### Key Components

- **Particle Policy** (`src/policy/particle_policy.py`): Runtime decision-making from Player 1's perspective using particle filter + MCTS
- **Particle Cache** (`src/policy/particle_cache.py`): Caches particle sets per game state (particles represent beliefs about Player 2's hand)
- **Game Engine** (`src/uno/`): Core UNO game logic and state management

### Mathematical Foundation

See `math/math.md` for the complete mathematical framework including:

- POMDP formulation
- Particle-based belief representation
- Bayesian filtering update equations
- MCTS integration

## Usage

### Running the Policy

```bash
python policy2.py
```

This will test the ParticlePolicy with a sample game state.

### Configuration

Edit `config.jsonc` to adjust parameters:

- `num_particles`: Number of particles for belief approximation
- `mcts_iterations`: MCTS tree search iterations per decision
- `planning_horizon`: Maximum lookahead depth
- `gamma`: Discount factor
- `ucb_c`: UCB1 exploration constant
- `rollout_particle_sample_size`: Particles sampled per rollout
- `resample_threshold`: ESS threshold for resampling

### Running the Game

```bash
python main.py
```

## Implementation Details

- **Dynamic Generation**: Particles generated on-demand at runtime
- **Caching**: Game-state keyed cache for particle reuse
- **No Precomputation**: All decisions made in real-time
- **Mathematical Correctness**: Enforces constraint `|D_g| = |L| - |H_2|` automatically
