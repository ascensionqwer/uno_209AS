# UNO POMCP Project Overview

## Project Purpose
UNO game engine with AI agent using dynamic particle filter and Monte Carlo Tree Search (MCTS) for decision-making. Implements Partially Observable Monte-Carlo Planning (POMCP) from Player 1's perspective.

## Tech Stack
- Python 3.13.6+
- Dependencies: matplotlib, pytest, lefthook
- UV for package management
- No external AI/ML libraries - pure Python implementation

## Code Style & Conventions
- Type hints used throughout
- Docstrings for classes and functions
- Single letter color constants (RED, YELLOW, GREEN, BLUE)
- Player 1 perspective for all policy decisions
- Particle filter for belief state approximation
- Dynamic particle generation with caching

## Key Architecture
- **Particle Policy**: Runtime decision-making using particle filter + MCTS
- **Particle Cache**: Caches particle sets per game state
- **Game Engine**: Core UNO game logic with full card set
- **Observation Utils**: Game state canonicalization
- **Game Runner**: Unified interface for different player types

## Entry Points
- `main.py`: Naive vs Naive simulation
- `batch_run.py`: Parameter sensitivity testing
- `results.py`: Results analysis and visualization

## Testing & Quality
- pytest for unit tests
- lefthook for git hooks
- Comprehensive test coverage for game logic and policies