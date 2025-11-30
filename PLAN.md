# UNO POMCP Simulation Implementation Plan

## Phase 1: Core Infrastructure
- [x] Create `src/utils/simulation_logger.py` with structured JSON output
- [x] Log format: timestamp, config, matchup, aggregate results
- [x] File management: timestamped files in `results/` directory
- [x] Error handling: terminate on any error, discard partial results
- [x] Add timing decorators to `ParticlePolicy.get_action()` in `src/policy/particle_policy.py`
- [x] Time naive policy decisions in `src/utils/game_runner.py`
- [x] Capture per-turn and cumulative decision times
- [x] Modify `src/utils/game_runner.py` to support naive vs naive
- [x] Create `choose_action_naive()` function
- [x] Add naive agent vs naive agent simulation to test.py (10 games, 10K turn cap, 0.5 win each if cap reached)
- [x] Update game_runner.py functions to use "naive" instead of "random" with proper naive behavior:
  - Play first legal card found (if multiple, randomly pick one)
  - If Wild, choose most frequent color in hand (random if tie)
  - If no legal play, draw 1 card
- [x] Create comprehensive unit tests in test_naive_agent.py to verify naive behavior:
  - Plays first legal card found
  - Draws when no legal plays available
  - Wild cards choose most frequent color (random on tie)
  - Wild Draw 4 cards follow same logic as Wild cards
  - Black cards ignored when counting colors
  - Random color selection when no colored cards in hand
  - Player 2 hand used when player is 2
  - Decision timing is recorded
- [x] Verify POMCP solver and game logic consistency:
  - Wild card color selection rules are consistent across codebase
  - Black color exclusion properly implemented
  - Single letter color constants (RED, YELLOW, GREEN, BLUE) used consistently
  - Game logic prevents black from being chosen as wild color
  - All agents follow same wild card rules
- [x] Verify comprehensive game rule consistency:
  - Card placement: Only playable cards can be placed on pile
  - Deck management: Proper draw pile reshuffling and card distribution
  - Turn management: Correct handling of skip, reverse, draw penalties
  - Wild card effects: Proper color selection and turn skipping
  - No card stacking: +2/+4 cards force draws, not additional plays
  - Turn alternation: Proper player switching unless special cards intervene
  - Game end detection: Correct winner determination when hand reaches zero
- [x] Verify POMCP solver optimization:
  - Automatic illegal move filtering in get_legal_actions()
  - No wasted computation on invalid game states
  - Rollout simulation respects legal move constraints
  - Optimal resource usage with reduced action space
  - Faster convergence through valid state sampling only
- [x] Fix infinite loop detection in game_runner.py:
  - Added consecutive draw monitoring with 20-draw threshold
  - Added state repetition detection with 50-turn threshold
  - Prevents particle vs particle infinite loops
- [x] Fix initial card dealing in game.py:
  - Ensure starting card is never a wild card
  - Move all wild cards to end of deck during initialization
  - Proper color initialization for first turn
- [x] Fix current color passing in particle_policy.py:
  - Pass game.current_color to particle policy get_action()
  - Proper wild card color handling in legal action detection
  - Ensure POMCP can correctly identify playable cards

## Phase 2: Multi-Matchup Framework
- [x] Create enum in `src/utils/matchup_types.py`: NAIVE, PARTICLE_POLICY
- [x] Extend `src/utils/game_runner.py` to accept matchup types
- [x] Unified interface for different player types
- [x] Naive vs Naive, Naive vs Particle, Particle vs Particle matchups in `src/utils/game_runner.py`
- [x] Ensure independent particle caches and policies for particle agents
- [x] Handle symmetric decision timing for particle vs particle
- [x] Modify `batch_run.py` to run all three matchups sequentially
- [x] JSON logging for each configuration
- [x] Progress reporting every 10 runs
- [x] Maintain full game console output as currently implemented

## Phase 3: Parameter Sensitivity Analysis
- [x] Create `src/utils/config_variator.py`
- [x] Generate +/- 25% variants for each parameter in config.jsonc
- [x] Maintain parameter combinations and metadata
- [x] Integrate config variator with modified `batch_run.py`
- [x] Test each variant vs random opponent
- [x] Log parameter impact on win rates and decision times
- [x] Add sensitivity_simulations parameter to config.jsonc (10 sims per variant)
- [x] Implement three matchup types: Particle vs Naive, Particle vs Particle (same config), Particle vs Particle (mixed config)
- [x] Add run_matchup_game_with_configs() for mixed config testing

## Phase 4: Results Analysis
- [x] Create `results.py` in root directory
- [x] Parse JSON logs from `results/` directory
- [x] Print formatted summaries with win rates, decision times, cache statistics
- [x] Support running 100+ games per configuration in `batch_run.py`
- [x] Comprehensively test true win rate across all parameter variants
- [x] No confidence intervals or statistical significance testing
- [x] Add comprehensive_simulations parameter to config.jsonc (100 sims)
- [x] Add --comprehensive flag to batch_run.py for true win rate testing
- [x] Enhanced results.py with parameter impact analysis and variant comparison
- [x] Add matplotlib dependency via uv
- [x] Create visualization functions in results.py
- [x] Add parameter impact charts (bar charts showing win rate, decision time, cache size changes)
- [x] Add win rate comparison plots (scatter plots and distributions)
- [x] Add decision time and cache size visualizations (performance metrics analysis)
- [x] Save plots to results/plots/ directory with --plots flag

## File Structure Plan CAN CHANGE AND UPDATE AS NECESSARY
```
Root:
- batch_run.py (modified for comprehensive simulations) ✅
- results.py (new analysis script)
- main.py (unchanged) ✅
- test.py (unchanged) ✅

src/utils:
- simulation_logger.py (new) ✅
- matchup_types.py (new) ✅
- config_variator.py (new)
- game_runner.py (modified) ✅
- config_loader.py (unchanged) ✅

src/policy:
- particle_policy.py (modified for timing and cache stats) ✅

results/ (new directory): ✅
- timestamped JSON files ✅
```

## Implementation Assumptions
- JSON file naming: `results/simulation_YYYY-MM-DD_HH-MM-SS.json`
- Aggregate logging only (no individual game details in JSON)
- Parameter testing order: sequential as generated by variator
- Cache tracking: log average/median cache size across simulations
- Progress reporting: print summary every 10 completed games
- Error handling: immediate termination with error message
- No memory optimization or time caps needed
- Full game console output maintained during batch runs

## Key Features
- Simple two-command interface: `uv run batch_run.py` and `uv run results.py`
- Comprehensive parameter sensitivity testing
- Real-time progress tracking during long runs
- Clean error handling with result discard
- Cache performance monitoring
- Aggregate JSON logging with detailed console output
- batch running configuration should be specified in `config.jsonc` under the `batch_run` key. This should be used for parameters where there is no 'right' or 'wrong' answer.

## Type Fixes Needed
- [x] Fix None type issues in game.py State tuple - RESOLVED
- [x] Fix Optional type handling in particle_policy.py - RESOLVED  
- [x] Fix card_to_string None handling in game_runner.py - RESOLVED
- [x] Fix Action __repr__ return type - RESOLVED
- [x] Fix config_loader cache type initialization - RESOLVED