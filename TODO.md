# Dynamic Particle Filter Implementation Plan

## Phase 1: Fix Core Sampling Bugs

**Issue**: Both policy generators have incorrect deck sampling logic
**Fix**: Ensure `deck_size = |L| - opponent_size` and sample all remaining cards correctly

## Phase 2: Dynamic Particle Filter Implementation

**Architecture Decisions:**
- **Cache**: In-memory only, no persistence
- **Growth**: Natural expansion, no size limits
- **Particle Count**: Fixed configuration (e.g., 1000)
- **Cache Keys**: Canonical observation strings
- **No Precomputation**: Generate on-demand only

## Phase 3: Integration Strategy

**Policy Generator 1:**
- Replace full enumeration with dynamic generation
- Cache exact value function results per observation
- Maintain mathematical correctness

**Policy Generator 2:**
- Dynamic particle generation for MCTS
- Cache particle sets per observation
- Reuse cached particles for similar states

## Phase 4: Implementation Details

**Cache Implementation:**
```python
class SimpleParticleCache:
    def __init__(self):
        self.cache = {}  # observation -> particles
    
    def get_particles(self, observation):
        if observation not in self.cache:
            self.cache[observation] = self._generate_particles(observation)
        return self.cache[observation]
```

**Particle Generation Fix:**
```python
def _generate_particles(self, H_1, opponent_size, deck_size, P):
    full_deck = build_full_deck()
    L = [c for c in full_deck if c not in H_1 and c not in P]
    
    # Auto-correct deck_size to match mathematical constraint
    actual_deck_size = len(L) - opponent_size
    
    particles = []
    for _ in range(num_particles):
        H_2 = random.sample(L, opponent_size)
        remaining = [c for c in L if c not in H_2]
        D_g = random.sample(remaining, len(remaining))  # Take all remaining
        particles.append(Particle(H_2, D_g, 1.0/num_particles))
    
    return particles
```

## Phase 5: Testing Strategy

**Validation Tests:**
- Particle generation produces valid card distributions
- Cache hit rate improves over time
- Memory usage grows linearly with unique states
- MCTS performance with dynamic particles

**Performance Tests:**
- Lookup speed improvements
- Particle generation overhead
- Cache memory efficiency

## Phase 6: Deployment

**Rollout Strategy:**
- Fix sampling bugs first (critical functionality)
- Add dynamic particle filter
- Implement caching layer
- Performance validation