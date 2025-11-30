# POMCP: Partially Observable Monte-Carlo Planning

## Overview

POMCP (Partially Observable Monte-Carlo Planning) extends UCT to partially observable environments by using a search tree of histories instead of states. The algorithm combines Monte-Carlo belief state updates with PO-UCT search for efficient planning in large POMDPs.

## Mathematical Framework

### Problem Setting

- **State Space**: $S = (H_1, H_2, D_g, P, P_t, G_o)$ - complete game state
- **Action Space**: $A = \{X_1\} \cup \{Y_n : n \in \{1, 2, 4\}\}$ - play card or draw cards
- **Observation Space**: $O = (H_1, |H_2|, |D_g|, P, P_t, G_o)$ - Player 1's observation
- **Transition Function**: $T(s' | s, a)$ - stochastic game dynamics
- **Observation Function**: $O(o | s', a)$ - deterministic observation given state
- **Reward Function**: $R(s', s, a) = \begin{cases} +1 & |H_1'| = 0 \\ -1 & |H_2'| = 0 \\ 0 & \text{otherwise} \end{cases}$

### Belief State

The belief state $b(s|h)$ represents probability distribution over states given observation history $h$. Since the state space is too large for exact representation, POMCP uses particle filtering:

$$\hat{b}(s|h) = \frac{1}{K} \sum_{i=1}^{K} \delta_{s,s^{(i)}}$$

where $\{s^{(i)}\}_{i=1}^{K}$ are $K$ particles and $\delta_{s,s'}$ is the Kronecker delta.

### POMCP Algorithm

#### 1. History-Based Tree Structure

Each tree node represents a history $h$ with statistics:

$$T(h) = \{N(h), V(h), B(h)\}$$

- $N(h)$: Visit count for history $h$
- $V(h)$: Value estimate for history $h$ (mean return of all simulations)
- $B(h)$: Belief state (particle set) for history $h$

#### 2. PO-UCT Selection

Actions are selected using UCB1 formula adapted for histories:

$$V^{\oplus}(ha) = V(ha) + c\sqrt{\frac{\log N(h)}{N(ha)}}$$

where $c$ is the exploration constant (typically $\sqrt{2}$).

#### 3. Monte-Carlo Simulation

Each simulation follows this process:

1. **Start State Sampling**: Sample $s \sim B(h_t)$ from current belief
2. **Tree Traversal**: Select actions using UCB1 until reaching unexplored node
3. **Rollout**: Simulate to terminal state using rollout policy $\pi_{\text{rollout}}(h,a)$
4. **Belief Update**: For every history $h$ encountered, update $B(h) \leftarrow B(h) \cup \{s\}$

#### 4. Backpropagation

Update statistics along traversed path:

$$N(h) \leftarrow N(h) + 1$$
$$V(h) \leftarrow \frac{V(h) \cdot N(h) + R}{N(h) + 1}$$

where $R$ is the simulation return.

#### 5. Tree Pruning

After executing real action $a_t$ and observing real outcome $o_t$:

1. **Find Child**: Locate child node $T(h_t a_t o_t)$
2. **Prune Tree**: Make this child the new root
3. **Update History**: $h_{t+1} = h_t \circ (a_t, o_t)$

All other histories become impossible and are discarded.

## Particle Filter Integration

### Particle Representation

Each particle represents a possible complete state:

$$s^{(i)} = (H_1^{(i)}, H_2^{(i)}, D_g^{(i)}, P^{(i)}, P_t^{(i)}, G_o^{(i)})$$

Only hidden components are stored:
- $H_2^{(i)}$: Opponent's hand (7 cards)
- $D_g^{(i)}$: Deck composition (remaining cards)

### Particle Generation

For observation $O = (H_1, |H_2|, |D_g|, P, P_t, G_o)$:

1. **Available Cards**: $L = D \setminus H_1 \setminus P$
2. **Constraint**: $|L| = |H_2| + |D_g|$ (all cards accounted for)
3. **Sampling**: For each particle $i$:
   - Sample $H_2^{(i)} \sim \text{Uniform}(\binom{L}{|H_2|})$
   - Set $D_g^{(i)} = L \setminus H_2^{(i)}$
   - Weight $w^{(i)} = \frac{1}{K}$

### Belief Updates

#### After Opponent Action

**Case 1**: Opponent plays card $z$
$$B'(h) = \{s^{(i)} : z \in H_2^{(i)}\}$$

**Case 2**: Opponent draws card
$$B'(h) = \{s^{(i)} : H_2^{(i)} \cap \text{LEGAL}(P_t) = \emptyset\}$$

### Monte-Carlo Rollout Policy

Simple randomized policy for simulation:

$$\pi_{\text{rollout}}(h,a) = \begin{cases}
\text{Random legal action} & \text{if } H_1 \cap \text{LEGAL}(P_t) \neq \emptyset \\
\text{Draw 1 card} & \text{otherwise}
\end{cases}$$

## Computational Complexity

- **Exact POMDP**: $O(|S|^3 \cdot |A| \cdot H)$ - intractable
- **POMCP**: $O(K \cdot |A| \cdot H)$ where $K$ is particle count
- **Space**: $O(K \cdot |S|)$ for particle storage
- **Time**: Linear in particle count and planning horizon

## Key Innovations

1. **History-Based Search**: Avoids state explosion by searching in history space
2. **Belief-Updated Tree**: Each node maintains particle set for its history
3. **Monte-Carlo State Updates**: Beliefs updated during simulations, not just real gameplay
4. **Tree Pruning**: Computational efficiency by eliminating impossible histories
5. **Importance Sampling**: Efficient belief representation using weighted particles

## UNO-Specific Adaptations

### Legal Play Function

$$\text{LEGAL}(P_t) = \{c \in D : c(\text{COLOR}) = P_t(\text{COLOR}) \lor c(\text{VALUE}) = P_t(\text{VALUE}) \lor c(\text{COLOR}) = \text{BLACK}\}$$

### Card Constraints

- **Wild Cards**: Always legal, can choose any color
- **Color Matching**: Red/Green/Blue/Yellow must match top card color
- **Value Matching**: Numbers and special cards must match top card value
- **Deck Partitioning**: $D = H_1 \sqcup H_2 \sqcup D_g \sqcup P$ (disjoint union)

### Special Card Handling

- **Skip/Reverse**: In 2-player game, acts as "play again" cards
- **Draw 2/4**: Force opponent to draw specified number of cards
- **Wild Draw 4**: Can only be played when no legal plays available

## Implementation Notes

### Particle Filtering

- **Particle Count**: 1000-5000 particles typical for good approximation
- **Resampling**: Systematic resampling when effective sample size $< \frac{K}{2}$
- **Caching**: Particle sets cached by game state for efficiency

### Search Parameters

- **Exploration Constant**: $c = \sqrt{2} \approx 1.414$ (UCB1 optimal)
- **Rollout Depth**: 5-10 moves lookahead typical
- **Simulations**: 1000-10000 simulations per decision for convergence
- **Discount Factor**: $\gamma = 0.95$ for future value discounting

### Convergence Properties

Under mild conditions:
- **Value Function Convergence**: $V(h) \to V^*(h)$ as $N(h) \to \infty$
- **Belief Consistency**: $\hat{b}(s|h) \to b(s|h)$ as $K \to \infty$
- **Optimal Action Selection**: $\arg\max_a V^{\oplus}(ha) \to \pi^*(h)$ as search expands

## Advantages Over Alternative Methods

1. **Scalability**: Handles large state spaces intractable for exact methods
2. **Anytime**: Can improve with more computation time
3. **Theoretically Sound**: Converges to optimal policy given sufficient samples
4. **Model-Free**: Works with black-box simulators, no explicit model needed
5. **Memory Efficient**: Particle representation scales linearly with state size

This mathematical foundation enables POMCP to effectively solve large partially observable sequential decision problems like UNO.