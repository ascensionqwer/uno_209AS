## Particle Filtering for Belief State Approximation

Since the state space is too large to enumerate all possible states, we use **particle filtering** (a form of Bayesian filtering) to maintain an approximate belief distribution. This approach generates particles dynamically at runtime, caching them per game state for efficiency.

**Player 1's Perspective:**

- Player 1 observes: $O = (H_1, |H_2|, |D_g|, P, P_t, G_o)$
- Player 1 maintains beliefs about Player 2's hidden hand $H_2$ using particles
- Each particle $s^{(i)} = (H_1, H_2^{(i)}, D_g^{(i)}, P, P_t, G_o)$ represents a possible complete state
- After Player 1 acts, system dynamics handle Player 2's turn
- Player 1 observes the new state and updates particles based on what Player 2 did

### Particle-Based Belief Representation

Instead of maintaining $b(s) = \Pr(s \mid o, \text{history})$ over all states $s \in S$, we maintain a **particle set**:

$\mathcal{P} = \{(s^{(i)}, w^{(i)})\}_{i=1}^{N}$

where:

- $N$ is the number of particles (e.g., $N = 1000$ or $N = 5000$)
- $s^{(i)} = (H_1, H_2^{(i)}, D_g^{(i)}, P, P_t, G_o)$ is a sampled state (particle)
- $w^{(i)} \geq 0$ is the importance weight of particle $i$
- $\sum_{i=1}^{N} w^{(i)} = 1$ (normalized weights)

The key insight: we only need to sample $H_2^{(i)}$ (opponent's hand) since $H_1, P, P_t, G_o$ are observable.

### Dynamic Particle Generation

Particles are generated **dynamically at runtime** based on the current observation. For a given observation $O = (H_1, |H_2|, |D_g|, P, P_t, G_o)$:

1. Compute $L = D \setminus H_1 \setminus P$ - available cards for opponent/deck
2. Enforce constraint: $|L| = |H_2| + |D_g|$ (all cards must be accounted for)
3. For each particle $i = 1, \ldots, N$:
   - Sample $H_2^{(i)}$ uniformly from all $\binom{|L|}{|H_2|}$ possible $|H_2|$-card subsets of $L$
   - Set $D_g^{(i)} = L \setminus H_2^{(i)}$ (all remaining cards go to deck)
   - Set $w^{(i)} = \frac{1}{N}$ (uniform initial weights)

**Critical constraint**: The deck size must satisfy $|D_g| = |L| - |H_2|$. If a parameter $deck\_size$ is provided that doesn't match this constraint, it is automatically corrected to $|L| - |H_2|$.

### Bayesian Filtering Update (Particle Filter)

When an observation $o$ is received after action $a$, update particles using **Bayesian filtering**:

**Step 1: Prediction (State Transition)**

For each particle $(s^{(i)}, w^{(i)})$:

- Sample $s'^{(i)} \sim T(\cdot \mid s^{(i)}, a)$
- Keep weight $w^{(i)}$ (temporarily)

**Step 2: Update (Observation Likelihood)**

For each particle $s'^{(i)}$:

- Compute observation likelihood: $w'^{(i)} = w^{(i)} \cdot O(o \mid s'^{(i)}, a)$
- This weights particles by how well they match the observation

**Step 3: Normalization**

Normalize weights:
$w^{(i)} = \frac{w'^{(i)}}{\sum_{j=1}^{N} w'^{(j)}}$

**Step 4: Resampling (Optional but Recommended)**

If effective sample size is too low:
$\text{ESS} = \frac{1}{\sum_{i=1}^{N} (w^{(i)})^2}$

If $\text{ESS} < \frac{N}{2}$ (or similar threshold):

- Resample $N$ particles with replacement from current particles, weighted by $w^{(i)}$
- Reset weights to $w^{(i)} = \frac{1}{N}$

### Specific Update Cases

**Case 1: Opponent plays card $z$**

Observation: opponent played card $z$, $|H_2'| = |H_2| - 1$

For each particle:

- If $z \notin H_2^{(i)}$: set $w^{(i)} = 0$ (inconsistent particle)
- If $z \in H_2^{(i)}$:
  - $H_2'^{(i)} = H_2^{(i)} \setminus \{z\}$
  - $P' = P \cup \{z\}$
  - Update $P_t'$ based on $z$ (handle WILD/+4 color choice)
  - $D_g'^{(i)} = D_g^{(i)}$ (unchanged)
  - $w^{(i)} = w^{(i)}$ (weight unchanged, observation is deterministic)

After update, normalize weights and resample if needed.

**Case 2: Opponent draws card (no legal move)**

Observation: opponent has no legal card, draws 1 card, $|H_2'| = |H_2| + 1$

For each particle:

- Check if $H_2^{(i)} \cap \text{LEGAL}(P_t) = \emptyset$ (no legal cards)
- If false: set $w^{(i)} = 0$ (inconsistent)
- If true:
  - Sample $c^{(i)} \sim \text{Uniform}(D_g^{(i)})$ (card drawn)
  - $H_2'^{(i)} = H_2^{(i)} \cup \{c^{(i)}\}$
  - $D_g'^{(i)} = D_g^{(i)} \setminus \{c^{(i)}\}$
  - $w^{(i)} = w^{(i)} \cdot \frac{1}{|D_g^{(i)}|}$ (likelihood of drawing that card)

Normalize and resample.

**Case 3: Opponent forced to draw by +2/+4**

Observation: $P_t(\text{VALUE}) \in \{+2, +4\}$, opponent draws $n$ cards, $|H_2'| = |H_2| + n$

For each particle:

- Sample $J_n^{(i)} \subset D_g^{(i)}$ with $|J_n^{(i)}| = n$ uniformly
- $H_2'^{(i)} = H_2^{(i)} \cup J_n^{(i)}$
- $D_g'^{(i)} = D_g^{(i)} \setminus J_n^{(i)}$
- $w^{(i)} = w^{(i)} \cdot \frac{1}{\binom{|D_g^{(i)}|}{n}}$ (likelihood of drawing those cards)

Normalize and resample.

**Case 4: Opponent's turn skipped**

Observation: $P_t(\text{VALUE}) \in \{\text{SKIP}, \text{REVERSE}\}$, $|H_2'| = |H_2|$

For each particle:

- $H_2'^{(i)} = H_2^{(i)}$ (unchanged)
- $D_g'^{(i)} = D_g^{(i)}$ (unchanged)
- $w^{(i)} = w^{(i)}$ (unchanged)

No update needed.

### Reshuffle Handling

When $D_g$ is empty:

- For each particle: $D_g'^{(i)} = P \setminus \{P_t\}$ (reshuffle played cards)
- Update $L = D \setminus H_1 \setminus P$ and resample $H_2^{(i)}$ if needed to maintain consistency

### Approximate Value Function

Instead of exact value function:
$V^*(b) = \max_{a \in A} \sum_{s} b(s) \sum_{s'} T(s' \mid s, a) \sum_{o} O(o \mid s', a) \left[ R(s', s, a) + \gamma V^*(b') \right]$

We use particle-based approximation:

$\hat{V}^*(b) = \max_{a \in A} \sum_{i=1}^{N} w^{(i)} \sum_{s'} T(s' \mid s^{(i)}, a) \sum_{o} O(o \mid s', a) \left[ R(s', s^{(i)}, a) + \gamma \hat{V}^*(b') \right]$

where $b'$ is the updated particle-based belief after action $a$ and observation $o$, and $\gamma$ (ex. 0.95) is the discount factor.

### Monte Carlo Tree Search (MCTS) with Particle Filter

For decision-making, combine particle filtering with **MCTS**:

1. **Selection**: Traverse tree using UCB1, using particle-based belief at each node
2. **Expansion**: Add new node for unexplored action
3. **Simulation**: Roll out using particle samples to estimate value
4. **Backpropagation**: Update node values based on simulation result

At each node, maintain particle set $\mathcal{P}$ representing belief over opponent's hand.

### Computational Complexity

- **Exact approach**: $O(|\mathcal{S}|)$ where $|\mathcal{S}|$ is exponential in deck size
- **Particle filter**: $O(N \cdot |A| \cdot H)$ where:
  - $N$ = number of particles (e.g., 1000-10000)
  - $|A|$ = action space size (typically small, ~10-20)
  - $H$ = planning horizon (depth of lookahead)

This makes the problem tractable for real-time play.

## Caching Strategy

Particles are cached per **game state** to avoid redundant generation. A game state key includes:

- $H_1$ (player's hand)
- $|H_2|$ (opponent hand size)
- $|D_g|$ (deck size)
- $P$ (played cards - implicitly contains action history of all play actions)
- $P_t$ (top card)
- $G_o$ (game over status)

**Note on action history:** The pile $P$ implicitly encodes the action history of all play actions, as every card played is added to $P$. While action history is part of the state definition through $P$, we don't need to track it explicitly for disambiguation since $P$ captures this information. Draw actions don't affect $P$ but are reflected in the hand sizes and deck size.

When the same game state is encountered, cached particles are reused, significantly improving performance.
