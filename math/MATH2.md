## Rules

- 2 players

- 108 card deck
  - 25 red, blue, yellow, green
    - 2x 1-9
    - 1x 0
- 2x SKIP - The next player's turn is forfeited.
- 2x REVERSE - The next player's turn is forfeited. (In 2-player UNO, REVERSE acts identically to SKIP.)
  - 2x +2 - The next player forfeits their turn and draws 2 cards.
  - 8 black
    - 4 WILD - Changes the pile color. Other players must play a card respecting the color.
    - 4 +4 - The next player forfeits their turn and draws 4 cards.

<br/>

- Each player starts with 7 cards randomly distributed. A player cannot see the other player's cards.

- The deck is hidden from all player's viewpoint. 1 card from the deck becomes the top card of the pile.

- When playing a card, either the number AND/OR the color need to match. This card then goes into the pile. Pile is visible to everyone.
  - If a player does not have a legal card to play, they must draw 1 card from the deck.
  - If a player can make a legal card play, they must play that card no matter what.

- When a +2 or +4 is played, the next player may not play a card, and instead draws the corresponding number of cards. This effect cannot stack with itself.

## Math for 2 Player Uno Game (Bayesian Filtering Approach)

$D = H_1 \sqcup H_2 \sqcup D_g \sqcup P$ - entire 108 uno card deck

- $H_1, H_2$ - player hands
- $D_g$ - current card deck SET (unordered) after distributions / actions / etc.
- $P$ - already played cards.

$S = (H_1, H_2, D_g, P, P_t, G_o)$ - system state (as seen by "God")

- $D_g$ - current card deck SET (unordered)
- $P_t = (\text{COLOR}, \text{VALUE}) \in P$ - top card of the pile; order of other cards in the pile doesn't matter
  - For WILD/+4: $P_t = (\text{CHOSEN\_COLOR}, \text{VALUE})$ where $\text{CHOSEN\_COLOR}$ is the color chosen by the player who played the WILD/+4
- $G_o \in \{\text{Active},\text{GameOver}\}$ - is game over or not

$A = \{X_1\} \cup \{Y_n : n \in \{1, 2, 4\}\}$ - action space

- $X_1 = (\text{COLOR}, \text{VALUE})$ - card played from hand
  - For WILD cards: $X_1 = (\text{BLACK}, \text{WILD}, \text{CHOSEN\_COLOR})$ where $\text{CHOSEN\_COLOR} \in \{\text{RED}, \text{BLUE}, \text{YELLOW}, \text{GREEN}\}$
  - For +4 cards: $X_1 = (\text{BLACK}, +4, \text{CHOSEN\_COLOR})$ where $\text{CHOSEN\_COLOR} \in \{\text{RED}, \text{BLUE}, \text{YELLOW}, \text{GREEN}\}$
- $Y_n = (\text{COLOR}, \text{VALUE})^n$ - cards drawn from deck
  - $n = \text{number of cards drawn}$

$O = (H_1, |H_2|, |D_g|, P, P_t, G_o)$ - observation (as seen by "Player 1")

- $H_1$ - my hand
- $|H_2|$ - num cards in opponent's hand
- $|D_g|$ - num cards remaining in deck
- $P_t = (\text{COLOR}, \text{VALUE}) \in P$ - top card of the pile; order of other cards in the pile don't matter
- $G_o$ - is game over or not

$T(s' \mid s, a)$ - transition function (same as in MATH.md)

$\text{LEGAL}(P_t)$ - set of all cards legally playable on top of $P_t$ (same as in MATH.md)

$R(s', s, a)$ - reward function (same as in MATH.md)

$O(o \mid s', a)$ - observation function (same as in MATH.md)

<br/>

## Bayesian Filtering for Belief State Approximation

Since the state space is too large to enumerate all possible states, we use **particle filtering** (a form of Bayesian filtering) to maintain an approximate belief distribution.

### Particle-Based Belief Representation

Instead of maintaining $b(s) = \Pr(s \mid o, \text{history})$ over all states $s \in S$, we maintain a **particle set**:

$\mathcal{P} = \{(s^{(i)}, w^{(i)})\}_{i=1}^{N}$

where:

- $N$ is the number of particles (e.g., $N = 1000$ or $N = 5000$)
- $s^{(i)} = (H_1, H_2^{(i)}, D_g^{(i)}, P, P_t, G_o)$ is a sampled state (particle)
- $w^{(i)} \geq 0$ is the importance weight of particle $i$
- $\sum_{i=1}^{N} w^{(i)} = 1$ (normalized weights)

The key insight: we only need to sample $H_2^{(i)}$ (opponent's hand) since $H_1, P, P_t, G_o$ are observable.

### Initial Belief (Particle Initialization)

At game start:

- $H_1$ is known (Player 1's hand)
- $P$ is known (initial pile card)
- $|H_2| = 7$ (opponent starts with 7 cards)
- $L = D \setminus H_1 \setminus P$ - available cards for opponent/deck

**Initialization:**

1. For each particle $i = 1, \ldots, N$:
   - Sample $H_2^{(i)}$ uniformly from all $\binom{|L|}{7}$ possible 7-card subsets of $L$
   - Set $D_g^{(i)} = L \setminus H_2^{(i)}$
   - Set $w^{(i)} = \frac{1}{N}$ (uniform initial weights)

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

### Approximate Value Function

Instead of exact value function:
$V^*(b) = \max_{a \in A} \sum_{s} b(s) \sum_{s'} T(s' \mid s, a) \sum_{o} O(o \mid s', a) \left[ R(s', s, a) + \gamma V^*(b') \right]$

We use particle-based approximation:

$\hat{V}^*(b) = \max_{a \in A} \sum_{i=1}^{N} w^{(i)} \sum_{s'} T(s' \mid s^{(i)}, a) \sum_{o} O(o \mid s', a) \left[ R(s', s^{(i)}, a) + \gamma \hat{V}^*(b') \right]$

where $b'$ is the updated particle-based belief after action $a$ and observation $o$.

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

### Reshuffle Handling

When $D_g$ is empty:

- For each particle: $D_g'^{(i)} = P \setminus \{P_t\}$ (reshuffle played cards)
- Update $L = D \setminus H_1 \setminus P$ and resample $H_2^{(i)}$ if needed to maintain consistency

### Advantages of Particle Filtering

1. **Tractable**: Only maintains $N$ samples instead of full state distribution
2. **Adaptive**: Automatically focuses particles on likely states
3. **Robust**: Handles non-linear/non-Gaussian distributions
4. **Scalable**: Can adjust $N$ based on computational budget

### Implementation Notes

- Use **systematic resampling** for better diversity
- **Rao-Blackwellization**: If some components are analytically tractable, marginalize them out
- **Adaptive particle count**: Increase $N$ in critical situations (e.g., near endgame)
- **Particle diversity**: Monitor and prevent particle collapse (all particles identical)
