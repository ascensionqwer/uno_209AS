## Mathematical Framework for UNO POMDP (Player 1 Perspective)

This document presents the mathematical framework for a 2-player UNO game from **Player 1's perspective**. Player 1 observes partial information about the game state and maintains beliefs about Player 2's hidden hand using particle filtering. System dynamics handle Player 2's actions, which Player 1 observes and uses to update beliefs.

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

## POMDP Formulation

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

### Transition Function

$T(s' \mid s, a) = \begin{cases}
1 & a = X_1, X_1 \in H_1 \cap \text{LEGAL}(P_t), \\
  & H_1' = H_1 \setminus \{X_1\}, P' = P \cup \{X_1\}, \\
  & P_t' = \begin{cases} (\text{CHOSEN\_COLOR}, \text{VALUE}) & \text{if } X_1(\text{VALUE}) \in \{\text{WILD}, +4\} \\ X_1 & \text{otherwise} \end{cases}, \\
  & G_o' = \begin{cases} \text{GameOver} & \text{if } |H_1'| = 0 \\ \text{Active} & \text{otherwise} \end{cases} \\[0.5em]
\frac{1}{|D_g|} & a = Y_1, H_1 \cap \text{LEGAL}(P_t) = \emptyset, \\
  & P_t(\text{VALUE}) \notin \{+2, +4, \text{SKIP}, \text{REVERSE}\}, \\
  & H_1' = H_1 \cup \{c\} \text{ for some } c \in D_g, D_g' = D_g \setminus \{c\} \\[0.5em]
\frac{1}{\binom{|D_g|}{n}} & a = Y_n, P_t(\text{VALUE}) \in \{+2, +4\}, n \in \{2, 4\}, \\
  & H_1' = H_1 \cup J_n \text{ for some } J_n \subset D_g \text{ with } |J_n| = n, D_g' = D_g \setminus J_n, \\
  & P_t' = P_t \text{ (top card unchanged, effect resolved)} \\[0.5em]
0 & \text{All other cases, including } G_o = \text{GameOver}
\end{cases}$

**Note on SKIP/REVERSE:** When a SKIP or REVERSE card is played (by opponent), Player 1's turn is automatically skipped. This is handled in the game flow, not as a separate action. After opponent plays SKIP/REVERSE, Player 1's next turn is skipped and control returns to opponent.

### Legal Play Function

$\text{LEGAL}(P_t)$ - set of all cards legally playable on top of $P_t$

$\text{LEGAL}(P_t) = \left\{c \in D : 
\begin{array}{l}
c(\text{COLOR}) = P_t(\text{COLOR}) \lor \\
c(\text{VALUE}) = P_t(\text{VALUE}) \lor \\
c(\text{COLOR}) = \text{BLACK} \text{ (WILD and +4 can always be played)}
\end{array}
\right\}$

Special card values: $\text{VALUE} \in \{0, 1, \ldots, 9, \text{SKIP}, \text{REVERSE}, +2, \text{WILD}, +4\}$

### Reward Function

$R(s', s, a) = \begin{cases}
+1 & |H_1'| = 0 \text{ (player 1 wins)} \\
-1 & |H_2'| = 0 \text{ (player 2 wins)} \\
0 & \text{otherwise}
\end{cases}$

### Observation Function

$O(o \mid s', a)$ - observation function, probability of observation $o$ given state $s'$ and action $a$

For this UNO POMDP, observations are deterministic given the state:

$O(o \mid s', a) = \begin{cases}
1 & o = (H_1', |H_2'|, |D_g'|, P', P_t', G_o') \text{ (observation matches state)} \\
0 & \text{otherwise}
\end{cases}$

where $o \in O$ and $s' \in S$.

**Special card observations:**

- When opponent plays WILD/+4: $P_t'$ shows the chosen color (observable).
- When opponent plays SKIP/REVERSE: $P_t'$ shows the SKIP/REVERSE card, and Player 1's turn is skipped (observable through game flow).
- When $P_t(\text{VALUE}) \in \{+2, +4\}$: Player 1 observes this and must draw accordingly (no legal play option).

## Particle Filtering for Belief State Approximation (Player 1 Perspective)

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

## Value Function and Decision Making

### Approximate Value Function

Instead of exact value function:
$V^*(b) = \max_{a \in A} \sum_{s} b(s) \sum_{s'} T(s' \mid s, a) \sum_{o} O(o \mid s', a) \left[ R(s', s, a) + \gamma V^*(b') \right]$

We use particle-based approximation:

$\hat{V}^*(b) = \max_{a \in A} \sum_{i=1}^{N} w^{(i)} \sum_{s'} T(s' \mid s^{(i)}, a) \sum_{o} O(o \mid s', a) \left[ R(s', s^{(i)}, a) + \gamma \hat{V}^*(b') \right]$

where $b'$ is the updated particle-based belief after action $a$ and observation $o$, and $\gamma = 0.95$ is the discount factor.

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
- $P$ (played cards)
- $P_t$ (top card)
- $G_o$ (game over status)
- Action history (for disambiguation)

When the same game state is encountered, cached particles are reused, significantly improving performance.

## Advantages of Dynamic Particle Filtering

1. **Tractable**: Only maintains $N$ samples instead of full state distribution
2. **Adaptive**: Automatically focuses particles on likely states
3. **Robust**: Handles non-linear/non-Gaussian distributions
4. **Scalable**: Can adjust $N$ based on computational budget
5. **Efficient**: Runtime generation with caching avoids precomputation overhead
6. **Memory-efficient**: Cache grows naturally with unique game states encountered

## Implementation Notes

- Use **systematic resampling** for better diversity
- **Rao-Blackwellization**: If some components are analytically tractable, marginalize them out
- **Adaptive particle count**: Increase $N$ in critical situations (e.g., near endgame)
- **Particle diversity**: Monitor and prevent particle collapse (all particles identical)
- **Cache management**: No size limits - cache grows naturally with game play
- **Canonical state representation**: Use sorted, canonical forms for cache keys
