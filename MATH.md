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

## Math for 2 Player Uno Game

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

<!-- $T(s' \mid s, a) = \begin{cases}
1 & H_1' = H_1 - X_1, X_1 \in \text{LEGAL}(P_t), X_1 \in H_1, G_o' = \text{Active} \text{ (played legal move)}\\
1 & H_1' = H_1 + Y_1, H_1 \cap \text{LEGAL}(P_t) = \emptyset, G_o' = \text{Active} \text{ (no legal moves; draw 1 card)}\\
1 & H_1' = H_1 + Y_n, P_t(\text{VALUE}) \in \{+2, +4\}, G_o' = \text{Active} \text{ (top card is +2 / +4)}\\
0 & \text{otherwise} \\
\end{cases}$ -->

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

- Cases:
  - Play 1 card, that card is in hand and legally playable, the new hand is the existing hand minus the played card, the updated pile contains the played card. For WILD/+4 cards, the top card color becomes the chosen color (not BLACK). For regular cards, the top card is the played card. Special effects: SKIP/REVERSE skip next player's turn; +2/+4 force next player to draw.
  - Draw 1 card, no legally playable moves AND top card is not a special action card (+2, +4, SKIP, REVERSE), the new hand is the existing hand plus the drawn card, the drawn card was part of the deck, the new deck is the existing deck minus the drawn card.
  - Draw $n$ cards ($n \in \{2, 4\}$), the top card is a +2/+4 card, the new hand is the existing hand plus all drawn cards ($J_n$), all drawn cards were part of the deck, the number of drawn cards matches $n$ (2 for +2, 4 for +4), the new deck is the existing deck minus the drawn cards, the top card remains unchanged after resolving the effect.
  - All other scenarios are invalid (especially game over)
- $s, s' \in S, a \in A$

<br/>
$\text{LEGAL}(P_t)$ - set of all cards legally playable on top of $P_t$

$\text{LEGAL}(P_t) = \left\{c \in D : 
\begin{array}{l}
c(\text{COLOR}) = P_t(\text{COLOR}) \lor \\
c(\text{VALUE}) = P_t(\text{VALUE}) \lor \\
c(\text{COLOR}) = \text{BLACK} \text{ (WILD and +4 can always be played)}
\end{array}
\right\}$

Special card values: $\text{VALUE} \in \{0, 1, \ldots, 9, \text{SKIP}, \text{REVERSE}, +2, \text{WILD}, +4\}$

$R(s', s, a) = \begin{cases}
+1 & |H_1'| = 0 \text{ (player 1 wins)} \\
-1 & |H_2'| = 0 \text{ (player 2 wins)} \\
0 & \text{otherwise}
\end{cases}$

<br/>

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

<br/>

$B$ - Belief space

$b(s)$ - belief state, probability distribution over states $s \in S$

- $b: S \to [0,1]$ where $\sum_{s \in S} b(s) = 1$
- $b(s) = \Pr(s \mid o, \text{history})$ - probability of state $s$ given observation $o$ and action history
- For Player 1, $b(s) = \Pr((H_1, H_2, D_g, P, P_t, G_o) \mid (H_1, |H_2|, |D_g|, P, P_t, G_o), \text{history})$
- Belief state $b \in B$ where $B$ is the space of all probability distributions over $S$

<br/>

**Belief Update**

Let $bel^-(s)$ denote the prior belief (before observation) and $bel^+(s)$ denote the posterior belief (after observation).

For Player 1, the unknown component is $H_2$ (opponent's hand). Let:

- $L = D \setminus H_1 \setminus P$ - set of cards that could be in opponent's hand or deck
- $k = |H_2|$ - opponent's hand size
- $N(P_t) = \text{LEGAL}(P_t) \cap L$ - legal cards available to opponent

**Prior Belief (Before Observation):**

$bel^-(H_2) = \begin{cases}
\frac{1}{\binom{|L|}{k}} & H_2 \subseteq L, |H_2| = k \\
0 & \text{otherwise}
\end{cases}$

This assumes uniform distribution over all possible $k$-card subsets of $L$.

**Posterior Belief (After Observation):**

**Case 1: Opponent plays a card**

Observation: opponent played card $z$

- $k' = k - 1$ (opponent hand size decreases)
- $P' = P \cup \{z\}$ (card added to pile)
- $P_t' = \begin{cases} (\text{CHOSEN\_COLOR}, \text{VALUE}) & \text{if } z(\text{VALUE}) \in \{\text{WILD}, +4\} \\ z & \text{otherwise} \end{cases}$ (new top card; WILD/+4 set chosen color)
- $L' = D \setminus (H_1 \cup P')$ (update available cards)
- Constraint: $z \in H_2$ (played card must have been in opponent's hand)

$bel^+(H_2' \mid z) = \begin{cases}
\frac{1}{\binom{|L'|}{k'}} & H_2' \subseteq L', |H_2'| = k', z \notin H_2' \\
0 & \text{otherwise}
\end{cases}$

The posterior is uniform over all $(k-1)$-card subsets of $L'$ (since $z$ was removed).

**Special card effects:**

- If $z(\text{VALUE}) \in \{+2, +4\}$: Next player (Player 1) must draw $n$ cards where $n = 2$ for +2, $n = 4$ for +4.
- If $z(\text{VALUE}) \in \{\text{SKIP}, \text{REVERSE}\}$: Next player's turn is skipped (in 2-player, REVERSE = SKIP).
- If $z(\text{VALUE}) \in \{\text{WILD}, +4\}$: The chosen color becomes the new $P_t(\text{COLOR})$ for determining legal plays.

**Case 2: Opponent draws a card (no legal move available)**

Observation: opponent has no legal card to play, draws one card

This case only applies when $P_t(\text{VALUE}) \notin \{+2, +4, \text{SKIP}, \text{REVERSE}\}$ (opponent can draw normally, not forced by special card).

Probability that opponent has no legal card:

$\Pr(\text{no\_legal}) = \frac{\binom{|L| - |N(P_t)|}{k}}{\binom{|L|}{k}}$

where $N(P_t) = \text{LEGAL}(P_t) \cap L$ is the set of legal cards available to opponent.

**Before drawing (after observation, before draw):**

$bel^+(H_2 \mid \text{no\_legal}) = \begin{cases}
\frac{1}{\binom{|L| - |N(P_t)|}{k}} & H_2 \subseteq L \setminus N(P_t), |H_2| = k \\
0 & \text{otherwise}
\end{cases}$

**After drawing (updated prior for next turn):**

$bel^-(H_2' \mid \text{no\_legal} + \text{draw}) = \begin{cases}
\frac{\binom{|L| - |N(P_t)| - (k+1)}{|D_g| - 1}}{\binom{|L| - |N(P_t)| - k}{|D_g|}} & H_2' \subseteq L \setminus N(P_t), |H_2'| = k+1 \\
0 & \text{otherwise}
\end{cases}$

This accounts for the fact that opponent drew one card from $D_g$, increasing hand size to $k+1$.

**Case 3: Opponent forced to draw by +2/+4**

Observation: $P_t(\text{VALUE}) \in \{+2, +4\}$, opponent draws $n$ cards where $n = 2$ for +2, $n = 4$ for +4.

- $k' = k + n$ (opponent hand size increases by $n$)
- $P_t$ remains unchanged (opponent did not play)
- $L' = D \setminus (H_1 \cup P)$ (available cards unchanged, but opponent drew from $D_g$)

$bel^-(H_2' \mid \text{forced\_draw}) = \begin{cases}
\frac{1}{\binom{|D_g|}{n}} & H_2' \subseteq L, |H_2'| = k', H_2 \subset H_2' \text{ for some } H_2 \text{ with } |H_2| = k \\
0 & \text{otherwise}
\end{cases}$

This accounts for opponent drawing $n$ cards from $D_g$, increasing hand size from $k$ to $k+n$.

**Case 4: Opponent's turn skipped by SKIP/REVERSE**

Observation: $P_t(\text{VALUE}) \in \{\text{SKIP}, \text{REVERSE}\}$, opponent's turn is skipped.

- $k' = k$ (opponent hand size unchanged)
- $P_t$ remains unchanged
- No cards drawn, no cards played

$bel^+(H_2' \mid \text{skipped}) = \begin{cases}
1 & H_2' = H_2, |H_2'| = k \\
0 & \text{otherwise}
\end{cases}$

The belief remains unchanged since no action was taken.

**Reshuffle Handling:**

When $D_g$ is empty, move $P \setminus \{P_t\}$ back into $D_g$, then compute $L = D \setminus (H_1 \cup P)$ and reset the prior belief accordingly using the updated $L$ and $k$.

<br/>

**Value Function and Expected Discounted Reward**

Let $\gamma = 0.95$ be the discount factor.

**Value Function:**

$V^*(b)$ - optimal value function for belief state $b$

$V^*(b) = \max_{a \in A} \sum_{s} b(s) \sum_{s'} T(s' \mid s, a) \sum_{o} O(o \mid s', a) \left[ R(s', s, a) + \gamma V^*(b') \right]$

where $b'$ is the updated belief state after taking action $a$ and observing $o$:

$b'(s') = \frac{O(o \mid s', a) \sum_{s} T(s' \mid s, a) b(s)}{\sum_{s''} O(o \mid s'', a) \sum_{s} T(s'' \mid s, a) b(s)}$

The value function represents the maximum expected discounted reward achievable from belief state $b$ under an optimal policy. The opponent's actions are treated as stochastic based on their unknown hand distribution (captured in the belief state), not as a fixed policy.

**Expected Discounted Reward:**

For a policy $\pi: B \to A$ (mapping belief states to actions), the expected discounted reward starting from belief $b_0$ is:

$E\left[\sum_{t=0}^{\infty} \gamma^t R_t \mid b_0, \pi\right] = V^\pi(b_0)$

where $R_t$ is the reward at time step $t$ and $\gamma = 0.95$.

The value function $V^\pi(b)$ for policy $\pi$ satisfies:

$V^\pi(b) = \sum_{s} b(s) \sum_{s'} T(s' \mid s, \pi(b)) \sum_{o} O(o \mid s', \pi(b)) \left[ R(s', s, \pi(b)) + \gamma V^\pi(b') \right]$

The optimal value function $V^*(b)$ is the maximum over all policies:

$V^*(b) = \max_{\pi} V^\pi(b)$
