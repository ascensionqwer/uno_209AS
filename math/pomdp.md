## POMDP Dynamics

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
