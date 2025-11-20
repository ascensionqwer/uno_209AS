## Math

$D = H_1 \sqcup H_2 \sqcup D_g \sqcup P$ - entire 108 uno card deck

- $H_1, H_2$ - player hands
- $D_g$ - current card deck SET (unordered) after distributions / actions / etc.
- $P$ - already played cards.

$S = (H_1, H_2, D_g, P, P_t, G_o)$ - system state (as seen by "God")

- $D_g$ - current card deck SET (unordered)
- $P_t = (\text{COLOR}, \text{VALUE}) \in P$ - top card of the pile; order of other cards in the pile doesn't matter
- $G_o \in \{\text{Active},\text{GameOver}\}$ - is game over or not

$A = \{X_1\} \cup \{Y_n : n \in \{1, 2, 4\}\}$ - action space

- $X_1 = (\text{COLOR}, \text{VALUE})$ - card played from hand
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
  & H_1' = H_1 \setminus \{X_1\}, P' = P \cup \{X_1\}, P_t' = X_1, \\
  & G_o' = \begin{cases} \text{GameOver} & \text{if } |H_1'| = 0 \\ \text{Active} & \text{otherwise} \end{cases} \\[0.5em]
\frac{1}{|D_g|} & a = Y_1, H_1 \cap \text{LEGAL}(P_t) = \emptyset, \\
  & H_1' = H_1 \cup \{c\} \text{ for some } c \in D_g, D_g' = D_g \setminus \{c\} \\[0.5em]
\frac{1}{\binom{|D_g|}{n}} & a = Y_n, P_t(\text{VALUE}) \in \{+2, +4\}, \\
  & H_1' = H_1 \cup J_n \text{ for some } J_n \subset D_g \text{ with } |J_n| = n, D_g' = D_g \setminus J_n \\[0.5em]
0 & \text{All other cases, including } G_o = \text{GameOver}
\end{cases}$

- Cases:
  - Play 1 card, that card is in hand and legally playable, the new hand is the existing hand minus the played card, the updated pile contains the played card, the updated top card in pile is the played card.
  - Draw 1 card, no legally playable moves, the new hand is the existing hand plus the drawn card, the drawn card was part of the deck, the new deck is the existing deck minus the drawn card.
  - Draw $n$ cards, the top card is a +2/+4 card, the new hand is the existing hand plus all drawn cards ($J_n$), all drawn cards were part of the deck, the number of drawn cards matches $n$, the new deck is the existing deck minus the drawn cards.
  - All other scenarios are invalid (especially game over)
- $s, s' \in S, a \in A$

<br/>
$LEGAL(P_t)$ - set of all cards legally playable on top of $P_t$

$\text{LEGAL}(P_t) = \left\{c \in D : 
\begin{array}{l}
c(\text{COLOR}) = P_t(\text{COLOR}) \lor \\
c(\text{VALUE}) = P_t(\text{VALUE}) \lor \\
c(\text{COLOR}) = \text{BLACK}
\end{array}
\right\}$

$R(s', s, a) = \begin{cases}
+1 & |H_1'| = 0 \text{ (player 1 wins)} \\
-1 & |H_2'| = 0 \text{ (player 2 wins)} \\
0 & \text{otherwise}
\end{cases}$

<br/>

$B$ - Belief space

<!--Following cards cannot be in $H_2$ or $D_g$
 - Cards in $H_1$
 - Cards in $P$

When we reshuffle $P - P_t$ back in to the deck
 - Cards in $H_2$ becomes known as $H_2 = D - H_1 - P$

// I don't know how opponents legal plays needs to be expressed as

1. Opponent not having legal play (having to draw 1) would deduce the contents of $H_2$
ex) If $P_t = y_3$ and opponent draws 1, $y_n$ when $(n = 0, 1, ... , 9) \notin L(P_t), \set{r3, g3, b3} \notin L(P_t)$

$b_t = Pr(H_2, D_g \mid O)$

$L = D - (H_1 \cup P)$
$b_t(H_2)$ over all $H_2 \subseteq L, |H_2| = k$
-->

Belief-

Prior- Before Observation:

- $bel^-(H_2) = \frac{1}{C_k^{|L|}} (L= D\setminus H_1 \setminus P ; k= |H_2|)

Posterior- Updated after observation:

-Case1- Opponent does have legal card to play

z: opponent played card

$k=k'=k−1;\quad P=P'=P\cup \{z\}; \quad P_t=P_t'=z; \quad L=L'=D\setminus (H_1\cup P) \quad ,z \in H_2

New turns with new $L$ and $k$ and reset prior

Case2- Opponent doesn't have legal card to play, and draw a card

Probibility of no legal card:\quad $Pr(no_legal)= \frac{C_k^{|L|-|N(P_t)|}}{C_k^{|L|}} \quad ,N(P_t)= Legal(P_t) \cap L

Before drawing card (After observation)

- $bel^+(H_2|no_legal)= \frac{1}{C_k^{|L|-|N(P_t)|}}, H_2\subseteq L\setminus N(P_t)

After drawing card

- $bel^-(H_2'|no\_legal+draw)= \frac{C_{|L|-|N(P_t)|-(k+1)}^{|D_g|-1}}{C_{|L|-|N(P_t)|-k}^{|D_g|}}, H_2'\subseteq L\setminus N(P_t), |H_2'|=k+1$

If need reshuffle:

when $D_g$ is empty, move $P\setminus\{P_t\}$ back into $D_g$, then compute $L= D\setminus (H_1\cup P)$ and reset the prior accordingly.

r: category counts in the current remainder

k: opponent hand size
