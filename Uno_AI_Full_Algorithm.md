# Complete Mathematical Algorithm: Uno AI with Expectiminimax

## 1. Problem Formulation: Partially Observable Zero-Sum Game

### 1.1 Game Model
Uno is a **two-player zero-sum game** with **imperfect information**:

\[ \mathcal{G} = \langle S, A, T, Z, p_1, p_2, R \rangle \]

Where:
- **\( S \)**: State space - \( s = (H_1, H_2, D_g, P, P_t, G_o) \)
- **\( A \)**: Action space - \( a \in A_{\text{play}} \cup A_{\text{draw}} \)
- **\( T(s' | s, a) \)**: State transition function (deterministic for play, stochastic for draw)
- **\( Z(s) \)**: Observation function - \( o = (H_1, |H_2|, |D_g|, P, P_t, G_o) \)
- **\( p_1, p_2 \)**: Two players (Player 1 = AI, Player 2 = Opponent)
- **\( R(s) \)**: Reward function (terminal payoff)

### 1.2 Zero-Sum Property

\[
R_{\text{AI}}(s) + R_{\text{opp}}(s) = 0
\]

When AI wins, opponent loses. Value from opponent's perspective is negative of AI's value.

---

## 2. Value Function: Expectiminimax Recursion

### 2.1 Recursive Definition

Define the **value function**:

\[
V(s, d, p) = \text{expected value of state } s \text{ at depth } d \text{ when player } p \text{ is to move}
\]

**Measured from Player 1 (AI) perspective**, so:
- Positive value = AI is winning
- Negative value = Opponent is winning

### 2.2 Base Cases

**Terminal by depth**:
\[
V(s, 0, p) = R(s)
\]

**Terminal by game status**:
\[
\text{if } G_o = \text{GameOver}: \quad V(s, d, p) = R(s)
\]

### 2.3 Terminal Reward Function

\[
R(s) = \begin{cases}
+100 & \text{if } |H_1| = 0 \text{ (AI wins)} \\
-100 & \text{if } |H_2| = 0 \text{ (opponent wins)} \\
|H_2| - |H_1| & \text{if } G_o = \text{Active}
\end{cases}
\]

### 2.4 Maximizing Node (Player = AI)

When player \( p = \text{AI} \), the AI chooses actions to maximize expected value:

\[
V(s, d, \text{AI}) = \max_{a \in A(s)} \mathbb{E}_{s' \sim T(\cdot | s, a)} \left[ V(s', d-1, 3-\text{AI}) \right]
\]

Which simplifies to:

\[
V(s, d, \text{AI}) = \max_{a \in A(s)} V_{\text{after}}(s, a, d)
\]

### 2.5 Minimizing Node (Player = Opponent)

When player \( p = \text{opponent} \), the opponent chooses actions to minimize AI's expected value:

\[
V(s, d, \text{opp}) = \min_{a \in A(s)} \mathbb{E}_{s' \sim T(\cdot | s, a)} \left[ V(s', d-1, 3-\text{opp}) \right]
\]

Which simplifies to:

\[
V(s, d, \text{opp}) = \min_{a \in A(s)} V_{\text{after}}(s, a, d)
\]

### 2.6 Combined Minimax Formulation

\[
V(s, d, p) = \begin{cases}
R(s) & \text{if } d = 0 \text{ or } G_o = \text{GameOver} \\[1.5em]
\displaystyle\max_{a \in A(s)} V_{\text{after}}(s, a, d) & \text{if } p = \text{AI} \\[1.5em]
\displaystyle\min_{a \in A(s)} V_{\text{after}}(s, a, d) & \text{if } p = \text{opponent}
\end{cases}
\]

---

## 3. Action Outcome Value: \( V_{\text{after}} \)

### 3.1 Deterministic Actions (Play Card)

For play actions \( a = X_1 \) (card to play):

\[
V_{\text{after}}(s, X_1, d) = V(T(s, X_1), d-1, 3-p)
\]

Where the state transition is deterministic:

\[
T(s, X_1) = \begin{cases}
H_1' = H_1 \setminus \{X_1\} \\
H_2' = H_2 \\
D_g' = D_g \\
P' = P \cup \{X_1\} \\
P_t' = X_1 \\
G_o' = \begin{cases} \text{GameOver} & \text{if } |H_1'| = 0 \\ \text{Active} & \text{else} \end{cases}
\end{cases}
\]

### 3.2 Stochastic Actions (Draw Cards)

For draw actions \( a = Y_n \) (draw \( n \) cards):

\[
V_{\text{after}}(s, Y_n, d) = \mathbb{E}_{c \sim \text{Uniform}(D_g)} \left[ V(T(s, Y_n, c), d-1, 3-p) \right]
\]

Where the state transition is stochastic over the drawn card \( c \):

\[
T(s, Y_n, c) = \begin{cases}
H_1' = H_1 \cup \{c\} \\
H_2' = H_2 \\
D_g' = D_g \setminus \{c\} \\
P' = P \\
P_t' = P_t \text{ (unchanged)} \\
G_o' = \text{Active}
\end{cases}
\]

### 3.3 Monte Carlo Approximation for Draw Actions

Exact expectation over all \( |D_g| \) possible cards is intractable. Approximate via sampling:

\[
\mathbb{E}_{c \sim \text{Uniform}(D_g)}[V] \approx \frac{1}{m} \sum_{j=1}^m V(T(s, Y_n, c_j), d-1, 3-p)
\]

where \( m = \min(5, |D_g|) \) and \( c_1, \ldots, c_m \) are sampled uniformly without replacement from \( D_g \).

---

## 4. Algorithm: Expectiminimax Search

### 4.1 Pseudocode

```
FUNCTION expectiminimax(s, d, p):
    // Terminal conditions
    IF d == 0 OR G_o == GameOver:
        RETURN R(s)
    
    // Get legal actions for player p
    A ← GetLegalActions(s, p)
    IF A is empty:
        RETURN R(s)
    
    // Determine if maximizing or minimizing
    isMaximizing ← (p == AI)
    
    IF isMaximizing:
        value ← -∞
        FOR EACH action a IN A:
            IF a is deterministic (play):
                nextState ← T(s, a)
                actionValue ← expectiminimax(nextState, d-1, opponent)
            ELSE:  // draw action
                actionValue ← 0
                FOR j = 1 TO m:
                    card ← SampleUniform(D_g)
                    nextState ← T(s, a, card)
                    actionValue ← actionValue + expectiminimax(nextState, d-1, opponent)
                actionValue ← actionValue / m
            value ← max(value, actionValue)
        RETURN value
    
    ELSE:  // Minimizing (opponent)
        value ← +∞
        FOR EACH action a IN A:
            IF a is deterministic (play):
                nextState ← T(s, a)
                actionValue ← expectiminimax(nextState, d-1, AI)
            ELSE:  // draw action
                actionValue ← 0
                FOR j = 1 TO m:
                    card ← SampleUniform(D_g)
                    nextState ← T(s, a, card)
                    actionValue ← actionValue + expectiminimax(nextState, d-1, AI)
                actionValue ← actionValue / m
            value ← min(value, actionValue)
        RETURN value
```

### 4.2 Time Complexity Analysis

**Parameters**:
- \( b \) = branching factor (avg legal actions per state) ≈ 2
- \( d \) = search depth = 3
- \( m \) = draw samples = 5

**Call tree**:
\[
\text{Nodes} = O(b^d \cdot m) = O(2^3 \cdot 5) = O(40)
\]

**Per action evaluation** in `choose_action`:
\[
\text{Calls} = N \cdot 40 = 50 \cdot 40 = 2000
\]

**Total time per decision**:
\[
O(N \cdot b^d \cdot m \cdot T_{\text{eval}})
\]

where \( T_{\text{eval}} \) is time to evaluate state (card comparisons).

---

## 5. Belief State: Partial Observability

### 5.1 Latent Set

Player 1 observes only \( o = (H_1, |H_2|, |D_g|, P, P_t, G_o) \).

Hidden information is the **latent set**:

\[
L = \mathcal{D} \setminus (H_1 \cup P)
\]

where \( \mathcal{D} \) is the full 76-card Uno deck.

### 5.2 Belief Distribution

The **belief** over opponent hand \( H_2 \) is:

\[
b(H_2 | o, \text{history}) = P(H_2 | o, \text{history})
\]

**Prior** (uniform, if no additional information):

\[
b^-(H_2) = \frac{1}{\binom{|L|}{|H_2|}} \quad \text{for } H_2 \subseteq L
\]

**Posterior** (after opponent draws):

If opponent had no legal cards, the belief is constrained:

\[
b^+(H_2 | \text{no legal}) = \frac{1}{\binom{|L| - |N_L(P_t)|}{|H_2| + n}} \quad \text{for } H_2 \subseteq L \setminus N_L(P_t)
\]

where \( N_L(P_t) \) are legal cards in the latent set.

### 5.3 Sampling from Belief

Generate sample state \( s^{\text{sample}} \):

```
FUNCTION sampleState(belief):
    L ← belief.latentSet
    
    // Sample opponent hand
    H_2^sample ← RandomSubset(L, |H_2|)
    
    // Remaining cards are in deck
    remaining ← L \ H_2^sample
    D_g^sample ← RandomSubset(remaining, |D_g|)
    
    RETURN (H_1, H_2^sample, D_g^sample, P, P_t, G_o)
```

By law of large numbers:

\[
\mathbb{E}_{s \sim b}[f(s)] \approx \frac{1}{N} \sum_{i=1}^N f(s_i)
\]

with error \( O(1/\sqrt{N}) \).

---

## 6. Policy: Belief-Based Q-Function

### 6.1 Q-Function Definition

The **action value** function given belief \( b \):

\[
Q(a | b, d) = \mathbb{E}_{s \sim b} \left[ V(T(s, a), d, \text{opponent}) \right]
\]

This is the expected value of taking action \( a \) in a state sampled from the belief, then letting the opponent play optimally for \( d \) steps.

### 6.2 Monte Carlo Q-Estimation

Approximate via sampling \( N \) states:

\[
\hat{Q}(a) = \frac{1}{N} \sum_{i=1}^N V(T(s_i, a), d, \text{opponent})
\]

where \( s_i \sim b \) independently sampled.

**Algorithm**:

```
FUNCTION estimateQ(action, belief, lookahead):
    qSum ← 0
    FOR i = 1 TO numSamples:
        s_i ← belief.sampleState()
        s_next ← T(s_i, action)
        v_i ← expectiminimax(s_next, lookahead, opponent)
        qSum ← qSum + v_i
    RETURN qSum / numSamples
```

### 6.3 Optimal Policy

\[
\pi_{\text{AI}}^*(H_1, b) = \arg\max_{a \in A(H_1)} \hat{Q}(a)
\]

The AI chooses the action with the **highest estimated Q-value**.

---

## 7. Complete Algorithm: choose_action()

### 7.1 Pseudocode

```
FUNCTION chooseAction():
    // Validate belief
    IF belief is NULL:
        RAISE Error
    
    // Get legal actions for AI
    A ← GetLegalActions(H_1)
    IF |A| == 0:
        RETURN NULL
    IF |A| == 1:
        RETURN A[0]
    
    // Evaluate each action
    actionValues ← {}
    FOR EACH action a IN A:
        qSum ← 0
        
        // Monte Carlo estimation
        FOR i = 1 TO numSamples:
            // 1. Sample state from belief
            s_i ← belief.sampleState()
            
            // 2. Simulate AI's action
            s_next ← T(s_i, a)
            
            // 3. Evaluate from opponent's perspective
            v_i ← expectiminimax(s_next, lookahead, opponent)
            
            // 4. Accumulate
            qSum ← qSum + v_i
        
        // 5. Estimate Q-value
        qHat ← qSum / numSamples
        actionValues[a] ← qHat
    
    // 6. Select best action
    bestAction ← argmax_a actionValues[a]
    RETURN bestAction
```

### 7.2 Full Mathematical Formulation

\[
\pi_{\text{AI}}(H_1, b, d, N) = \arg\max_{a \in A(H_1)} \left[ \frac{1}{N} \sum_{i=1}^N V(T(s_i, a), d, \text{opp}) \right]
\]

Where:
- \( b \) = belief distribution over hidden states
- \( d \) = lookahead depth (typically 3)
- \( N \) = number of belief samples (typically 50)
- \( s_i \sim b \) = \( i \)-th sampled state
- \( T(s_i, a) \) = state after AI's action
- \( V(\cdot) \) = expectiminimax value from opponent's perspective

---

## 8. Integration: Full Pipeline

### 8.1 Initialization

```
FUNCTION initGame():
    game ← NewUnoGame()
    ai ← Uno_AI(
        player_id = 1,
        num_samples = 50,
        lookahead = 3
    )
    
    // Initialize belief from first observation
    observation ← game.getObservation()
    ai.belief ← Belief(observation)
```

### 8.2 Decision Loop

```
WHILE game is active:
    // AI's turn
    action ← ai.chooseAction()
    game.executeAction(action, AI)
    
    // Opponent's turn (heuristic)
    oppAction ← heuristic.chooseAction()
    game.executeAction(oppAction, opponent)
    
    // Update AI's belief
    observation ← game.getObservation()
    ai.belief.update(oppAction, observation)
```

### 8.3 Belief Update

```
FUNCTION updateBelief(oppAction, newObservation):
    IF oppAction is play:
        // Remove played card from latent set
        L ← L \ {cardPlayed}
        |H_2| ← |H_2| - 1
        P ← P ∪ {cardPlayed}
        P_t ← cardPlayed
        
        // Reset to prior with updated state
        belief ← UniformOverSubsets(L, |H_2|)
    
    ELSE:  // oppAction is draw
        // Opponent had no legal cards
        N_L ← {c ∈ L : c is legal on P_t}
        
        // Posterior excludes legal cards
        belief ← UniformOverSubsets(L \ N_L, |H_2| + n)
```

---

## 9. Convergence and Optimality Guarantees

### 9.1 Belief Accuracy

By law of large numbers, with \( N \) samples:

\[
\left| \hat{Q}(a) - \mathbb{E}_{s \sim b}[V(T(s,a), d, \text{opp})] \right| = O(1/\sqrt{N})
\]

With \( N = 50 \): error ≈ \( 14\% \) of value range.

### 9.2 Depth-Limited Optimality

At lookahead depth \( d \), the policy is optimal:

\[
V^{\pi^*}(b, d) = \max_\pi V^\pi(b, d)
\]

Beyond depth \( d \), error grows exponentially with planning horizon.

### 9.3 Total Approximation Error

Combined error from sampling, depth, and draw card approximation:

\[
\epsilon_{\text{total}} = \epsilon_{\text{belief}} + \epsilon_{\text{depth}} + \epsilon_{\text{draw}}
\]

\[
\epsilon_{\text{total}} \leq \frac{1}{\sqrt{N}} + \gamma^d + \frac{1}{\sqrt{m}}
\]

For typical parameters: \( \epsilon_{\text{total}} \approx 0.14 + 0.01 + 0.45 \approx 0.6 \)

---

## 10. Summary Table: Algorithm Components

| Component | Mathematical Form | Complexity | Role |
|-----------|-------------------|------------|------|
| **Value function** | \( V(s,d,p) \) | \( O(b^d m) \) per node | Core recursion |
| **Minimax** | max/min over actions | \( O(b) \) per level | Adversarial play |
| **Expectation** | \( \frac{1}{m}\sum \) over draws | \( O(m) \) | Stochastic cards |
| **Belief sampling** | \( s_i \sim b \) | \( O(N) \) samples | Partial observability |
| **Q-function** | \( \frac{1}{N}\sum V \) | \( O(N \cdot b^d m) \) | Action evaluation |
| **Policy** | \( \arg\max \hat{Q} \) | \( O(1) \) | Final decision |

---

## 11. Key Insights

1. **Minimax handles adversarial play**: Max on AI's turn, min on opponent's turn
2. **Expectation handles uncertainty**: Monte Carlo samples approximate distributions
3. **Belief sampling handles partial observability**: Models uncertainty about opponent's hand
4. **Depth-limited search**: Tractable computation by truncating at \( d=3 \)
5. **Greedy + lookahead**: Combines myopic Q-value with 3-ply foresight

This algorithm achieves **approximately optimal play** by efficiently combining game tree search with belief-based reasoning in a partially observable two-player game.
