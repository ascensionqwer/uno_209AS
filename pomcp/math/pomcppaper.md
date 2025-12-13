3 Monte-Carlo Planning in POMDPs
Partially Observable Monte-Carlo Planning (POMCP) consists of a UCT search that selects
actions at each time-step; and a particle filter that updates the agent’s belief state.
3.1 Partially Observable UCT (PO–UCT)
We extend the UCT algorithm to partially observable environments by using a search tree
of histories instead of states. The tree contains a node T(h) = hN(h), V (h)i for each
represented history h. N(h) counts the number of times that history h has been visited.
V (h) is the value of history h, estimated by the mean return of all simulations starting with h.
New nodes are initialised to hVinit(h), Ninit(h)i if domain knowledge is available, and to h0, 0i
otherwise. We assume for now that the belief state B(s, h) is known exactly. Each simulation
starts in an initial state that is sampled from B(·, ht). As in the fully observable algorithm,
the simulations are divided into two stages. In the first stage of simulation, when child
nodes exist for all children, actions are selected by UCB1, V
⊕(ha) = V (ha) + c
qlog N(h)
N(ha)
.
Actions are then selected to maximise this augmented value, argmaxaV
⊕(ha). In the second
3
N=1
V=2
a1 a2
o1 o2 o1 o2
o1 o2
a1 a2 a1 a2
N=1
V=-1
N=1
V=-3
N=2
V=-2
N=3
V=-1
N=1
V=4
N=1
V=6
N=2
V=5
N=1
V=-1
N=3
V=3
N=3
V=1
N=6
V=2
N=9
V=1.5
a=a2
o=o2
o1 o2
a1 a2
N=1
V=4
N=1
V=6
N=2
V=5
N=1
V=-1
N=3
V=3
S={17,34,26,31}
S={42} S={27,36,44}
o1 o2
a1 a2
N=1
V=4
N=1
V=6
N=2
V=5
N=1
V=-1
N=3
V=3 S={27,36,44}
S={38} S={7} S={38} S={7}
S={27,36,44}
h h
hao
S={38} S={7}
r=+2 r=-1 r=+3 r=+4 r=+6 r=-1
Figure 1: An illustration of POMCP in an environment with 2 actions, 2 observations, 50 states,
and no intermediate rewards. The agent constructs a search tree from multiple simulations, and
evaluates each history by its mean return (left). The agent uses the search tree to select a real
action a, and observes a real observation o (middle). The agent then prunes the tree and begins a
new search from the updated history hao (right).
stage of simulation, actions are selected by a history based rollout policy πrollout(h, a) (e.g.
uniform random action selection). After each simulation, precisely one new node is added
to the tree, corresponding to the first new history encountered during that simulation.
3.2 Monte-Carlo Belief State Updates
In small state spaces, the belief state can be updated exactly by Bayes’ theorem, B(s
0
P
, hao) =
s∈S Z
a
s0oP
a
P ss0B(s,h)
s∈S
P
s00∈S Za
s00o
Pa
ss00B(s,h)
. The majority of POMDP planning methods operate in this manner [13]. However, in large state spaces even a single Bayes update may be computationally
infeasible. Furthermore, a compact represention of the transition or observation probabilities may not be available. To plan efficiently in large POMDPs, we approximate the
belief state using an unweighted particle filter, and use a Monte-Carlo procedure to update
particles based on sample observations, rewards, and state transitions. Although weighted
particle filters are used widely to represent belief states, an unweighted particle filter can
be implemented particularly efficiently with a black box simulator, without requiring an
explicit model of the POMDP, and providing excellent scalability to larger problems.
We approximate the belief state for history ht by K particles, Bi
t ∈ S, 1 ≤ i ≤ K. Each
particle corresponds to a sample state, and the belief state is the sum of all particles,
Bˆ(s, ht) = 1
K
PK
i=1 δsBi
t
, where δss0 is the kronecker delta function. At the start of the
algorithm, K particles are sampled from the initial state distribution, Bi
0 ∼ I, 1 ≤ i ≤ K.
After a real action at is executed, and a real observation ot is observed, the particles are
updated by Monte-Carlo simulation. A state s is sampled from the current belief state
Bˆ(s, ht), by selecting a particle at random from Bt. This particle is passed into the black
box simulator, to give a successor state s
0 and observation o
0
, (s
0
, o0
, r) ∼ G(s, at). If the
sample observation matches the real observation, o = ot, then a new particle s
0
is added
to Bt+1. This process repeats until K particles have been added. This approximation to
the belief state approaches the true belief state with sufficient particles, limK→∞ Bˆ(s, ht) =
B(s, ht), ∀s ∈ S. As with many particle filter approaches, particle deprivation is possible
for large t. In practice we combine the belief state update with particle reinvigoration. For
example, new particles can be introduced by adding artificial noise to existing particles.
3.3 Partially Observable Monte-Carlo
POMCP combines Monte-Carlo belief state updates with PO–UCT, and shares the same
simulations for both Monte-Carlo procedures. Each node in the search tree, T(h) =
hN(h), V (h), B(h)i, contains a set of particles B(h) in addition to its count N(h) and value
V (h). The search procedure is called from the current history ht. Each simulation begins
from a start state that is sampled from the belief state B(ht). Simulations are performed
4
Algorithm 1 Partially Observable Monte-Carlo Planning
procedure Search(h)
repeat
if h = empty then
s ∼ I
else
s ∼ B(h)
end if
Simulate(s, h, 0)
until Timeout()
return argmax
b
V (hb)
end procedure
procedure Rollout(s, h, depth)
if γ
depth <  then
return 0
end if
a ∼ πrollout(h, ·)
(s
0
, o, r) ∼ G(s, a)
return r + γ.Rollout(s
0
, hao, depth+1)
end procedure
procedure Simulate(s, h, depth)
if γ
depth <  then
return 0
end if
if h /∈ T then
for all a ∈ A do
T(ha) ← (Ninit(ha), Vinit(ha), ∅)
end for
return Rollout(s, h, depth)
end if
a ← argmax
b
V (hb) + c
qlog N(h)
N(hb)
(s
0
, o, r) ∼ G(s, a)
R ← r + γ.Simulate(s
0
, hao, depth + 1)
B(h) ← B(h) ∪ {s}
N(h) ← N(h) + 1
N(ha) ← N(ha) + 1
V (ha) ← V (ha) + R−V (ha)
N(ha)
return R
end procedure
using the partially observable UCT algorithm, as described above. For every history h
encountered during simulation, the belief state B(h) is updated to include the simulation
state. When search is complete, the agent selects the action at with greatest value, and
receives a real observation ot from the world. At this point, the node T(htatot) becomes the
root of the new search tree, and the belief state B(htao) determines the agent’s new belief
state. The remainder of the tree is pruned, as all other histories are now impossible. The
complete POMCP algorithm is described in Algorithm 1 and Figure 1.
4 Convergence
The UCT algorithm converges to the optimal value function in fully observable MDPs [8].
This suggests two simple ways to apply UCT to POMDPs: either by converting every belief
state into an MDP state, or by converting every history into an MDP state, and then
applying UCT directly to the derived MDP. However, the first approach is computationally
expensive in large POMDPs, where even a single belief state update can be prohibitively
costly. The second approach requires a history-based simulator that can sample the next
history given the current history, which is usually more costly and hard to encode than a
state-based simulator. The key innovation of the PO–UCT algorithm is to apply a UCT
search to a history-based MDP, but using a state-based simulator to efficiently sample states
from the current beliefs. In this section we prove that given the true belief state B(s, h),
PO–UCT also converges to the optimal value function. We prove convergence for POMDPs
with finite horizon T; this can be extended to the infinite horizon case as suggested in [8].
Lemma 1. Given a POMDP M = (S, A,P, R, Z), consider the derived MDP with histories as states, M˜ = (H, A,P˜, R˜), where P˜a
h,hao =
P
s∈S
P
s
0∈S
B(s, h)P
a
ss0Z
a
s
0o
and R˜a
h =
P
s∈S
B(s, h)Ra
s
. Then the value function V˜ π
(h) of the derived MDP is equal to the value
function V
π
(h) of the POMDP, ∀π V˜ π
(h) = V
π
(h).
Proof. By backward induction on the Bellman equation, starting from
the horizon, V
π
(h) = P
s∈S
P
a∈A
P
s0∈S
P
o∈O
B(s, h)π(h, a) (Ra
s + γP
a
ss0Z
a
s0oV
π
(hao)) =
P
a∈A
P
o∈O
π(h, a)

R˜a
h + γP˜a
h,haoV˜ π
(hao)

= V˜ π
(h).
Let Dπ
(hT ) be the POMDP rollout distribution. This is the distribution of histories generated by sampling an initial state st ∼ B(s, ht), and then repeatedly sampling actions from
policy π(h, a) and sampling states, observations and rewards from M, until terminating at
5
time T. Let D˜π
(hT ) be the derived MDP rollout distribution. This is the distribution of
histories generated by starting at ht and then repeatedly sampling actions from policy π
and sampling state transitions and rewards from M˜ , until terminating at time T.
Lemma 2. For any rollout policy π, the POMDP rollout distribution is equal to the derived
MDP rollout distribution, ∀π Dπ
(hT ) = D˜π
(hT ).
Proof. By forward induction from ht, Dπ
(hao) = Dπ
(h)π(h, a)
P
s∈S
P
s
0∈S B(s, h)P
a
ss0Z
a
s
0o =
D˜π
(h)π(h, a)P˜a
h,hao = D˜π
(hao).
Theorem 1. For suitable choice of c, the value function constructed by PO–UCT converges
in probability to the optimal value function, V (h)
p→ V
∗
(h), for all histories h that are
prefixed by ht. As the number of visits N(h) approaches infinity, the bias of the value
function, E [V (h) − V
∗
(h)] is O(log N(h)/N(h)).
Proof. By Lemma 2 the PO–UCT simulations can be mapped into UCT simulations in the
derived MDP. By Lemma 1, the analysis of UCT in [8] can then be applied to PO–UCT