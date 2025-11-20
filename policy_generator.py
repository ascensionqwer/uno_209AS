import itertools
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
import random
from cards import Card, RED, YELLOW, GREEN, BLUE
from pomdp import Action, State
from uno import Uno

# MDP State: (H_1, H_2, P_t, G_o)
# We simplify by ignoring the exact draw pile content (D_g) and played pile history (P)
# for the transition table, assuming infinite/random deck for probabilities,
# or we can include deck size if critical.
# For a tractable table, we'll stick to:
# H_1: Tuple[Card] (sorted)
# H_2: Tuple[Card] (sorted) - Perfect Information assumption
# P_t: Card (Top card)
# G_o: str ("Active", "GameOver")
# Turn: int (1 for Player 1, 2 for Player 2)
MDPState = Tuple[Tuple[Card, ...], Tuple[Card, ...], Optional[Card], str, int]

class PolicyGenerator:
    def __init__(self):
        self.states: List[MDPState] = []
        self.state_to_idx: Dict[MDPState, int] = {}
        self.transitions: Dict[int, Dict[str, List[Tuple[int, float, float]]]] = {} # state_idx -> action_str -> [(next_state_idx, prob, reward)]
        self.policy: Dict[int, str] = {} # state_idx -> best_action_str
        self.V: Dict[int, float] = {} # state_idx -> value

    def _to_canonical(self, h1: List[Card], h2: List[Card], p_t: Optional[Card], g_o: str, turn: int) -> MDPState:
        """Converts game state components to a canonical immutable MDPState."""
        return (
            tuple(sorted(h1)),
            tuple(sorted(h2)),
            p_t,
            g_o,
            turn
        )

    def generate_states(self, all_cards: List[Card], max_hand_size: int = 2):
        """
        Generates all possible states for a simplified game with a subset of cards.
        This is computationally expensive, so we limit hand size.
        """
        # 1. Generate all possible hands
        # For very small games, we can iterate combinations.
        # For larger games, we just discover states via BFS from a start state.
        pass

    def get_possible_actions(self, state: MDPState) -> List[Action]:
        """Returns legal actions for the current player in the given state."""
        h1, h2, p_t, g_o, turn = state
        
        if g_o == "GameOver":
            return []

        hand = h1 if turn == 1 else h2
        legal_actions = []

        # Check plays
        if p_t is not None:
            for card in hand:
                if card[0] == p_t[0] or card[1] == p_t[1]: # Color or Value match
                    legal_actions.append(Action(X_1=card))
        
        # If no legal plays (or always allowed to draw?), standard UNO rules say you MUST play if you can?
        # Actually standard UNO allows drawing even if you can play, but usually people play.
        # The provided `uno.py` `get_legal_actions` implies:
        # "If no legal plays, must draw 1 card". It doesn't seem to allow drawing if you have a play.
        # Let's stick to `uno.py` logic:
        
        if not legal_actions:
             legal_actions.append(Action(n=1))
             
        return legal_actions

    def get_transition_probs(self, state: MDPState, action: Action) -> List[Tuple[MDPState, float, float]]:
        """
        Returns list of (next_state, probability, reward).
        Reward: +1 for winning, -1 for losing, 0 otherwise.
        """
        h1, h2, p_t, g_o, turn = state
        
        if g_o == "GameOver":
            return [(state, 1.0, 0.0)]

        next_turn = 2 if turn == 1 else 1
        current_hand = list(h1) if turn == 1 else list(h2)
        other_hand = list(h2) if turn == 1 else list(h1)
        
        if action.is_play():
            # Deterministic transition (mostly)
            card = action.X_1
            new_hand = current_hand.copy()
            if card in new_hand:
                new_hand.remove(card)
            else:
                # Should not happen if action is legal
                return []

            new_p_t = card
            
            # Check win condition
            if not new_hand:
                new_g_o = "GameOver"
                reward = 1.0 if turn == 1 else -1.0 # Reward is from Player 1's perspective
                # Game ends, turn doesn't really matter, but let's keep it consistent
                return [(self._to_canonical(
                    new_hand if turn == 1 else other_hand,
                    other_hand if turn == 1 else new_hand,
                    new_p_t, new_g_o, next_turn
                ), 1.0, reward)]
            
            return [(self._to_canonical(
                new_hand if turn == 1 else other_hand,
                other_hand if turn == 1 else new_hand,
                new_p_t, g_o, next_turn
            ), 1.0, 0.0)]

        elif action.is_draw():
            # Stochastic transition!
            # We don't know what card will be drawn.
            # We need to estimate probabilities based on "unknown" cards.
            # For the MDP generation, we assume a distribution over the remaining deck.
            
            # Simplified approach:
            # Assume infinite deck or uniform distribution over all cards NOT in hands or P_t.
            # This is an approximation.
            
            # Construct the set of "possible draw cards"
            # In a real game, this is D_g. In this abstract MDP, we might not track D_g explicitly.
            # Let's assume a standard deck distribution minus visible cards.
            
            # For the purpose of this "Table Generation", we might need to pass in the full deck info
            # or just assume a uniform probability over all defined card types.
            
            all_possible_cards = []
            colors = [RED, YELLOW, GREEN, BLUE]
            for c in colors:
                all_possible_cards.append((c, 0))
                for n in range(1, 10):
                    all_possible_cards.append((c, n))
                    all_possible_cards.append((c, n))
            
            # Filter out cards we know are in hands or P_t
            # (This is imperfect because we don't track the full discard pile P in MDPState)
            # But for the transition probabilities, we can just iterate over ALL unique card types
            # and assign probability based on their frequency in a full deck.
            
            # Optimization: Group by (Color, Value) since identical cards behave identically.
            
            possible_draws = []
            # Simple version: Uniform probability over all card types (ignoring counts for now to keep it simple)
            # Or better: Use the full deck counts.
            
            # Let's just iterate over all unique cards in the game definition.
            unique_cards = sorted(list(set(all_possible_cards)))
            prob = 1.0 / len(unique_cards) # Very rough approximation
            
            transitions = []
            for card in unique_cards:
                new_hand = current_hand + [card]
                # Turn passes to next player after draw? 
                # Standard UNO: If you draw, you can play it if playable, otherwise turn passes?
                # `uno.py` `execute_action` just adds to hand. It doesn't say anything about playing immediately.
                # But `full_game.py` (not read yet) likely handles the loop.
                # Usually in simplified UNO (and `uno.py` logic), drawing is an action that ends the turn (or updates state).
                # Wait, `uno.py` `get_legal_actions` returns `Action(n=1)`.
                # `execute_action` adds cards.
                # It doesn't automatically play.
                # So the state transitions to (Hand+1, NextTurn).
                
                transitions.append((
                    self._to_canonical(
                        new_hand if turn == 1 else other_hand,
                        other_hand if turn == 1 else new_hand,
                        p_t, g_o, next_turn
                    ),
                    prob,
                    0.0
                ))
                
            return transitions

        return []

    def build_transition_table(self, start_state: MDPState, max_depth: int = 10):
        """
        Builds the state space and transition table via BFS/DFS from a start state.
        """
        queue = [start_state]
        visited = set()
        visited.add(start_state)
        
        self.states = []
        self.state_to_idx = {}
        self.transitions = {}
        
        idx_counter = 0
        
        while queue:
            s = queue.pop(0)
            if s not in self.state_to_idx:
                self.state_to_idx[s] = idx_counter
                self.states.append(s)
                idx_counter += 1
            
            s_idx = self.state_to_idx[s]
            self.transitions[s_idx] = {}
            
            # Stop if terminal
            if s[3] == "GameOver":
                continue
                
            # Limit depth/size? For now, let's just run until exhaustion or limit
            if len(self.states) > 1000000: # Safety break - 1M states for better coverage
                print(f"State space limit reached: {len(self.states)} states")
                break
            
            actions = self.get_possible_actions(s)
            for action in actions:
                action_str = str(action)
                trans_list = self.get_transition_probs(s, action)
                
                self.transitions[s_idx][action_str] = []
                
                for next_s, prob, reward in trans_list:
                    if next_s not in visited:
                        visited.add(next_s)
                        queue.append(next_s)
                    
                    # We need the index of next_s. 
                    # If it's not assigned yet, we might need to assign it later or do a two-pass.
                    # Actually, we can just store the state object temporarily or assign index on the fly.
                    # Let's assign index on the fly if not exists (but add to queue)
                    if next_s not in self.state_to_idx:
                        self.state_to_idx[next_s] = idx_counter
                        self.states.append(next_s)
                        idx_counter += 1
                        
                    next_s_idx = self.state_to_idx[next_s]
                    self.transitions[s_idx][action_str].append((next_s_idx, prob, reward))

    def value_iteration(self, gamma: float = 0.95, theta: float = 1e-4):
        """
        Performs Value Iteration to compute the optimal policy.
        Stores Q(s, a) for Q-MDP.
        """
        # Initialize V
        for idx in range(len(self.states)):
            self.V[idx] = 0.0
            
        # Initialize Q-table: state_idx -> action_str -> value
        self.Q_table: Dict[int, Dict[str, float]] = defaultdict(dict)

        while True:
            delta = 0
            for s_idx in range(len(self.states)):
                v = self.V[s_idx]
                
                if self.states[s_idx][3] == "GameOver":
                    continue
                
                if s_idx not in self.transitions or not self.transitions[s_idx]:
                    continue
                    
                turn = self.states[s_idx][4]
                is_max_node = (turn == 1)
                
                if is_max_node:
                    max_val = -float('inf')
                else:
                    max_val = float('inf')
                
                best_action = None
                
                for action_str, outcomes in self.transitions[s_idx].items():
                    # Expected value for this action
                    val = 0
                    for next_s_idx, prob, reward in outcomes:
                        val += prob * (reward + gamma * self.V[next_s_idx])
                    
                    # Store Q-value
                    self.Q_table[s_idx][action_str] = val
                    
                    if is_max_node:
                        if val > max_val:
                            max_val = val
                            best_action = action_str
                    else:
                        if val < max_val:
                            max_val = val
                            best_action = action_str
                
                self.V[s_idx] = max_val
                self.policy[s_idx] = best_action
                
                delta = max(delta, abs(v - self.V[s_idx]))
            
            if delta < theta:
                break
        
        print(f"Value Iteration converged. Processed {len(self.states)} states.")

    def get_optimal_action(self, state: MDPState) -> Optional[str]:
        if state in self.state_to_idx:
            idx = self.state_to_idx[state]
            return self.policy.get(idx)
        return None

    def get_best_action(self, belief, num_particles: int = 100) -> Optional[Action]:
        """
        Q-MDP Online Phase:
        1. Sample states from belief.
        2. For each action, compute average Q-value across samples.
        3. Return action with max average Q-value.
        """
        samples = belief.sample_states(n_samples=num_particles)
        
        # Aggregate Q-values
        action_values: Dict[str, float] = defaultdict(float)
        action_counts: Dict[str, int] = defaultdict(int)
        
        valid_samples = 0
        
        for sample_state in samples:
            # Convert POMDP state to MDPState canonical form
            # POMDP State: (H_1, H_2, D_g, P, P_t, G_o)
            # MDPState: (H_1, H_2, P_t, G_o, Turn)
            # We assume it's Player 1's turn if we are asking for an action
            
            h1 = tuple(sorted(sample_state[0]))
            h2 = tuple(sorted(sample_state[1]))
            p_t = sample_state[4]
            g_o = sample_state[5]
            turn = 1 
            
            canonical_s = self._to_canonical(h1, h2, p_t, g_o, turn)
            
            if canonical_s in self.state_to_idx:
                s_idx = self.state_to_idx[canonical_s]
                valid_samples += 1
                
                # Sum Q-values for all valid actions in this state
                if s_idx in self.Q_table:
                    for action_str, q_val in self.Q_table[s_idx].items():
                        action_values[action_str] += q_val
                        action_counts[action_str] += 1
            else:
                # If state not in table, we can't use it.
                # This happens if table was built with limited depth/scope.
                pass
                
        if not action_values:
            print("Warning: No valid states found in Q-table for belief samples.")
            return None
            
        # Find best action
        # Note: We should probably normalize by count if actions aren't available in all states,
        # but Q-MDP usually assumes actions are available or penalizes illegal ones.
        # Here, we just take the max sum (equivalent to max average if action is available in all samples).
        # If an action is only legal in some samples, its sum will be lower, which is correct (risk of being illegal).
        
        best_action_str = max(action_values.items(), key=lambda x: x[1])[0]
        
        # Convert string back to Action object (hacky parsing or we need a mapping)
        # Since we stored str(Action), we need to reconstruct.
        # Better way: Store Action objects in Q-table keys? No, not hashable easily.
        # Let's just parse or use a lookup if we had one.
        # For now, let's try to find a sample that had this action and return that object?
        # Or just parse the string.
        
        # Simple parsing for now:
        if "X_1" in best_action_str:
            # Action(X_1=('Color', Value))
            # Extract content between Action(X_1= and )
            import ast
            # This is brittle but works for the string repr we have
            # "Action(X_1=('R', 1))" -> "('R', 1)"
            val_str = best_action_str.split("X_1=")[1][:-1]
            card_tuple = ast.literal_eval(val_str)
            return Action(X_1=card_tuple)
        elif "Y_" in best_action_str:
            # Action(Y_n=[...]) or Action(Y_n=[])
            # We just need n.
            if "Y_1" in best_action_str:
                return Action(n=1)
            elif "Y_2" in best_action_str:
                return Action(n=2)
            elif "Y_4" in best_action_str:
                return Action(n=4)
                
        return None
