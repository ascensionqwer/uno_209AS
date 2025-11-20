import unittest
from cards import RED, YELLOW, GREEN, BLUE
from pomdp import Action
from policy_generator import PolicyGenerator, MDPState
from belief import Belief

class TestPolicyGenerator(unittest.TestCase):
    def test_simple_endgame(self):
        pg = PolicyGenerator()
        
        start_state = pg._to_canonical(
            h1=[(RED, 1)],
            h2=[(BLUE, 1)],
            p_t=(RED, 2),
            g_o="Active",
            turn=1
        )
        
        pg.build_transition_table(start_state, max_depth=5)
        pg.value_iteration()
        
        optimal_action = pg.get_optimal_action(start_state)
        self.assertIn("('R', 1)", str(optimal_action))
        
        # Check Q-table
        start_idx = pg.state_to_idx[start_state]
        self.assertTrue(len(pg.Q_table[start_idx]) > 0)

    def test_draw_scenario(self):
        pg = PolicyGenerator()
        
        start_state = pg._to_canonical(
            h1=[(BLUE, 1)],
            h2=[(RED, 1)],
            p_t=(RED, 2),
            g_o="Active",
            turn=1
        )
        
        pg.build_transition_table(start_state, max_depth=5)
        pg.value_iteration()
        
        optimal_action = pg.get_optimal_action(start_state)
        self.assertIn("Y_1", str(optimal_action))

    def test_qmdp_online(self):
        # Test the online phase with a mock belief
        pg = PolicyGenerator()
        
        # Scenario:
        # H_1 = [RED 1]
        # P_t = RED 2
        # Unknown H_2: Could be [BLUE 1] (Win for us) or [RED 3] (Win for us)
        # Actually, H_2 doesn't affect our immediate ability to win if we have a match.
        # Let's construct a scenario where H_2 matters?
        # In Q-MDP, H_2 affects the *future* value.
        # If we play RED 1, we win immediately (Reward 1).
        # So H_2 shouldn't matter.
        
        # Let's try a scenario where we don't win immediately.
        # H_1 = [RED 1, BLUE 1]
        # P_t = RED 2
        # Action: Play RED 1.
        # Next State: H_1=[BLUE 1], P_t=RED 1. Opponent's turn.
        # If Opponent has RED 3, they play and maybe win?
        
        # To properly test Q-MDP, we need to:
        # 1. Build a table that covers the possible states in the belief.
        # 2. Create a belief that samples those states.
        # 3. Call get_best_action.
        
        # Let's define the "Universe" of cards for this small test
        # Cards: R1, R2, B1, B2
        # H_1 = [R1]
        # P_t = R2
        # H_2 could be B1 or B2.
        
        # State 1: H_2 = [B1]. (R1, B1, R2, Active, 1)
        # State 2: H_2 = [B2]. (R1, B2, R2, Active, 1)
        
        s1 = pg._to_canonical([(RED, 1)], [(BLUE, 1)], (RED, 2), "Active", 1)
        s2 = pg._to_canonical([(RED, 1)], [(BLUE, 2)], (RED, 2), "Active", 1)
        
        # Build table for both
        pg.build_transition_table(s1, max_depth=5)
        pg.build_transition_table(s2, max_depth=5) # Add s2 and its children
        
        pg.value_iteration()
        
        # Mock Belief
        # We need a belief object that returns samples s1 and s2.
        # Since creating a real Belief object requires complex setup, we'll mock it.
        class MockBelief:
            def sample_states(self, n_samples):
                # Return POMDP-like tuples corresponding to s1 and s2
                # POMDP: (H_1, H_2, D_g, P, P_t, G_o)
                # s1: H_2=[B1]
                p1 = ([(RED, 1)], [(BLUE, 1)], [], [], (RED, 2), "Active")
                # s2: H_2=[B2]
                p2 = ([(RED, 1)], [(BLUE, 2)], [], [], (RED, 2), "Active")
                return [p1, p2] * (n_samples // 2)

        mock_belief = MockBelief()
        action = pg.get_best_action(mock_belief, num_particles=10)
        
        print(f"Q-MDP Action: {action}")
        # Should be Play R1 (Action(X_1=('R', 1)))
        self.assertIn("('R', 1)", str(action))

if __name__ == '__main__':
    unittest.main()
