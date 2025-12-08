import random
from stage_1_mini_uno.mini_uno import MiniUno
from stage_1_mini_uno.offline_solver import OfflineSolver
from stage_1_mini_uno.online_solver_adapter import MiniUnoAI
from uno import Uno
from simulate_games import Uno_AI
from pomdp import Action

class GameManager:
    def __init__(self):
        self.game = None
        self.opponent_type = None
        self.game_mode = 'mini'
        self.solver = None
        self.offline_solver_instance = OfflineSolver() # Reuse memoization
        self.turn = 1 # 1 for Player, 2 for AI
        
    def start_new_game(self, opponent_type, seed, game_mode='mini'):
        self.opponent_type = opponent_type
        self.game_mode = game_mode
        
        if game_mode == 'full':
            self.game = Uno()
            self.game.new_game(seed=seed)
            # Full Uno only supports Online AI
            # We use fewer samples/lookahead for speed in web UI if needed
            self.solver = Uno_AI(player_id=2, num_samples=20, lookahead=2)
        else:
            self.game = MiniUno()
            self.game.new_game(seed=seed)
            
            # Initialize Solver
            if opponent_type == 'offline':
                self.solver = self.offline_solver_instance
            else:
                # Online solver (Player 2)
                self.solver = MiniUnoAI(player_id=2, num_samples=100, lookahead=2)
        
        self.turn = 1 # Player always starts? Or random?
        # Let's say Player 1 starts for now, or we can check game logic.
        # MiniUno/Uno doesn't enforce turn order in state, it's external.
            
    def get_game_state(self):
        if not self.game:
            return {"status": "Not Started"}
            
        # Serialize state for frontend
        return {
            "status": self.game.G_o,
            "turn": self.turn, # 1 or 2
            "player_hand": [f"{c[0]} {c[1]}" for c in self.game.H_1],
            "opponent_hand_count": len(self.game.H_2),
            "top_card": f"{self.game.P_t[0]} {self.game.P_t[1]}" if self.game.P_t else None,
            "deck_count": len(self.game.D_g),
            "winner": self.get_winner()
        }
        
    def get_winner(self):
        if len(self.game.H_1) == 0: return 1
        if len(self.game.H_2) == 0: return 2
        return None
        
    def player_play_card(self, card_index):
        if self.turn != 1:
            return False, "Not your turn"
            
        if card_index < 0 or card_index >= len(self.game.H_1):
            return False, "Invalid card index"
            
        card = self.game.H_1[card_index]
        action = Action(X_1=card)
        
        # Check legality
        legal_actions = self.game.get_legal_actions(player=1)
        is_legal = False
        for la in legal_actions:
            if la.is_play() and la.X_1 == card:
                is_legal = True
                break
                
        if not is_legal:
            return False, "Illegal move"
            
        self.game.execute_action(action, player=1)
        
        # Check win
        if self.get_winner():
            return True, "You Win!"
            
        self.turn = 2 # Switch to AI
        return True, "Played " + str(card)
        
    def player_draw_card(self):
        if self.turn != 1:
            return False, "Not your turn"
            
        # Find draw action
        legal_actions = self.game.get_legal_actions(player=1)
        draw_action = None
        for la in legal_actions:
            if la.is_draw():
                draw_action = la
                break
                
        if not draw_action:
            return False, "Cannot draw (maybe you have a playable card?)"
            
        self.game.execute_action(draw_action, player=1)
        self.turn = 2 # Switch to AI
        return True, "Drew a card"
        
    def ai_make_move(self):
        if self.turn != 2:
            return {"success": False, "message": "Not AI turn"}
            
        if self.game.G_o != "Active":
             return {"success": False, "message": "Game Over"}
             
        action = None
        
        if self.game_mode == 'mini' and self.opponent_type == 'offline':
            # Offline Solver (Perfect Info)
            # Turn=1 for P2 in solver state (0=P1, 1=P2)
            state = self.solver.get_canonical_state(self.game, turn=1)
            self.solver.solve(state)
            action = self.solver.policy.get(state)
            
        else:
            # Online Solver (Belief) - Works for both Mini and Full
            self.solver.init_belief(self.game)
            action = self.solver.choose_action()
            
        if action:
            self.game.execute_action(action, player=2)
            self.turn = 1 # Switch to Player
            return {"success": True, "action": str(action)}
        else:
            # Fallback (should not happen)
            return {"success": False, "message": "AI failed to choose action"}
