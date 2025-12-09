from web_ui.game_manager import GameManager
from cards import Card

def test_gm_illegal_move():
    print("Testing GameManager Illegal Move...")
    gm = GameManager()
    gm.start_new_game(opponent_type='online', seed=42, game_mode='full')
    
    # Force state
    # P_t = Green 5
    green_five = ('G', 5)
    gm.game.P = [green_five]
    gm.game.P_t = green_five
    
    # H_1 = [Green 0]
    green_zero = ('G', 0)
    gm.game.H_1 = [green_zero]
    
    gm.game.create_S()
    
    print(f"Top Card: {gm.game.P_t}")
    print(f"Hand: {gm.game.H_1}")
    
    # Try to play card at index 0
    success, msg = gm.player_play_card(0)
    print(f"Result: {success}, {msg}")

if __name__ == "__main__":
    test_gm_illegal_move()
