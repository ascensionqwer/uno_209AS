from uno import Uno
from cards import Card

def test_green_zero_on_green():
    print("Testing Green 0 on Green Card...")
    game = Uno()
    game.new_game()
    
    # Force state
    # P_t = Green 5
    green_five = ('G', 5)
    game.P = [green_five]
    game.P_t = green_five
    
    # H_1 = [Green 0]
    green_zero = ('G', 0)
    game.H_1 = [green_zero]
    
    game.create_S()
    
    print(f"Top Card: {game.P_t}")
    print(f"Hand: {game.H_1}")
    
    # Check is_legal_play directly
    is_legal = game.is_legal_play(green_zero)
    print(f"is_legal_play({green_zero}) on {green_five}: {is_legal}")
    
    # Check get_legal_actions
    actions = game.get_legal_actions(player=1)
    print(f"Legal Actions: {actions}")
    
    found = any(a.is_play() and a.X_1 == green_zero for a in actions)
    print(f"Found Green 0 in legal actions: {found}")

if __name__ == "__main__":
    test_green_zero_on_green()
