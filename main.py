import time
import pygame
import sys
from uno_ai import Uno_AI
from uno_ui import Uno_UI
from uno_naive import Uno_Naive
from uno_human_player import UnoHumanPlayer
from game_controller import GameController

def main():
    # 1. Initialize UI
    ui = Uno_UI()
    
    # 2. Initialize Players
    player1 = UnoHumanPlayer(player_id=1, ui=ui)
    player2 = Uno_AI(player_id=2)
   
    # 3. Initialize Controller
    controller = GameController(player1, player2, verbose=True)
    
    # 4. Start Game
    controller.init_game(seed=int(time.time()))
    
    game_running = True
    
    while game_running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        # Check Win Condition
        if controller.game.G_o == "GameOver":
            winner = "Player 1" if len(controller.game.H_1) == 0 else "Player 2"
            ui.draw_game_state(controller.game, f"GAME OVER! {winner} WINS!")
            pygame.time.wait(4000)
            game_running = False
            continue

        # --- Turn Logic ---
        if controller.current_player == 1:
            # Human Turn (Calls get_input_action inside)
            controller.play_turn()
        else:
            # AI Turn
            ui.draw_game_state(controller.game, "Opponent is thinking...")
            time.sleep(1.0) # Small delay
            controller.play_turn()

    pygame.quit()

if __name__ == "__main__":
    main()
