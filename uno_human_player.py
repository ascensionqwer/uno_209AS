import pygame
import sys
from uno import Uno, Action
from typing import Optional
from uno_ui import Uno_UI, FPS

class UnoHumanPlayer:
    def __init__(self, player_id: int, ui: Uno_UI):
        self.player_id = player_id
        self.ui = ui
        self.game = None  # Reference to the game object

    def set_game(self, game: Uno):
        self.game = game

    def choose_action(self) -> Action:
        """
        Delegates the input loop entirely to the UI.
        This fixes the AttributeError by calling the correct method on the UI.
        """
        if not self.game:
            raise ValueError("Game not set for Human Player")
            
        # This calls the method that blocks and waits for a click
        return self.ui.get_input_action(self.game, self.player_id)