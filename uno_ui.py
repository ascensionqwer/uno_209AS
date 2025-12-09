import pygame
import sys
from typing import Optional, List, Tuple

# Try to import Uno/Action. Handle Card type definition manually if import fails.
from uno import Uno, Action
try:
    from uno import Card
except ImportError:
    Card = Tuple[str, int]

# === Configuration ===
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 768
BG_COLOR = (20, 100, 20)  # Dark Green Felt color
TEXT_COLOR = (255, 255, 255)
CARD_WIDTH = 100
CARD_HEIGHT = 152
FPS = 30

COLOR_MAP = {'R': 'red', 'Y': 'yellow', 'G': 'green', 'B': 'blue'}

class Uno_UI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Uno - Human vs AI")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 24)
        self.large_font = pygame.font.SysFont("Arial", 48)
        
        self.card_images = {}
        self.back_image = None
        self.load_images()

    def load_images(self):
        try:
            for color_code, color_name in COLOR_MAP.items():
                for number in range(10):
                    filename = f"images/{color_name}_{number}.png"
                    try:
                        image = pygame.image.load(filename).convert_alpha()
                        image = pygame.transform.scale(image, (CARD_WIDTH, CARD_HEIGHT))
                        self.card_images[(color_code, number)] = image
                    except FileNotFoundError:
                        pass
            try:
                self.back_image = pygame.image.load("images/back.png").convert_alpha()
                self.back_image = pygame.transform.scale(self.back_image, (CARD_WIDTH, CARD_HEIGHT))
            except FileNotFoundError:
                pass
        except Exception as e:
            print(f"Error initializing images: {e}")

    def draw_card(self, card: Optional[Card], x, y, face_up=True):
        if not face_up:
            if self.back_image:
                self.screen.blit(self.back_image, (x, y))
            else:
                pygame.draw.rect(self.screen, (0, 0, 0), (x, y, CARD_WIDTH, CARD_HEIGHT))
                pygame.draw.rect(self.screen, (255, 255, 255), (x+2, y+2, CARD_WIDTH-4, CARD_HEIGHT-4), 2)
            return

        if card in self.card_images:
            self.screen.blit(self.card_images[card], (x, y))
        else:
            color_code, number = card
            color_rgb = {'R': (255, 50, 50), 'Y': (255, 255, 0), 'G': (50, 200, 50), 'B': (50, 50, 255)}[color_code]
            pygame.draw.rect(self.screen, color_rgb, (x, y, CARD_WIDTH, CARD_HEIGHT))
            text = self.large_font.render(str(number), True, (0, 0, 0))
            self.screen.blit(text, (x + 35, y + 50))

    def draw_game_state(self, game: Uno, message: str = "") -> Tuple[List[Tuple[Card, pygame.Rect]], pygame.Rect]:
        """Draws the current frame and returns interactable rects."""
        self.screen.fill(BG_COLOR)
        
        # 1. Opponent Hand
        opp_hand_size = len(game.H_2)
        start_x_opp = (SCREEN_WIDTH - (opp_hand_size * 40)) // 2
        for i in range(opp_hand_size):
            self.draw_card(None, start_x_opp + i * 40, 50, face_up=False)
        
        # 2. Center Piles
        center_y = SCREEN_HEIGHT // 2 - CARD_HEIGHT // 2
        
        # Deck
        self.draw_card(None, SCREEN_WIDTH // 2 - CARD_WIDTH - 20, center_y, face_up=False)
        deck_text = self.font.render("Draw", True, TEXT_COLOR)
        self.screen.blit(deck_text, (SCREEN_WIDTH // 2 - CARD_WIDTH - 10, center_y + CARD_HEIGHT + 10))
        deck_rect = pygame.Rect(SCREEN_WIDTH // 2 - CARD_WIDTH - 20, center_y, CARD_WIDTH, CARD_HEIGHT)

        # Discard Pile
        if game.P_t:
            self.draw_card(game.P_t, SCREEN_WIDTH // 2 + 20, center_y, face_up=True)
        
        # 3. Human Hand
        hand = game.H_1
        hand_rects = []
        hand_size = len(hand)
        spacing = 60 if hand_size < 15 else 40
        total_width = (hand_size - 1) * spacing + CARD_WIDTH
        start_x = (SCREEN_WIDTH - total_width) // 2
        start_y = SCREEN_HEIGHT - CARD_HEIGHT - 30
        
        mouse_x, mouse_y = pygame.mouse.get_pos()

        for i, card in enumerate(hand):
            x = start_x + i * spacing
            y = start_y
            
            # Hover Effect
            card_rect = pygame.Rect(x, y, CARD_WIDTH, CARD_HEIGHT)
            if card_rect.collidepoint(mouse_x, mouse_y):
                y -= 20
                card_rect.y = y 
            
            self.draw_card(card, x, y, face_up=True)
            hand_rects.append((card, card_rect))

        # 4. Message
        if message:
            msg_surf = self.font.render(message, True, TEXT_COLOR)
            self.screen.blit(msg_surf, (20, SCREEN_HEIGHT - 30))

        pygame.display.flip()
        return hand_rects, deck_rect

    def get_input_action(self, game: Uno, player_id: int) -> Action:
        """
        Blocking input loop: waits for user to click a valid action.
        """
        legal_actions = game.get_legal_actions(player_id)
        
        # Determine prompt
        if len(legal_actions) == 1 and legal_actions[0].is_draw():
            status_msg = "No playable cards. Click Deck to Draw."
        else:
            status_msg = "Your Turn! Select a card."

        while True:
            self.clock.tick(FPS)
            
            # Use draw_game_state here
            card_rects, deck_rect = self.draw_game_state(game, status_msg)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    pos = event.pos
                    
                    # Check Hand Clicks (Reverse order for layering)
                    for i in range(len(card_rects) - 1, -1, -1):
                        card, rect = card_rects[i]
                        if rect.collidepoint(pos):
                            for act in legal_actions:
                                if act.is_play() and act.X_1 == card:
                                    return act
                            status_msg = "Invalid Move! Card doesn't match."
                            break 
                    
                    # Check Deck Click
                    if deck_rect.collidepoint(pos):
                        for act in legal_actions:
                            if act.is_draw():
                                return act
                        status_msg = "You cannot draw if you have a playable card."
