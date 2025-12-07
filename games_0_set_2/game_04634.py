
# Generated: 2025-08-28T02:59:11.239846
# Source Brief: brief_04634.md
# Brief Index: 4634

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Space to flip a card."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist memory-matching game. Find all the pairs before you make 3 mistakes."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_ROWS = 4
    GRID_COLS = 5
    CARD_COUNT = GRID_ROWS * GRID_COLS
    PAIR_COUNT = CARD_COUNT // 2
    MAX_MISTAKES = 3
    MAX_STEPS = 1000

    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (50, 60, 70)
    COLOR_CARD_BACK = (70, 80, 100)
    COLOR_CARD_FACE = (220, 220, 230)
    COLOR_CURSOR = (255, 180, 0)
    COLOR_TEXT = (240, 240, 240)
    COLOR_SYMBOL = (10, 10, 10)
    COLOR_MATCH = (40, 200, 80)
    COLOR_MISMATCH = (220, 50, 50)
    COLOR_MATCHED_TINT = (20, 30, 40, 100) # RGBA

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_ui = pygame.font.SysFont("consola", 24)
        self.font_game_over = pygame.font.SysFont("consola", 60)

        # Card geometry
        padding = 20
        grid_width = self.SCREEN_WIDTH - 2 * padding
        grid_height = self.SCREEN_HEIGHT - 60  # Space for UI
        self.card_width = (grid_width - (self.GRID_COLS - 1) * 10) / self.GRID_COLS
        self.card_height = (grid_height - (self.GRID_ROWS - 1) * 10) / self.GRID_ROWS
        self.grid_start_x = padding
        self.grid_start_y = 50

        self.cards = []
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.mistakes_left = self.MAX_MISTAKES
        self.matched_pairs = 0

        self.cursor_pos = [0, 0] # col, row
        self.selected_cards_indices = []
        
        # Game states: PLAYER_TURN, ANIMATING, GAME_OVER
        self.game_state = "PLAYER_TURN"
        self.animation_timer = 0
        self.animation_flash_color = None

        self.last_space_press = False
        self.steps_since_revealed = {i: self.MAX_STEPS for i in range(self.CARD_COUNT)}
        
        self._initialize_cards()

        return self._get_observation(), self._get_info()
    
    def _initialize_cards(self):
        symbols = list(range(self.PAIR_COUNT)) * 2
        self.rng.shuffle(symbols)
        
        self.cards = []
        for i in range(self.CARD_COUNT):
            row = i // self.GRID_COLS
            col = i % self.GRID_COLS
            
            x = self.grid_start_x + col * (self.card_width + 10)
            y = self.grid_start_y + row * (self.card_height + 10)
            
            card = {
                "symbol": symbols[i],
                "state": "down", # 'down', 'up', 'matched'
                "rect": pygame.Rect(x, y, self.card_width, self.card_height),
                "flash_timer": 0
            }
            self.cards.append(card)

    def step(self, action):
        self.clock.tick(30) # Maintain 30 FPS for smooth animations
        
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        step_reward = 0
        self.steps += 1
        
        # Update timers and states
        for i in range(self.CARD_COUNT):
            self.steps_since_revealed[i] += 1
        
        if self.game_state == "ANIMATING":
            self.animation_timer -= 1
            if self.animation_timer <= 0:
                self.game_state = "PLAYER_TURN"
                # Flip mismatched cards back down
                card1_idx, card2_idx = self.selected_cards_indices
                self.cards[card1_idx]["state"] = "down"
                self.cards[card2_idx]["state"] = "down"
                self.selected_cards_indices = []

        elif self.game_state == "PLAYER_TURN":
            # Handle movement
            if movement == 1: # Up
                self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_ROWS) % self.GRID_ROWS
            elif movement == 2: # Down
                self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_ROWS
            elif movement == 3: # Left
                self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_COLS) % self.GRID_COLS
            elif movement == 4: # Right
                self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_COLS

            # Handle card selection on space press (rising edge)
            space_press = space_held and not self.last_space_press
            if space_press:
                cursor_idx = self.cursor_pos[1] * self.GRID_COLS + self.cursor_pos[0]
                card = self.cards[cursor_idx]

                if card["state"] == "down" and len(self.selected_cards_indices) < 2:
                    # Flip card
                    card["state"] = "up"
                    self.selected_cards_indices.append(cursor_idx)
                    
                    # Calculate reveal reward
                    if self.steps_since_revealed[cursor_idx] > 5:
                        step_reward += 0.1
                    else:
                        step_reward -= 0.2
                    self.steps_since_revealed[cursor_idx] = 0

                    # Check for pair
                    if len(self.selected_cards_indices) == 2:
                        idx1, idx2 = self.selected_cards_indices
                        card1, card2 = self.cards[idx1], self.cards[idx2]

                        if card1["symbol"] == card2["symbol"]: # Match
                            # SFX: Correct match
                            card1["state"] = "matched"
                            card2["state"] = "matched"
                            card1["flash_timer"] = 15 # 0.5s flash
                            card2["flash_timer"] = 15
                            self.selected_cards_indices = []
                            self.matched_pairs += 1
                            step_reward += 10.0
                        else: # Mismatch
                            # SFX: Incorrect match
                            self.mistakes_left -= 1
                            self.game_state = "ANIMATING"
                            self.animation_timer = 30 # 1s pause
                            self.animation_flash_color = self.COLOR_MISMATCH
                            card1["flash_timer"] = 30
                            card2["flash_timer"] = 30
                            step_reward -= 5.0
        
        self.last_space_press = space_held
        
        terminated = self._check_termination()
        if terminated:
            self.game_state = "GAME_OVER"
            if self.win:
                step_reward += 100.0
            else: # Loss or timeout
                step_reward -= 100.0

        self.score += step_reward

        return (
            self._get_observation(),
            step_reward,
            terminated,
            False,
            self._get_info()
        )

    def _check_termination(self):
        if self.matched_pairs == self.PAIR_COUNT:
            self.win = True
            self.game_over = True
            return True
        if self.mistakes_left <= 0:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "mistakes_left": self.mistakes_left,
            "matched_pairs": self.matched_pairs,
        }

    def _render_game(self):
        # Draw cards
        for i, card in enumerate(self.cards):
            if card["state"] == "down":
                pygame.draw.rect(self.screen, self.COLOR_CARD_BACK, card["rect"], border_radius=5)
            else: # 'up' or 'matched'
                pygame.draw.rect(self.screen, self.COLOR_CARD_FACE, card["rect"], border_radius=5)
                self._draw_symbol(self.screen, card["symbol"], card["rect"])
            
            # Flash effect for match/mismatch
            if card["flash_timer"] > 0:
                card["flash_timer"] -= 1
                flash_surface = pygame.Surface((card["rect"].width, card["rect"].height), pygame.SRCALPHA)
                
                color = self.COLOR_MATCH if card["state"] == "matched" else self.COLOR_MISMATCH
                alpha = int(200 * (card["flash_timer"] / 30.0))
                flash_surface.fill((*color, alpha))
                
                self.screen.blit(flash_surface, card["rect"].topleft)
            
            if card["state"] == "matched":
                # Add a subtle overlay to show it's solved
                s = pygame.Surface(card["rect"].size, pygame.SRCALPHA)
                s.fill(self.COLOR_MATCHED_TINT)
                self.screen.blit(s, card["rect"].topleft)

        # Draw cursor
        cursor_idx = self.cursor_pos[1] * self.GRID_COLS + self.cursor_pos[0]
        cursor_rect = self.cards[cursor_idx]["rect"]
        inflated_rect = cursor_rect.inflate(8, 8)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, inflated_rect, 3, border_radius=8)
    
    def _draw_symbol(self, surface, symbol_id, rect):
        center = rect.center
        size = min(rect.width, rect.height) * 0.35
        
        points = []
        if symbol_id == 0: # Circle
            pygame.gfxdraw.aacircle(surface, int(center[0]), int(center[1]), int(size), self.COLOR_SYMBOL)
            pygame.gfxdraw.filled_circle(surface, int(center[0]), int(center[1]), int(size), self.COLOR_SYMBOL)
        elif symbol_id == 1: # Square
            pygame.draw.rect(surface, self.COLOR_SYMBOL, (center[0] - size, center[1] - size, size * 2, size * 2))
        elif symbol_id == 2: # Triangle
            points = [(center[0], center[1] - size), (center[0] - size, center[1] + size), (center[0] + size, center[1] + size)]
            pygame.gfxdraw.aapolygon(surface, points, self.COLOR_SYMBOL)
            pygame.gfxdraw.filled_polygon(surface, points, self.COLOR_SYMBOL)
        elif symbol_id == 3: # X
            pygame.draw.line(surface, self.COLOR_SYMBOL, (center[0] - size, center[1] - size), (center[0] + size, center[1] + size), 7)
            pygame.draw.line(surface, self.COLOR_SYMBOL, (center[0] - size, center[1] + size), (center[0] + size, center[1] - size), 7)
        elif symbol_id == 4: # Diamond
            points = [(center[0], center[1] - size), (center[0] + size, center[1]), (center[0], center[1] + size), (center[0] - size, center[1])]
            pygame.gfxdraw.aapolygon(surface, points, self.COLOR_SYMBOL)
            pygame.gfxdraw.filled_polygon(surface, points, self.COLOR_SYMBOL)
        elif symbol_id == 5: # Plus
            pygame.draw.rect(surface, self.COLOR_SYMBOL, (center[0] - size, center[1] - size/4, size*2, size/2))
            pygame.draw.rect(surface, self.COLOR_SYMBOL, (center[0] - size/4, center[1] - size, size/2, size*2))
        elif symbol_id == 6: # Star
            points = []
            for i in range(10):
                angle = math.pi / 5 * i - math.pi / 2
                r = size if i % 2 == 0 else size * 0.4
                points.append((center[0] + r * math.cos(angle), center[1] + r * math.sin(angle)))
            pygame.gfxdraw.aapolygon(surface, points, self.COLOR_SYMBOL)
            pygame.gfxdraw.filled_polygon(surface, points, self.COLOR_SYMBOL)
        elif symbol_id == 7: # Hexagon
            points = [(center[0] + size * math.cos(2*math.pi/6 * i), center[1] + size * math.sin(2*math.pi/6 * i)) for i in range(6)]
            pygame.gfxdraw.aapolygon(surface, points, self.COLOR_SYMBOL)
            pygame.gfxdraw.filled_polygon(surface, points, self.COLOR_SYMBOL)
        elif symbol_id == 8: # Pentagon
            points = [(center[0] + size * math.cos(2*math.pi/5 * i - math.pi/2), center[1] + size * math.sin(2*math.pi/5 * i - math.pi/2)) for i in range(5)]
            pygame.gfxdraw.aapolygon(surface, points, self.COLOR_SYMBOL)
            pygame.gfxdraw.filled_polygon(surface, points, self.COLOR_SYMBOL)
        elif symbol_id == 9: # Inverted Triangle
            points = [(center[0], center[1] + size), (center[0] - size, center[1] - size), (center[0] + size, center[1] - size)]
            pygame.gfxdraw.aapolygon(surface, points, self.COLOR_SYMBOL)
            pygame.gfxdraw.filled_polygon(surface, points, self.COLOR_SYMBOL)
            
    def _render_ui(self):
        mistakes_text = f"Mistakes Left: {self.mistakes_left}"
        text_surface = self.font_ui.render(mistakes_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (15, 15))
        
        pairs_text = f"Pairs Found: {self.matched_pairs} / {self.PAIR_COUNT}"
        pairs_surface = self.font_ui.render(pairs_text, True, self.COLOR_TEXT)
        self.screen.blit(pairs_surface, (self.SCREEN_WIDTH - pairs_surface.get_width() - 15, 15))

        if self.game_state == "GAME_OVER":
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win:
                msg = "YOU WIN!"
                color = self.COLOR_MATCH
            else:
                msg = "GAME OVER"
                color = self.COLOR_MISMATCH
                
            game_over_surface = self.font_game_over.render(msg, True, color)
            pos_x = self.SCREEN_WIDTH / 2 - game_over_surface.get_width() / 2
            pos_y = self.SCREEN_HEIGHT / 2 - game_over_surface.get_height() / 2
            self.screen.blit(game_over_surface, (pos_x, pos_y))
            
    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to "dummy" to run headlessly
    import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv()
    obs, info = env.reset()
    
    # --- To run with a human player ---
    pygame.display.set_caption(env.game_description)
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0
    
    # Map Pygame keys to our action space
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    while running:
        movement = 0
        space_held = 0
        
        # Get user input
        keys = pygame.key.get_pressed()
        for key, move_action in key_to_action.items():
            if keys[key]:
                movement = move_action
                break # Prioritize one move action
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        
        action = np.array([movement, space_held, 0]) # Shift is not used
        
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- Game Reset ---")
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished. Final Score: {total_reward}")
            print("Press 'R' to reset.")

    env.close()
    pygame.quit()