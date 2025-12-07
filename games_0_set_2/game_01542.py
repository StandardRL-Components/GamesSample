
# Generated: 2025-08-28T00:02:40.385574
# Source Brief: brief_01542.md
# Brief Index: 1542

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


# Helper class for Card state and animation
class Card:
    def __init__(self, pattern_id, color):
        self.pattern_id = pattern_id
        self.color = color
        self.state = "hidden"  # hidden, revealed, matched
        self.flip_progress = 0.0  # 0.0 (hidden) to 1.0 (revealed)
        self.flip_speed = 0.2
        self.is_flipping_up = False
        self.is_flipping_down = False
        self.match_glow_alpha = 0

    def update(self):
        if self.is_flipping_up:
            self.flip_progress = min(1.0, self.flip_progress + self.flip_speed)
            if self.flip_progress == 1.0:
                self.is_flipping_up = False
        elif self.is_flipping_down:
            self.flip_progress = max(0.0, self.flip_progress - self.flip_speed)
            if self.flip_progress == 0.0:
                self.is_flipping_down = False
                self.state = "hidden"

        if self.match_glow_alpha > 0:
            self.match_glow_alpha = max(0, self.match_glow_alpha - 10)

    def reveal(self):
        if self.state == "hidden":
            self.state = "revealed"
            self.is_flipping_up = True
            self.is_flipping_down = False

    def hide(self):
        if self.state == "revealed":
            self.is_flipping_down = True
            self.is_flipping_up = False
    
    def match(self):
        self.state = "matched"
        self.match_glow_alpha = 255


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = "Controls: Use arrow keys to move the cursor. Press space to flip a card."
    game_description = "A fast-paced memory game. Match all pairs of cards before the timer runs out!"

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and grid dimensions
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_ROWS, self.GRID_COLS = 4, 4
        self.CARD_SIZE = 70
        self.CARD_PADDING = 15
        self.GRID_WIDTH = self.GRID_COLS * (self.CARD_SIZE + self.CARD_PADDING) - self.CARD_PADDING
        self.GRID_HEIGHT = self.GRID_ROWS * (self.CARD_SIZE + self.CARD_PADDING) - self.CARD_PADDING
        self.GRID_X_OFFSET = (self.SCREEN_WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y_OFFSET = (self.SCREEN_HEIGHT - self.GRID_HEIGHT) // 2 + 20

        # Colors
        self.COLOR_BG = (20, 30, 70)
        self.COLOR_CARD_BACK = (60, 80, 120)
        self.COLOR_CARD_BACK_BORDER = (90, 110, 150)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_MATCH_GLOW = (100, 255, 100)
        self.COLOR_MISMATCH_FLASH = (255, 100, 100)
        self.PATTERN_COLORS = [
            (255, 128, 128), (128, 255, 128), (128, 128, 255), (255, 255, 128),
            (255, 128, 255), (128, 255, 255), (255, 192, 128), (192, 128, 255)
        ]

        # Game constants
        self.MAX_TIME = 120  # seconds
        self.MAX_STEPS = 30 * self.MAX_TIME # 30fps
        self.MISMATCH_DELAY_FRAMES = 15 # 0.5 seconds at 30fps

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # State variables (initialized in reset)
        self.grid = []
        self.cursor_pos = [0, 0]
        self.flipped_cards_pos = []
        self.score = 0
        self.steps = 0
        self.time_remaining = 0
        self.mismatch_timer = 0
        self.game_over = False
        self.prev_space_held = False
        self.mismatch_flash_alpha = 0
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.time_remaining = self.MAX_TIME
        self.game_over = False
        self.cursor_pos = [0, 0]
        self.flipped_cards_pos = []
        self.mismatch_timer = 0
        self.prev_space_held = False
        self.mismatch_flash_alpha = 0

        # Create and shuffle card patterns
        num_pairs = (self.GRID_ROWS * self.GRID_COLS) // 2
        pattern_ids = list(range(num_pairs)) * 2
        self.np_random.shuffle(pattern_ids)

        self.grid = []
        for r in range(self.GRID_ROWS):
            row = []
            for c in range(self.GRID_COLS):
                pattern_id = pattern_ids.pop()
                color = self.PATTERN_COLORS[pattern_id]
                row.append(Card(pattern_id, color))
            self.grid.append(row)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        self.game_over = False

        # --- Handle Time and Animations ---
        self.clock.tick(30)
        self.time_remaining -= 1/30
        
        if self.mismatch_timer > 0:
            self.mismatch_timer -= 1
            if self.mismatch_timer == 0:
                card1_pos = self.flipped_cards_pos[0]
                card2_pos = self.flipped_cards_pos[1]
                self.grid[card1_pos[0]][card1_pos[1]].hide()
                self.grid[card2_pos[0]][card2_pos[1]].hide()
                self.flipped_cards_pos = []
                self.mismatch_flash_alpha = 150 # Start flash
                
        if self.mismatch_flash_alpha > 0:
            self.mismatch_flash_alpha = max(0, self.mismatch_flash_alpha - 15)

        for row in self.grid:
            for card in row:
                card.update()

        # --- Handle Actions ---
        movement, space_held, _ = action
        space_pressed = space_held and not self.prev_space_held
        self.prev_space_held = space_held

        # Movement
        if movement == 1: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1) # Up
        elif movement == 2: self.cursor_pos[0] = min(self.GRID_ROWS - 1, self.cursor_pos[0] + 1) # Down
        elif movement == 3: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1) # Left
        elif movement == 4: self.cursor_pos[1] = min(self.GRID_COLS - 1, self.cursor_pos[1] + 1) # Right

        # Flip Card
        if space_pressed:
            card = self.grid[self.cursor_pos[0]][self.cursor_pos[1]]
            can_flip = len(self.flipped_cards_pos) < 2 and self.mismatch_timer == 0
            
            if card.state == "hidden" and can_flip:
                card.reveal()
                self.flipped_cards_pos.append(list(self.cursor_pos))
                reward += 0.1 # Reward for revealing a card
                #_play_flip_sound()
            elif card.state != "hidden":
                reward -= 0.01 # Small penalty for clicking revealed card

        # --- Game Logic: Check for Match ---
        if len(self.flipped_cards_pos) == 2:
            pos1 = self.flipped_cards_pos[0]
            pos2 = self.flipped_cards_pos[1]
            card1 = self.grid[pos1[0]][pos1[1]]
            card2 = self.grid[pos2[0]][pos2[1]]

            if card1.pattern_id == card2.pattern_id:
                card1.match()
                card2.match()
                self.score += 1
                reward += 10 # Reward for a match
                self.flipped_cards_pos = []
                #_play_match_sound()
            else:
                self.mismatch_timer = self.MISMATCH_DELAY_FRAMES
                reward -= 1 # Penalty for a mismatch
                #_play_mismatch_sound()

        # --- Check Termination Conditions ---
        if self.score == (self.GRID_ROWS * self.GRID_COLS) // 2:
            self.game_over = True
            reward += 100 # Win bonus
        elif self.time_remaining <= 0:
            self.game_over = True
            reward -= 50 # Time out penalty
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            self.game_over,
            False,
            self._get_info()
        )
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw cards
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                card = self.grid[r][c]
                if card.state == 'matched' and card.match_glow_alpha == 0:
                    continue # Don't draw fully matched cards

                x = self.GRID_X_OFFSET + c * (self.CARD_SIZE + self.CARD_PADDING)
                y = self.GRID_Y_OFFSET + r * (self.CARD_SIZE + self.CARD_PADDING)
                
                self._render_card(self.screen, pygame.Rect(x, y, self.CARD_SIZE, self.CARD_SIZE), card)
        
        # Draw cursor
        cursor_x = self.GRID_X_OFFSET + self.cursor_pos[1] * (self.CARD_SIZE + self.CARD_PADDING) - 5
        cursor_y = self.GRID_Y_OFFSET + self.cursor_pos[0] * (self.CARD_SIZE + self.CARD_PADDING) - 5
        cursor_rect = pygame.Rect(cursor_x, cursor_y, self.CARD_SIZE + 10, self.CARD_SIZE + 10)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, 5)

        # Mismatch flash overlay
        if self.mismatch_flash_alpha > 0:
            flash_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            flash_surface.fill((*self.COLOR_MISMATCH_FLASH, self.mismatch_flash_alpha))
            self.screen.blit(flash_surface, (0, 0))

    def _render_card(self, surface, rect, card):
        # Card flip animation
        progress = card.flip_progress
        center_x = rect.centerx
        
        # Width shrinks and grows based on cosine
        display_width = rect.width * math.cos(progress * math.pi)
        display_rect = pygame.Rect(0, 0, abs(display_width), rect.height)
        display_rect.center = rect.center
        
        card_surface = pygame.Surface(display_rect.size, pygame.SRCALPHA)

        # Determine if we're drawing the front or back
        is_front = progress > 0.5

        if is_front:
            # Draw pattern
            pygame.draw.rect(card_surface, card.color, (0, 0, *display_rect.size), border_radius=8)
            pattern_rect = card_surface.get_rect().inflate(-20, -20)
            self._draw_pattern(card_surface, card.pattern_id, pattern_rect)
        else:
            # Draw back
            pygame.draw.rect(card_surface, self.COLOR_CARD_BACK, (0, 0, *display_rect.size), border_radius=8)
            pygame.draw.rect(card_surface, self.COLOR_CARD_BACK_BORDER, (0, 0, *display_rect.size), 3, 8)

        surface.blit(card_surface, display_rect)

        # Match glow effect
        if card.match_glow_alpha > 0:
            glow_color = (*self.COLOR_MATCH_GLOW, card.match_glow_alpha)
            glow_radius = int(rect.width * 0.75 * (1 - card.match_glow_alpha / 255))
            pygame.gfxdraw.aacircle(surface, rect.centerx, rect.centery, glow_radius, glow_color)
            pygame.gfxdraw.aacircle(surface, rect.centerx, rect.centery, glow_radius-1, glow_color)

    def _draw_pattern(self, surface, pattern_id, rect):
        color = (255, 255, 255)
        cx, cy = rect.centerx, rect.centery
        w, h = rect.width, rect.height
        
        # Ensure points are integers for drawing
        points = lambda pts: [(int(p[0]), int(p[1])) for p in pts]

        if pattern_id == 0: # Circle
            pygame.gfxdraw.filled_circle(surface, cx, cy, int(w/2.5), color)
        elif pattern_id == 1: # Square
            pygame.draw.rect(surface, color, rect.inflate(-w/4, -h/4))
        elif pattern_id == 2: # Triangle
            pygame.draw.polygon(surface, color, points([(cx, rect.top), (rect.right, rect.bottom), (rect.left, rect.bottom)]))
        elif pattern_id == 3: # Cross
            pygame.draw.rect(surface, color, (rect.left, cy - h/8, w, h/4))
            pygame.draw.rect(surface, color, (cx - w/8, rect.top, w/4, h))
        elif pattern_id == 4: # Star
            num_points = 5
            angle = 2 * math.pi / num_points
            outer_radius, inner_radius = w / 2, w / 4
            star_points = []
            for i in range(num_points * 2):
                r = outer_radius if i % 2 == 0 else inner_radius
                current_angle = i * angle / 2 - math.pi / 2
                star_points.append((cx + r * math.cos(current_angle), cy + r * math.sin(current_angle)))
            pygame.draw.polygon(surface, color, points(star_points))
        elif pattern_id == 5: # Diamond
            pygame.draw.polygon(surface, color, points([(cx, rect.top), (rect.right, cy), (cx, rect.bottom), (rect.left, cy)]))
        elif pattern_id == 6: # C-shape
            pygame.draw.rect(surface, color, (rect.left, rect.top, w/3, h))
            pygame.draw.rect(surface, color, (rect.left, rect.top, w, h/3))
            pygame.draw.rect(surface, color, (rect.left, rect.bottom - h/3, w, h/3))
        elif pattern_id == 7: # X-shape
            pygame.draw.line(surface, color, rect.topleft, rect.bottomright, int(w/4))
            pygame.draw.line(surface, color, rect.topright, rect.bottomleft, int(w/4))

    def _render_ui(self):
        # Render Score
        score_text = self.font_large.render(f"Pairs: {self.score}/8", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 15))

        # Render Timer
        time_str = f"Time: {max(0, int(self.time_remaining)):02d}"
        time_color = (255, 100, 100) if self.time_remaining < 10 else self.COLOR_TEXT
        time_text = self.font_large.render(time_str, True, time_color)
        time_rect = time_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 15))
        self.screen.blit(time_text, time_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "cursor_pos": self.cursor_pos
        }
        
    def close(self):
        pygame.quit()

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
if __name__ == "__main__":
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless for the check
    env = GameEnv()
    env.close()

    # To visualize and play the game, comment out the os.environ line above
    # and run the following code:
    
    # env = GameEnv(render_mode="rgb_array")
    # obs, info = env.reset()
    # screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    # pygame.display.set_caption("Memory Match")
    # running = True
    # total_reward = 0
    
    # while running:
    #     action = [0, 0, 0] # Default to no-op
    #     keys = pygame.key.get_pressed()
        
    #     if keys[pygame.K_UP]: action[0] = 1
    #     elif keys[pygame.K_DOWN]: action[0] = 2
    #     elif keys[pygame.K_LEFT]: action[0] = 3
    #     elif keys[pygame.K_RIGHT]: action[0] = 4
        
    #     if keys[pygame.K_SPACE]: action[1] = 1
    #     if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     total_reward += reward
        
    #     # Render to the display window
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
        
    #     if terminated or truncated:
    #         print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    #         obs, info = env.reset()
    #         total_reward = 0
        
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False
                
    # env.close()