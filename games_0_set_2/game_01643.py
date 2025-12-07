
# Generated: 2025-08-28T02:14:25.265357
# Source Brief: brief_01643.md
# Brief Index: 1643

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = "Controls: Use arrow keys to select a plot. Press Space to plant/harvest. Hold Shift to sell harvested crops."

    # Must be a short, user-facing description of the game:
    game_description = "Manage your farm to earn 1000 gold before the timer runs out. Plant seeds, wait for them to grow, harvest, and sell for profit."

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.CELL_SIZE = 32
        self.GRID_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_X_OFFSET = (self.SCREEN_WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y_OFFSET = (self.SCREEN_HEIGHT - self.GRID_HEIGHT) // 2 + 20

        # Game parameters
        self.START_GOLD = 50
        self.START_TIMER = 600
        self.WIN_GOLD = 1000
        self.PLANT_COST = 5
        self.SELL_PRICE = 10
        self.CROP_GROW_TIME = 5 # 5 steps to mature
        self.CROP_STATE_EMPTY = 0
        self.CROP_STATE_READY = self.CROP_GROW_TIME + 1
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_SOIL = (87, 56, 42)
        self.COLOR_GRID_LINES = (70, 45, 30)
        self.COLOR_SELECTOR = (255, 255, 0, 150)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_GOLD = (255, 215, 0)
        self.COLOR_TIMER = (255, 100, 100)
        self.COLOR_SELL_BIN = (139, 69, 19)
        self.COLOR_SELL_BIN_LID = (160, 82, 45)
        
        # Crop colors [sprout, ...growing..., ready]
        self.CROP_COLORS = [
            (102, 153, 0),    # Stage 1
            (85, 128, 0),     # Stage 2
            (68, 102, 0),     # Stage 3
            (51, 77, 0),      # Stage 4
            (34, 51, 0),      # Stage 5
            (255, 200, 0)     # Ready
        ]

        # --- Gym Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)
        
        # --- State Variables ---
        self.np_random = None
        self.selector_pos = [0, 0]
        self.farm_grid = None
        self.gold = 0
        self.timer = 0
        self.harvested_crops = 0
        self.game_over = False
        self.feedback_text = ""
        self.feedback_timer = 0
        self.feedback_color = self.COLOR_TEXT
        
        # Initialize state
        self.reset()
        
        # --- Validation ---
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.selector_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.farm_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.gold = self.START_GOLD
        self.timer = self.START_TIMER
        self.harvested_crops = 0
        self.game_over = False
        self.feedback_text = ""
        self.feedback_timer = 0
        self.steps = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        # Unpack action
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        # 1. Update Crop Growth (Time passes every step)
        growing_plots = self.farm_grid > 0
        ready_plots = self.farm_grid == self.CROP_STATE_READY
        self.farm_grid[growing_plots & ~ready_plots] += 1
        
        # 2. Handle Player Input
        # Movement
        if movement == 1: self.selector_pos[0] = (self.selector_pos[0] - 1) % self.GRID_SIZE # Up
        elif movement == 2: self.selector_pos[0] = (self.selector_pos[0] + 1) % self.GRID_SIZE # Down
        elif movement == 3: self.selector_pos[1] = (self.selector_pos[1] - 1) % self.GRID_SIZE # Left
        elif movement == 4: self.selector_pos[1] = (self.selector_pos[1] + 1) % self.GRID_SIZE # Right
        
        sel_r, sel_c = self.selector_pos
        plot_state = self.farm_grid[sel_r, sel_c]

        # Space Action (Plant/Harvest)
        if space_pressed:
            if plot_state == self.CROP_STATE_EMPTY and self.gold >= self.PLANT_COST:
                self.gold -= self.PLANT_COST
                self.farm_grid[sel_r, sel_c] = 1 # Start growing
                self._set_feedback(f"-{self.PLANT_COST}G: Planted!", self.COLOR_GOLD)
            elif plot_state == self.CROP_STATE_READY:
                self.farm_grid[sel_r, sel_c] = self.CROP_STATE_EMPTY
                self.harvested_crops += 1
                reward += 0.1
                self._set_feedback("Harvested!", (200, 255, 200))
            else:
                self._set_feedback("Cannot act here.", (255, 150, 150))
        
        # Shift Action (Sell)
        if shift_pressed:
            if self.harvested_crops > 0:
                earnings = self.harvested_crops * self.SELL_PRICE
                self.gold += earnings
                self.gold = min(self.gold, self.WIN_GOLD) # Cap gold at win condition
                reward += 1.0
                self._set_feedback(f"+{earnings}G: Sold crops!", self.COLOR_GOLD)
                self.harvested_crops = 0
            else:
                self._set_feedback("Nothing to sell.", (255, 150, 150))
        
        # 3. Update Game State
        self.timer -= 1
        if self.feedback_timer > 0:
            self.feedback_timer -= 1

        # 4. Check Termination
        terminated = False
        if self.gold >= self.WIN_GOLD:
            reward += 100
            terminated = True
            self.game_over = True
            self._set_feedback("YOU WIN!", self.COLOR_GOLD, 180)
        elif self.timer <= 0 or self.steps >= self.MAX_STEPS:
            reward -= 100
            terminated = True
            self.game_over = True
            self._set_feedback("TIME'S UP!", self.COLOR_TIMER, 180)
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _set_feedback(self, text, color, duration=30):
        self.feedback_text = text
        self.feedback_color = color
        self.feedback_timer = duration

    def _get_info(self):
        return {
            "score": self.gold, # Use gold as the primary score metric
            "steps": self.steps,
            "timer": self.timer,
            "harvested_crops": self.harvested_crops
        }
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render farm grid and crops
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                rect = pygame.Rect(
                    self.GRID_X_OFFSET + c * self.CELL_SIZE,
                    self.GRID_Y_OFFSET + r * self.CELL_SIZE,
                    self.CELL_SIZE,
                    self.CELL_SIZE
                )
                # Draw soil plot
                pygame.draw.rect(self.screen, self.COLOR_SOIL, rect)
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINES, rect, 1)

                # Draw crop
                plot_state = self.farm_grid[r, c]
                if plot_state > self.CROP_STATE_EMPTY:
                    stage_index = min(plot_state - 1, len(self.CROP_COLORS) - 1)
                    color = self.CROP_COLORS[stage_index]
                    
                    # Visual representation of growth
                    if plot_state == self.CROP_STATE_READY:
                        # Ready crop is large and fills the cell
                        pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, self.CELL_SIZE // 2 - 4, color)
                        pygame.gfxdraw.aacircle(self.screen, rect.centerx, rect.centery, self.CELL_SIZE // 2 - 4, color)
                    else:
                        # Growing crop gets bigger
                        radius = int(2 + (plot_state / self.CROP_GROW_TIME) * (self.CELL_SIZE // 2 - 6))
                        pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, radius, color)
        
        # Render selector
        sel_r, sel_c = self.selector_pos
        selector_rect = pygame.Rect(
            self.GRID_X_OFFSET + sel_c * self.CELL_SIZE,
            self.GRID_Y_OFFSET + sel_r * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE
        )
        # Use a surface for transparency
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2 # 0 to 1
        alpha = 100 + pulse * 100
        pygame.draw.rect(s, (*self.COLOR_SELECTOR[:3], alpha), (0, 0, self.CELL_SIZE, self.CELL_SIZE), 4)
        self.screen.blit(s, selector_rect.topleft)

    def _render_ui(self):
        # --- Top Bar ---
        top_bar_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, 50)
        pygame.draw.rect(self.screen, (10, 20, 30), top_bar_rect)
        pygame.draw.line(self.screen, (60, 70, 80), (0, 50), (self.SCREEN_WIDTH, 50), 2)

        # Gold Display
        gold_text = self.font_large.render(f"Gold: {self.gold}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (self.SCREEN_WIDTH - gold_text.get_width() - 20, 10))

        # Timer Display
        timer_text = self.font_large.render(f"Time: {max(0, self.timer)}", True, self.COLOR_TIMER)
        self.screen.blit(timer_text, (20, 10))

        # --- Bottom Bar (Inventory/Sell Area) ---
        bottom_bar_y = self.GRID_Y_OFFSET + self.GRID_HEIGHT + 10
        
        # Harvested Crops Display
        inv_text = self.font_medium.render(f"Harvested: {self.harvested_crops}", True, self.COLOR_TEXT)
        self.screen.blit(inv_text, (self.GRID_X_OFFSET, bottom_bar_y))
        
        # Sell Bin Visual
        sell_bin_rect = pygame.Rect(self.SCREEN_WIDTH - self.GRID_X_OFFSET - 80, bottom_bar_y - 5, 80, 30)
        pygame.draw.rect(self.screen, self.COLOR_SELL_BIN, sell_bin_rect, border_bottom_left_radius=5, border_bottom_right_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_SELL_BIN_LID, (sell_bin_rect.x, sell_bin_rect.y, sell_bin_rect.width, 10), border_top_left_radius=5, border_top_right_radius=5)
        sell_text = self.font_small.render("SELL (SHIFT)", True, self.COLOR_TEXT)
        self.screen.blit(sell_text, (sell_bin_rect.centerx - sell_text.get_width() // 2, sell_bin_rect.centery - sell_text.get_height() // 2 + 5))

        # Feedback Text
        if self.feedback_timer > 0:
            alpha = int(255 * (self.feedback_timer / 30)) # Fade out
            feedback_surf = self.font_medium.render(self.feedback_text, True, self.feedback_color)
            feedback_surf.set_alpha(alpha)
            pos_x = self.SCREEN_WIDTH / 2 - feedback_surf.get_width() / 2
            pos_y = self.GRID_Y_OFFSET - 30
            self.screen.blit(feedback_surf, (pos_x, pos_y))

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

        # Test game logic assertions from brief
        self.reset()
        self.gold = self.WIN_GOLD + 100
        sell_action = [0, 0, 1] # Sell action
        self.harvested_crops = 1
        self.step(sell_action)
        assert self.gold <= self.WIN_GOLD, f"Gold exceeded WIN_GOLD: {self.gold}"

        self.reset()
        self.timer = 1
        self.step(self.action_space.sample())
        assert self.timer >= 0, f"Timer went below 0: {self.timer}"

        self.reset()
        self.farm_grid[0,0] = 1 # Not ready
        self.selector_pos = [0,0]
        harvest_action = [0, 1, 0] # Select [0,0], press space
        self.step(harvest_action)
        assert self.harvested_crops == 0, "Harvested a crop that was not ready"
        
        print("✓ Implementation validated successfully")