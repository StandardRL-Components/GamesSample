
# Generated: 2025-08-28T03:32:34.836628
# Source Brief: brief_02057.md
# Brief Index: 2057

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press Space to paint the selected square."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Recreate the target pixel art image by painting a grid, square by square. Race against the clock to achieve 90% accuracy!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 10
    MAX_STEPS = 600
    WIN_ACCURACY = 0.9

    # --- Visuals ---
    COLOR_BG = (20, 25, 40)
    COLOR_GRID_LINE = (40, 50, 70)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_BG = (30, 35, 55)
    
    PALETTE = [
        (80, 90, 110),    # 0: Blank
        (230, 70, 80),    # 1: Red
        (250, 210, 90),   # 2: Yellow
    ]

    # --- Target Image ---
    TARGET_IMAGE = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 1, 0, 0, 2, 1, 0, 0],
        [0, 2, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.uint8)


    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_title = pygame.font.Font(None, 20)
        
        # Game grid layout
        self.cell_size = 32
        self.grid_width = self.GRID_SIZE * self.cell_size
        self.grid_height = self.GRID_SIZE * self.cell_size
        self.grid_x_start = (self.SCREEN_WIDTH - self.grid_width) // 2
        self.grid_y_start = (self.SCREEN_HEIGHT - self.grid_height) // 2 + 20
        
        # Initialize state variables
        self.grid = None
        self.cursor_pos = None
        self.steps = 0
        self.score = 0
        self.accuracy = 0.0
        self.game_over = False
        
        self.reset()

        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.uint8)
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.steps = 0
        self.score = 0
        self.accuracy = 0.0
        self.game_over = False
        
        self._calculate_accuracy()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        # Handle Movement
        if movement == 1:  # Up
            self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_SIZE) % self.GRID_SIZE
        elif movement == 2:  # Down
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_SIZE
        elif movement == 3:  # Left
            self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_SIZE) % self.GRID_SIZE
        elif movement == 4:  # Right
            self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_SIZE

        # Handle Painting
        if space_held:
            # Sound effect placeholder: # sfx_paint.play()
            y, x = self.cursor_pos
            current_color_idx = self.grid[y, x]
            target_color_idx = self.TARGET_IMAGE[y, x]
            
            if current_color_idx != target_color_idx:
                self.grid[y, x] = target_color_idx
                reward += 0.1  # Reward for correcting a pixel
                self._calculate_accuracy()
            else:
                reward -= 0.01 # Small penalty for redundant painting

        # Update step counter
        self.steps += 1
        
        # Check termination conditions
        terminated = False
        if self.accuracy >= self.WIN_ACCURACY:
            terminated = True
            reward += 100  # Win bonus
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            reward -= 100  # Loss penalty for running out of time
            self.game_over = True
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _calculate_accuracy(self):
        correct_pixels = np.sum(self.grid == self.TARGET_IMAGE)
        total_pixels = self.GRID_SIZE * self.GRID_SIZE
        self.accuracy = correct_pixels / total_pixels
        self.score = self.accuracy * 100

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render Target Image Preview
        self._render_target_preview()

        # Render Main Grid
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                color_idx = self.grid[y, x]
                color = self.PALETTE[color_idx]
                rect = pygame.Rect(
                    self.grid_x_start + x * self.cell_size,
                    self.grid_y_start + y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINE, rect, 1)

        # Render Cursor
        cursor_y, cursor_x = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.grid_x_start + cursor_x * self.cell_size,
            self.grid_y_start + cursor_y * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3) # Thick border

    def _render_target_preview(self):
        preview_cell_size = 8
        preview_size = self.GRID_SIZE * preview_cell_size
        preview_x_start = 30
        preview_y_start = 30
        
        # Title
        self._draw_text("TARGET", (preview_x_start + preview_size // 2, preview_y_start - 10), self.font_title, self.COLOR_UI_TEXT)

        # Grid
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                color_idx = self.TARGET_IMAGE[y, x]
                color = self.PALETTE[color_idx]
                rect = pygame.Rect(
                    preview_x_start + x * preview_cell_size,
                    preview_y_start + y * preview_cell_size,
                    preview_cell_size,
                    preview_cell_size
                )
                pygame.draw.rect(self.screen, color, rect)

    def _render_ui(self):
        # UI Panel
        ui_panel_rect = pygame.Rect(0, self.SCREEN_HEIGHT - 40, self.SCREEN_WIDTH, 40)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_panel_rect)
        pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (0, self.SCREEN_HEIGHT-40), (self.SCREEN_WIDTH, self.SCREEN_HEIGHT-40), 1)

        # Accuracy Text
        accuracy_text = f"ACCURACY: {self.accuracy:.1%}"
        self._draw_text(accuracy_text, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 20), self.font_ui, self.COLOR_UI_TEXT)

        # Time Text
        time_left = self.MAX_STEPS - self.steps
        time_text = f"TIME: {time_left}"
        self._draw_text(time_text, (self.SCREEN_WIDTH - 80, self.SCREEN_HEIGHT - 20), self.font_ui, self.COLOR_UI_TEXT)

        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = "SUCCESS!" if self.accuracy >= self.WIN_ACCURACY else "TIME UP!"
            self._draw_text(end_text, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 20), pygame.font.Font(None, 60), self.COLOR_CURSOR)
            
    def _draw_text(self, text, pos, font, color):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=pos)
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "accuracy": self.accuracy,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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